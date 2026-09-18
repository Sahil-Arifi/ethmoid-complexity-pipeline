"""Durable local uploads, complete catalog ingestion and explicit analysis intents."""
from __future__ import annotations

import errno
import hashlib
import os
from pathlib import PurePosixPath
import shutil
import stat
import threading
import uuid
import zipfile

import pydicom
from pydicom.errors import InvalidDicomError

from ..dicom_frames import frame_headers
from ..util import sha256_file, utc_now, validate_roots
from .catalog import fingerprint_hashes, safe_descriptor
from .service import WebAppError
from .store import encode_json

MAX_CHUNK = 8 * 1024 ** 2
REASONS = {
    "DISK_SPACE": "The protected drive needs more free space. The original upload is retained; retry after freeing space.",
    "ARCHIVE_INVALID": "The archive is damaged, encrypted, or uses an unsupported compression method.",
    "ARCHIVE_UNSAFE": "The archive contains an unsafe path or link and was not imported.",
    "UPLOAD_QUOTA": "The upload exceeds the configured expanded-data or file-count limit.",
    "INDEXING_ERROR": "Indexing could not finish. The original upload is retained and can be retried.",
    "NO_DICOM": "No readable DICOM studies were found. Every uploaded file is retained for review.",
    "ANCILLARY": "This file is not a DICOM image; it is retained without counting.",
    "UNREADABLE": "This DICOM could not be read or lacks study, series, or image identity.",
    "SOURCE_CONFLICT": "This DICOM identity already exists with different source bytes. Existing work is preserved; this upload requires review.",
    "UNSUPPORTED_INPUT": "No uploaded series passed three-dimensional preparation.",
    "COUNT_SCHEDULING": "Analysis could not be scheduled. The verified volume is available for an explicit Run.",
    "WORKER_STORAGE": "The upload worker is waiting for local storage to recover. Saved uploads are retained.",
}


class UploadFailure(Exception):
    pass


class UploadService:
    def __init__(self, app, catalog, create_count_run, *, max_expanded_bytes=20 * 1024 ** 3, max_files=50000):
        self.app, self.store, self.catalog = app, app.store, catalog
        self.create_count_run = create_count_run
        self.max_expanded_bytes, self.max_files = max_expanded_bytes, max_files
        if catalog.config is None:
            from .catalog_config import load_local_config
            try:
                catalog.config = load_local_config(app.root)
            except Exception:
                catalog._status, catalog._error = "error", "CONFIGURATION"
        work = catalog.work_root.resolve()
        source_parent = work.parent if catalog.config or app.root.name != "webapp" else app.root.parent.parent
        self.source_root = source_parent / ("uploaded-dicom-" + hashlib.sha256(str(app.root).encode()).hexdigest()[:12])
        self.source_root.mkdir(parents=True, exist_ok=True)
        validate_roots(self.source_root, work)
        self._stop, self._wake = threading.Event(), threading.Event()
        self._thread = None
        self._lock_file = None
        self.last_error = None
        self._initialize()

    def _initialize(self):
        # Store has already backed up and migrated the application database.
        with self.store.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS upload_sessions (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL, source_relpath TEXT NOT NULL,
                    error_code TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS upload_files (
                    upload_id TEXT NOT NULL REFERENCES upload_sessions(id), file_index INTEGER NOT NULL,
                    name TEXT NOT NULL, expected_size INTEGER NOT NULL, received_bytes INTEGER NOT NULL DEFAULT 0,
                    sha256 TEXT, status TEXT NOT NULL DEFAULT 'pending', reason_code TEXT,
                    PRIMARY KEY(upload_id,file_index));
                CREATE TABLE IF NOT EXISTS upload_items (
                    upload_id TEXT NOT NULL REFERENCES upload_sessions(id), item_index INTEGER NOT NULL,
                    relative_path TEXT NOT NULL, original_name TEXT NOT NULL, sha256 TEXT NOT NULL,
                    study_uid TEXT, series_uid TEXT, sop_uid TEXT, status TEXT NOT NULL, reason_code TEXT,
                    PRIMARY KEY(upload_id,item_index));
                CREATE TABLE IF NOT EXISTS upload_studies (
                    upload_id TEXT NOT NULL REFERENCES upload_sessions(id), study_id TEXT NOT NULL REFERENCES catalog_studies(id),
                    has_new_series INTEGER NOT NULL DEFAULT 0, selected_series_id TEXT,
                    status TEXT NOT NULL DEFAULT 'preparing', reason_code TEXT, count_run_id TEXT,
                    PRIMARY KEY(upload_id,study_id));
                CREATE TABLE IF NOT EXISTS upload_series (
                    upload_id TEXT NOT NULL REFERENCES upload_sessions(id), series_id TEXT NOT NULL REFERENCES catalog_series(id),
                    disposition TEXT NOT NULL, PRIMARY KEY(upload_id,series_id));
                CREATE TABLE IF NOT EXISTS upload_analysis_intents (
                    series_id TEXT NOT NULL REFERENCES catalog_series(id), source_fingerprint TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE, count_run_id TEXT, status TEXT NOT NULL DEFAULT 'pending',
                    PRIMARY KEY(series_id,source_fingerprint));
            """)

    def _session(self, identity):
        with self.store.connect() as db:
            row = db.execute("SELECT * FROM upload_sessions WHERE id=?", (identity,)).fetchone()
        if row is None:
            raise WebAppError("Upload session not found.", status_code=404)
        return row

    def _directory(self, identity):
        row = self._session(identity)
        result = (self.source_root / row['source_relpath']).resolve()
        if not result.is_relative_to(self.source_root.resolve()):
            raise WebAppError("Upload storage is unavailable.", status_code=409)
        return result

    def create(self, files):
        if not files or len(files) > self.max_files or sum(f['size'] for f in files) > self.max_expanded_bytes:
            raise WebAppError(REASONS['UPLOAD_QUOTA'], status_code=413)
        if any(type(f.get('size')) is not int or f['size'] < 0 or not isinstance(f.get('name'), str) or not f['name'] or len(f['name']) > 1024 for f in files):
            raise WebAppError("Each upload file needs a valid name and byte size.", status_code=422)
        identity, now = uuid.uuid4().hex, utc_now()
        directory = self.source_root / identity
        directory.mkdir()
        (directory / 'files').mkdir()
        with self.store.connect() as db:
            db.execute("INSERT INTO upload_sessions VALUES(?,'uploading',?,NULL,?,?)", (identity, identity, now, now))
            db.executemany("INSERT INTO upload_files(upload_id,file_index,name,expected_size) VALUES(?,?,?,?)", [(identity, i, f['name'], f['size']) for i, f in enumerate(files)])
        return self.status(identity)

    def put(self, identity, index, offset, data):
        if len(data) > MAX_CHUNK or offset < 0:
            raise WebAppError("Upload chunk exceeds its limit or has an invalid offset.", status_code=413)
        directory = self._directory(identity)
        path = directory / 'files' / f'{index:06d}.source'
        try:
            with self.store.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                session = db.execute('SELECT * FROM upload_sessions WHERE id=?', (identity,)).fetchone()
                row = db.execute('SELECT * FROM upload_files WHERE upload_id=? AND file_index=?', (identity, index)).fetchone()
                if row is None:
                    raise WebAppError('Upload file not found.', status_code=404)
                received = row['received_bytes']
                if offset + len(data) > row['expected_size'] or offset > received:
                    raise WebAppError('Chunk offset does not match the saved upload. Refresh its status.', status_code=409)
                if offset < received:
                    if offset + len(data) > received or not path.is_file():
                        raise WebAppError('Retry the chunk at the exact saved offset.', status_code=409)
                    with path.open('rb') as stream:
                        stream.seek(offset)
                        if stream.read(len(data)) != data:
                            raise WebAppError('Reselected file differs from the saved upload. Start a new upload.', status_code=409)
                    return self._file_summary(row)
                if session['status'] != 'uploading':
                    if received == row['expected_size'] and not data:
                        return self._file_summary(row)
                    raise WebAppError('The finalized upload is immutable.', status_code=409)
                if shutil.disk_usage(directory).free < len(data) + self.catalog.minimum_free_bytes:
                    raise UploadFailure('DISK_SPACE')
                with path.open('r+b' if path.exists() else 'w+b') as stream:
                    # Discard only an uncommitted tail from an interrupted write.
                    stream.truncate(received)
                    stream.seek(received)
                    stream.write(data)
                    stream.flush()
                    os.fsync(stream.fileno())
                received += len(data)
                complete = received == row['expected_size']
                digest = sha256_file(path) if complete else None
                db.execute("UPDATE upload_files SET received_bytes=?,sha256=?,status=? WHERE upload_id=? AND file_index=?", (received, digest, 'received' if complete else 'uploading', identity, index))
                db.execute('UPDATE upload_sessions SET updated_at=? WHERE id=?', (utc_now(), identity))
                return {'index': index, 'name': row['name'], 'size': row['expected_size'], 'received_bytes': received, 'status': 'received' if complete else 'uploading', 'reason_code': None}
        except (OSError, UploadFailure) as error:
            raise WebAppError(REASONS['DISK_SPACE'] if isinstance(error, UploadFailure) or getattr(error, 'errno', None) == errno.ENOSPC else 'Upload storage is temporarily unavailable; the last committed offset is retained.', status_code=507) from error

    @staticmethod
    def _file_summary(row):
        return {'index': row['file_index'], 'name': row['name'], 'size': row['expected_size'], 'received_bytes': row['received_bytes'], 'status': row['status'], 'reason_code': row['reason_code']}

    def finalize(self, identity):
        self._session(identity)
        with self.store.connect() as db:
            db.execute('BEGIN IMMEDIATE')
            row = db.execute('SELECT * FROM upload_sessions WHERE id=?', (identity,)).fetchone()
            if row['status'] == 'uploading' or row['status'] == 'failed' and row['error_code'] in {'DISK_SPACE', 'INDEXING_ERROR'}:
                incomplete = db.execute('SELECT 1 FROM upload_files WHERE upload_id=? AND (received_bytes!=expected_size OR sha256 IS NULL)', (identity,)).fetchone()
                if incomplete:
                    raise WebAppError('All files must finish uploading before finalization.', status_code=409)
                db.execute("UPDATE upload_sessions SET status='queued',error_code=NULL,updated_at=? WHERE id=?", (utc_now(), identity))
        self._wake.set()
        return self.status(identity)

    def status(self, identity):
        session = self._session(identity)
        with self.store.connect() as db:
            files = db.execute('SELECT * FROM upload_files WHERE upload_id=? ORDER BY file_index', (identity,)).fetchall()
            items = db.execute("SELECT item_index,status,reason_code FROM upload_items WHERE upload_id=? AND status NOT IN ('dicom','duplicate') ORDER BY item_index", (identity,)).fetchall()
            rows = db.execute('SELECT u.*,t.study_label,t.recommended_series_id FROM upload_studies u JOIN catalog_studies t ON t.id=u.study_id WHERE upload_id=? ORDER BY t.study_label', (identity,)).fetchall()
            studies = []
            for row in rows:
                series = db.execute('SELECT s.* FROM catalog_series s JOIN upload_series us ON us.series_id=s.id WHERE us.upload_id=? AND s.study_id=?', (identity,row['study_id'])).fetchall()
                selected = next((s for s in series if s['id'] == (row['selected_series_id'] or row['recommended_series_id'])), None)
                run = db.execute('SELECT status FROM count_runs WHERE id=?', (row['count_run_id'],)).fetchone() if row['count_run_id'] else None
                studies.append({'study_id': row['study_id'], 'study_label': row['study_label'], 'series_count':len(series), 'recommended_series_id':row['selected_series_id'] or row['recommended_series_id'], 'case_id': selected['case_id'] if selected else None, 'preparation_status': selected['preparation_status'] if selected else 'unavailable' if row['status']=='unavailable' else 'preparing', 'count_run_id':row['count_run_id'], 'analysis_status': run['status'] if run else row['status'], 'reason_code': row['reason_code'], 'reason':REASONS.get(row['reason_code'])})
        return {'upload_id':identity, 'status':session['status'], 'reason_code':session['error_code'], 'reason':REASONS.get(session['error_code'] or self.last_error), 'worker_error':self.last_error, 'files':[self._file_summary(f) for f in files], 'studies':studies, 'issues':[{'item_index':r['item_index'],'status':r['status'],'reason_code':r['reason_code'],'reason':REASONS.get(r['reason_code'])} for r in items], 'limits':{'max_expanded_bytes':self.max_expanded_bytes,'max_files':self.max_files,'max_chunk_bytes':MAX_CHUNK}}

    def start(self):
        if self._thread:
            return
        self._lock_file = (self.app.root / 'upload-worker.lock').open('a+b')
        if self._lock_file.tell() == 0:
            self._lock_file.write(b'0'); self._lock_file.flush()
        self._lock_file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self._lock_file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.close(); self._lock_file = None
            return
        with self.store.connect() as db:
            db.execute("UPDATE upload_sessions SET status='queued' WHERE status='indexing'")
        self._thread = threading.Thread(target=self._loop, name='ethmoid-upload', daemon=True)
        self._thread.start()

    def close(self):
        self._stop.set(); self._wake.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _loop(self):
        try:
            while not self._stop.is_set():
                try:
                    with self.store.connect() as db:
                        queued = db.execute("SELECT id FROM upload_sessions WHERE status='queued' ORDER BY created_at,id LIMIT 1").fetchone()
                    if queued:
                        self.process(queued[0])
                    self.schedule_ready()
                    self.last_error = None
                except Exception:
                    # A storage outage cannot be written to the unavailable DB.
                    # Keep every persisted intent and retry without abandoning
                    # unrelated uploads or losing the worker thread.
                    self.last_error = "WORKER_STORAGE"
                self._wake.wait(timeout=1); self._wake.clear()
        finally:
            if self._lock_file:
                self._lock_file.close(); self._lock_file = None

    def _members(self, identity):
        directory = self._directory(identity)
        with self.store.connect() as db:
            files = db.execute('SELECT * FROM upload_files WHERE upload_id=? ORDER BY file_index', (identity,)).fetchall()
        entries, expanded = [], 0
        for file in files:
            path = directory / 'files' / f"{file['file_index']:06d}.source"
            if not path.is_file() or path.stat().st_size != file['expected_size'] or sha256_file(path) != file['sha256']:
                raise UploadFailure('INDEXING_ERROR')
            archive = file['name'].lower().endswith('.zip') or zipfile.is_zipfile(path)
            if not archive:
                expanded += path.stat().st_size
                entries.append((path, file['name'],file['file_index']))
            else:
                try:
                    with zipfile.ZipFile(path) as source:
                        members = source.infolist()
                        for member in members:
                            normalized = member.filename.replace('\\','/')
                            name = PurePosixPath(normalized)
                            mode = member.external_attr >> 16
                            if name.is_absolute() or '..' in name.parts or ':' in normalized or '\0' in normalized or stat.S_ISLNK(mode):
                                raise UploadFailure('ARCHIVE_UNSAFE')
                            if member.is_dir():
                                continue
                            if member.flag_bits & 1 or member.compress_type not in {zipfile.ZIP_STORED,zipfile.ZIP_DEFLATED}:
                                raise UploadFailure('ARCHIVE_INVALID')
                            expanded += member.file_size
                            if expanded > self.max_expanded_bytes or len(entries) >= self.max_files:
                                raise UploadFailure('UPLOAD_QUOTA')
                            if shutil.disk_usage(directory).free < member.file_size + self.catalog.minimum_free_bytes:
                                raise UploadFailure('DISK_SPACE')
                            destination = directory / 'expanded' / f'{len(entries):06d}.source'
                            destination.parent.mkdir(exist_ok=True)
                            temporary = destination.with_suffix('.partial')
                            copied = 0
                            with source.open(member) as inp, temporary.open('wb') as out:
                                for block in iter(lambda: inp.read(1024*1024), b''):
                                    copied += len(block)
                                    if copied > member.file_size:
                                        raise UploadFailure('ARCHIVE_INVALID')
                                    out.write(block)
                                out.flush(); os.fsync(out.fileno())
                            if copied != member.file_size:
                                raise UploadFailure('ARCHIVE_INVALID')
                            os.replace(temporary,destination)
                            entries.append((destination,member.filename,file['file_index']))
                except (zipfile.BadZipFile,RuntimeError,NotImplementedError,EOFError) as error:
                    raise UploadFailure('ARCHIVE_INVALID') from error
            if expanded > self.max_expanded_bytes or len(entries) > self.max_files:
                raise UploadFailure('UPLOAD_QUOTA')
        return entries

    def process(self, identity):
        with self.store.connect() as db:
            claimed = db.execute("UPDATE upload_sessions SET status='indexing',updated_at=? WHERE id=? AND status='queued'", (utc_now(),identity))
            if claimed.rowcount != 1:
                return
        try:
            entries = self._members(identity)
            groups, items, seen = {}, [], {}
            series_studies, sop_locations, conflicting_series = {}, {}, set()
            for index,(path,name,file_index) in enumerate(entries):
                digest = sha256_file(path)
                item = {'index':index,'path':path,'name':name,'hash':digest,'study':None,'series':None,'sop':None,'status':'dicom','reason':None,'file_index':file_index}
                try:
                    ds = pydicom.dcmread(path,stop_before_pixels=True)
                    item.update(study=str(ds.StudyInstanceUID),series=str(ds.SeriesInstanceUID),sop=str(ds.SOPInstanceUID))
                    if not all(pydicom.uid.UID(item[k]).is_valid for k in ('study','series','sop')):
                        raise ValueError('invalid identity')
                    key = (item['study'],item['series'],item['sop'])
                    series_key = (item['study'], item['series'])
                    old_study = series_studies.setdefault(item['series'], item['study'])
                    if old_study != item['study']:
                        conflicting_series.update({series_key, (old_study, item['series'])})
                    old_location = sop_locations.setdefault(item['sop'], series_key)
                    if old_location != series_key:
                        conflicting_series.update({series_key, old_location})
                    if key in seen:
                        item.update(status='duplicate' if seen[key]==digest else 'conflict',reason=None if seen[key]==digest else 'SOURCE_CONFLICT')
                    else:
                        seen[key]=digest
                    groups.setdefault((item['study'],item['series']),[]).append((item,ds))
                except InvalidDicomError:
                    item.update(status='unreadable' if name.lower().endswith('.dcm') else 'ancillary',reason='UNREADABLE' if name.lower().endswith('.dcm') else 'ANCILLARY')
                except Exception:
                    item.update(status='unreadable',reason='UNREADABLE')
                items.append(item)
            for key in conflicting_series:
                for item,_ in groups.get(key, []):
                    item.update(status='conflict', reason='SOURCE_CONFLICT')
            self._catalog(identity,groups,items)
        except Exception as error:
            code = str(error) if isinstance(error,UploadFailure) else 'DISK_SPACE' if isinstance(error,OSError) and error.errno==errno.ENOSPC else 'INDEXING_ERROR'
            with self.store.connect() as db:
                db.execute("UPDATE upload_sessions SET status='failed',error_code=?,updated_at=? WHERE id=?", (code,utc_now(),identity))

    @staticmethod
    def _descriptor(ds,path,count):
        try:
            headers = list(frame_headers(path))
            first = headers[0][1]

        except Exception:
            first=ds
        def number(value):
            try:
                result=float(value)
                return result if result > 0 and result < float('inf') else None
            except (TypeError,ValueError):
                return None
        pixels = getattr(first,'PixelSpacing',None)
        return safe_descriptor({'pixel_spacing_mm':[number(v) for v in pixels] if pixels is not None else None,'ipp_spacing_median_mm':number(getattr(first,'SpacingBetweenSlices',None)),'slice_thickness_mm':number(getattr(first,'SliceThickness',None)),'image_count':count,'kernel':str(getattr(first,'ConvolutionKernel','')),'image_type':'\\'.join(str(v) for v in getattr(first,'ImageType',[]))})

    def _catalog(self,identity,groups,items):
        directory=self._directory(identity)
        with self.catalog._sync_lock:
            with self.store.connect() as db:
                existing=db.execute('SELECT s.*,t.study_uid FROM catalog_series s JOIN catalog_studies t ON t.id=s.study_id').fetchall()
            known={}
            for row in existing:
                if (row['study_uid'],row['series_uid']) in groups:
                    try:
                        known[(row['study_uid'],row['series_uid'])]=(row,self.catalog.fingerprint(row))
                    except Exception:
                        known[(row['study_uid'],row['series_uid'])]=(row,None)
            with self.store.connect() as db:
                db.execute('BEGIN IMMEDIATE')
                source=db.execute('SELECT id FROM catalog_sources WHERE root_path=?',(str(directory),)).fetchone()
                source_id=source[0] if source else uuid.uuid4().hex
                db.execute("INSERT INTO catalog_sources(id,root_path,status,updated_at) VALUES(?,?,'ready',?) ON CONFLICT(root_path) DO NOTHING",(source_id,str(directory),utc_now()))
                for (study_uid,series_uid), members in sorted(groups.items()):
                    unique=[item for item,_ in members if item['status']=='dicom']
                    conflicted=any(item['status']=='conflict' for item,_ in members)
                    fingerprint=fingerprint_hashes(item['hash'] for item in unique)
                    previous=known.get((study_uid,series_uid))
                    if any(row['series_uid']==series_uid and row['study_uid']!=study_uid for row in existing):
                        conflicted=True
                    if previous and previous[1]!=fingerprint:
                        conflicted=True
                    if conflicted:
                        for item,_ in members:
                            item.update(status='conflict',reason='SOURCE_CONFLICT')
                        continue
                    if previous:
                        row,_=previous
                        study_id,series_id=row['study_id'],row['id']
                        disposition='duplicate'
                        for item,_ in members:
                            item.update(status='duplicate',reason=None)
                    else:
                        study=db.execute('SELECT * FROM catalog_studies WHERE study_uid=? ORDER BY study_label LIMIT 1',(study_uid,)).fetchone()
                        if study:
                            study_id=study['id']
                        else:
                            study_id=uuid.uuid4().hex
                            labels={r[0] for r in db.execute('SELECT study_label FROM catalog_studies')}
                            label=self.catalog._label('',labels,'Case')
                            db.execute('INSERT INTO catalog_studies(id,source_id,study_uid,study_label) VALUES(?,?,?,?)',(study_id,source_id,study_uid,label))
                        series_id=uuid.uuid4().hex
                        labels={r[0] for r in db.execute('SELECT series_label FROM catalog_series WHERE study_id=?',(study_id,))}
                        label=self.catalog._label('',labels,'Series')
                        frame_count = 0
                        for member, dataset in members:
                            if member['status'] == 'dicom':
                                try:
                                    frame_count += max(1, int(getattr(dataset, 'NumberOfFrames', 1)))
                                except (TypeError, ValueError):
                                    frame_count += 1
                        metadata=self._descriptor(members[0][1],unique[0]['path'],frame_count)
                        reason='SCOUT' if metadata['source_kind']=='scout' else 'PROJECTION' if metadata['source_kind']=='projection' else None
                        relative=[str(item['path'].relative_to(directory)) for item in unique]
                        db.execute('INSERT INTO catalog_series(id,study_id,series_uid,series_label,descriptor_json,files_json,inventory_signature,source_fingerprint,preparation_status,reason_code) VALUES(?,?,?,?,?,?,?,?,?,?)',(series_id,study_id,series_uid,label,encode_json(metadata),encode_json(relative),fingerprint,fingerprint,'unavailable' if reason else 'not_prepared',reason))
                        db.execute('INSERT INTO catalog_series_sources(series_id,source_id) VALUES(?,?)',(series_id,source_id))
                        disposition='new'
                    db.execute("INSERT INTO upload_studies(upload_id,study_id,has_new_series) VALUES(?,?,?) ON CONFLICT(upload_id,study_id) DO UPDATE SET has_new_series=MAX(has_new_series,excluded.has_new_series)",(identity,study_id,int(disposition=='new')))
                    db.execute('INSERT OR IGNORE INTO upload_series VALUES(?,?,?)',(identity,series_id,disposition))
                for item in items:
                    db.execute('INSERT OR REPLACE INTO upload_items VALUES(?,?,?,?,?,?,?,?,?,?)',(identity,item['index'],str(item['path'].relative_to(directory)),item['name'],item['hash'],item['study'],item['series'],item['sop'],item['status'],item['reason']))
                for file_index in {i['file_index'] for i in items}:
                    related=[i for i in items if i['file_index']==file_index]
                    state=related[0]['status'] if len(related)==1 else 'archive'
                    reason=next((i['reason'] for i in related if i['reason']),None)
                    db.execute('UPDATE upload_files SET status=?,reason_code=? WHERE upload_id=? AND file_index=?',(state,reason,identity,file_index))
                count=db.execute('SELECT COUNT(*) FROM upload_studies WHERE upload_id=?',(identity,)).fetchone()[0]
                db.execute('UPDATE upload_sessions SET status=?,error_code=?,updated_at=? WHERE id=?',('preparing' if count else 'completed',None if count else 'SOURCE_CONFLICT' if groups else 'NO_DICOM',utc_now(),identity))
            self.catalog.refresh_recommendations()
        self.schedule_ready()

    def schedule_ready(self):
        queue=self.catalog._queue
        if queue is None:
            return
        with self.store.connect() as db:
            pending=db.execute("SELECT u.* FROM upload_studies u JOIN upload_sessions s ON s.id=u.upload_id WHERE s.status='preparing' AND u.status='preparing'").fetchall()
        for study in pending:
            with self.store.connect() as db:
                rows=db.execute('SELECT s.*,us.disposition FROM catalog_series s JOIN upload_series us ON us.series_id=s.id WHERE us.upload_id=? AND s.study_id=?',(study['upload_id'],study['study_id'])).fetchall()
            if not study['has_new_series']:
                with self.store.connect() as db:
                    db.execute("UPDATE upload_studies SET status='duplicate' WHERE upload_id=? AND study_id=?",(study['upload_id'],study['study_id']))
                continue
            active=False
            for row in rows:
                if row['preparation_status']=='not_prepared':
                    queue.enqueue(row['id'],priority=0)
                    active=True
                elif row['preparation_status'] in {'queued','preparing'}:
                    active=True
            if active:
                continue
            ready=[r for r in rows if r['preparation_status']=='ready' and r['case_id']]
            if not ready:
                with self.store.connect() as db:
                    db.execute("UPDATE upload_studies SET status='unavailable',reason_code='UNSUPPORTED_INPUT' WHERE upload_id=? AND study_id=?",(study['upload_id'],study['study_id']))
                continue
            row=min(ready,key=self.catalog.rank)
            key='upload-'+hashlib.sha256((row['id']+':'+row['source_fingerprint']).encode()).hexdigest()
            with self.store.connect() as db:
                db.execute('INSERT OR IGNORE INTO upload_analysis_intents(series_id,source_fingerprint,idempotency_key) VALUES(?,?,?)',(row['id'],row['source_fingerprint'],key))
                intent=db.execute('SELECT * FROM upload_analysis_intents WHERE series_id=? AND source_fingerprint=?',(row['id'],row['source_fingerprint'])).fetchone()
            run_id=intent['count_run_id']
            if run_id is None:
                try:
                    run=self.create_count_run(row['case_id'],idempotency_key=key)
                    run_id=run.get('run_id') or run.get('id')
                    if run_id is None:
                        raise ValueError('invalid count response')
                except WebAppError as error:
                    if error.status_code==409:
                        with self.store.connect() as db:
                            active_run=db.execute("SELECT id FROM count_runs WHERE case_id=? AND status IN ('queued','running')",(row['case_id'],)).fetchone()
                        if active_run:
                            run_id=active_run[0]
                    if run_id is None:
                        with self.store.connect() as db:
                            db.execute("UPDATE upload_studies SET status='needs_run',reason_code='COUNT_SCHEDULING' WHERE upload_id=? AND study_id=?",(study['upload_id'],study['study_id']))
                        continue
                except Exception:
                    with self.store.connect() as db:
                        db.execute("UPDATE upload_studies SET status='needs_run',reason_code='COUNT_SCHEDULING' WHERE upload_id=? AND study_id=?", (study['upload_id'],study['study_id']))
                    continue
            with self.store.connect() as db:
                db.execute("UPDATE upload_analysis_intents SET status='scheduled',count_run_id=? WHERE series_id=? AND source_fingerprint=?",(run_id,row['id'],row['source_fingerprint']))
                db.execute("UPDATE upload_studies SET status='scheduled',selected_series_id=?,count_run_id=?,reason_code=NULL WHERE upload_id=? AND study_id=?",(row['id'],run_id,study['upload_id'],study['study_id']))
        with self.store.connect() as db:
            db.execute("UPDATE upload_sessions SET status='completed',updated_at=? WHERE status='preparing' AND NOT EXISTS (SELECT 1 FROM upload_studies WHERE upload_id=upload_sessions.id AND status='preparing')",(utc_now(),))
