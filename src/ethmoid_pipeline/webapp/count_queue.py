"""Durable, exclusively owned local inference queue.

The OS lock owns inference, including crash recovery. Heartbeats are diagnostic;
they never permit a second worker to steal a live process's GPU job.
"""
from __future__ import annotations

import os
import threading
import time
import uuid

from ..util import utc_now


class CountQueue:
    def __init__(self, service):
        self.service = service
        self.store = service.store
        self.owner = uuid.uuid4().hex
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.lock = None
        self.last_error = None
        self.thread = threading.Thread(target=self._loop, name="ethmoid-count", daemon=True)
        self.thread.start()

    def _acquire(self):
        path = self.service.root / "count-worker.lock"
        stream = path.open("a+b")
        if path.stat().st_size == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            stream.close()
            return False
        self.lock = stream
        self._recover()
        return True

    def _recover(self):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # Upgrade unfinished authorized runs only. No catalog read schedules work.
            db.execute("""INSERT OR IGNORE INTO count_jobs(run_id,created_at,updated_at)
                SELECT id,created_at,updated_at FROM count_runs WHERE status IN ('queued','running')""")
            # A server may die after accepting cancellation but before its worker
            # observes the flag. Such jobs must never remain queued and unclaimable.
            now = utc_now()
            db.execute("""UPDATE count_runs SET status='failed',failure_code='CANCELLED',updated_at=?
                WHERE status IN ('queued','running') AND id IN
                (SELECT run_id FROM count_jobs WHERE cancel_requested=1)""", (now,))
            db.execute("""UPDATE count_jobs SET state='done',stage='finished',outcome='cancelled',
                owner=NULL,last_error='CANCELLED',updated_at=? WHERE cancel_requested=1
                AND run_id IN (SELECT id FROM count_runs WHERE status='failed' AND failure_code='CANCELLED')""", (now,))
            db.execute("UPDATE count_jobs SET state='queued',owner=NULL WHERE state='running'")
            db.execute("""UPDATE count_runs SET status='queued' WHERE status='running'
                AND id IN (SELECT run_id FROM count_jobs WHERE state='queued')""")

    def _claim(self):
        with self.store.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT j.run_id,r.case_id FROM count_jobs j JOIN count_runs r ON r.id=j.run_id
                WHERE j.state='queued' AND r.status='queued' AND j.cancel_requested=0
                ORDER BY j.created_at,j.run_id LIMIT 1""").fetchone()
            if row is None:
                return None
            now = utc_now()
            db.execute("""UPDATE count_jobs SET state='running',owner=?,heartbeat_at=?,
                attempt=attempt+1,updated_at=? WHERE run_id=?""", (self.owner, now, now, row['run_id']))
            return row

    def _heartbeat(self, run_id, finished):
        while not finished.wait(2):
            try:
                with self.store.connect() as db:
                    db.execute("UPDATE count_jobs SET heartbeat_at=? WHERE run_id=? AND owner=?",
                               (utc_now(), run_id, self.owner))
            except Exception:
                # The active owner remains protected by the OS lock. Work itself
                # handles database errors without publishing an unverified result.
                pass

    def _loop(self):
        try:
            while not self.stop.is_set():
                try:
                    if self.lock is None and not self._acquire():
                        self.wake.wait(1)
                        self.wake.clear()
                        continue
                    if self.last_error:
                        self._recover()
                    row = self._claim()
                    self.last_error = None
                except Exception:
                    self.last_error = 'COUNT_QUEUE_STORAGE_UNAVAILABLE'
                    self.stop.wait(1)
                    continue
                if row is None:
                    self.wake.wait(0.25)
                    self.wake.clear()
                    continue
                finished = threading.Event()
                beat = threading.Thread(target=self._heartbeat, args=(row['run_id'], finished), daemon=True)
                beat.start()
                try:
                    self.service._work(row['case_id'], row['run_id'])
                except Exception:
                    # Storage can fail even while recording an error. Keep the
                    # owner alive and recover its unfinished claim once storage
                    # becomes available; never present an uncommitted count.
                    self.last_error = 'COUNT_QUEUE_STORAGE_UNAVAILABLE'
                    self.stop.wait(1)
                finally:
                    finished.set()
                    beat.join(timeout=3)
        finally:
            if self.lock is not None:
                self.lock.close()
                self.lock = None

    def notify(self):
        self.wake.set()

    def wait_idle(self, timeout=30):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.store.connect() as db:
                n = db.execute("SELECT COUNT(*) FROM count_jobs WHERE state IN ('queued','running')").fetchone()[0]
            if not n:
                return
            time.sleep(0.02)
        raise TimeoutError("Counting queue did not become idle")

    def close(self):
        self.stop.set()
        self.wake.set()
        self.thread.join(timeout=10)
