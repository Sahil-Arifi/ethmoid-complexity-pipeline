import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import { transferUpload, uploadApi, type UploadSession } from "../api/uploads";

type Props = { onClose: () => void; onImported: (caseId: string) => void; onCatalogUpdated?: () => void };
const STORAGE_KEY = "ethmoid.pending-upload";
const friendly = (value: string) => value.toLowerCase().replaceAll("_", " ");
const pendingId = () => { try { return localStorage.getItem(STORAGE_KEY); } catch { return null; } };
const remember = (id: string | null) => { try { if (id) localStorage.setItem(STORAGE_KEY, id); else localStorage.removeItem(STORAGE_KEY); } catch { /* Upload remains resumable in this tab. */ } };

export function ImportDialog({ onClose, onImported, onCatalogUpdated }: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const folderRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [kind, setKind] = useState<"dicom" | "zip" | "nifti">("dicom");
  const [caseLabel, setCaseLabel] = useState("");
  const [nifti, setNifti] = useState<File | null>(null);
  const [qcReport, setQcReport] = useState<File | null>(null);
  const [files, setFiles] = useState<File[]>([]);
  const [session, setSession] = useState<UploadSession | null>(null);
  const [resuming, setResuming] = useState(!!pendingId());
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState<[number, number]>([0, 0]);
  const [error, setError] = useState("");
  const [paused, setPaused] = useState(false);
  const [pollTick, setPollTick] = useState(0);
  const close = useCallback(() => { if (!busy) { onCatalogUpdated?.(); onClose(); } }, [busy, onCatalogUpdated, onClose]);

  useEffect(() => {
    if (kind === "dicom") { folderRef.current?.setAttribute("webkitdirectory", ""); folderRef.current?.setAttribute("directory", ""); }
  }, [kind, resuming, session?.status]);
  useEffect(() => {
    const previous = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const frame = requestAnimationFrame(() => dialogRef.current?.focus());
    return () => { cancelAnimationFrame(frame); abortRef.current?.abort(); previous?.focus(); };
  }, []);
  useEffect(() => {
    const id = pendingId(); if (!id) return;
    let stopped = false;
    void uploadApi.get(id).then(value => { if (!stopped) { setSession(value); if (value.status === "completed") onCatalogUpdated?.(); } })
      .catch(caught => { if (!stopped) setError(caught instanceof Error ? caught.message : "The previous upload could not be loaded."); })
      .finally(() => { if (!stopped) setResuming(false); });
    return () => { stopped = true; };
  }, []);
  const processing = session && ["queued", "indexing", "preparing"].includes(session.status);
  useEffect(() => {
    if (!processing || !session || busy) return;
    let stopped = false;
    const timer = setTimeout(() => {
      void uploadApi.get(session.upload_id).then(value => {
        if (!stopped) { setSession(value); setError(""); if (value.status === "completed") onCatalogUpdated?.(); }
      }).catch(caught => { if (!stopped) setError(caught instanceof Error ? caught.message : "Upload status is temporarily unavailable. Reconnecting…"); })
        .finally(() => { if (!stopped) setPollTick(value => value + 1); });
    }, 2000);
    return () => { stopped = true; clearTimeout(timer); };
  }, [session, processing, busy, pollTick, onCatalogUpdated]);
  useEffect(() => {
    const handle = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) { event.preventDefault(); close(); }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = Array.from(dialogRef.current.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), select:not([disabled]), a[href], [tabindex="0"]')).filter(element => !element.matches(":disabled"));
      const first = focusable[0]; const last = focusable.at(-1);
      if (event.shiftKey && (document.activeElement === first || document.activeElement === dialogRef.current)) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    };
    window.addEventListener("keydown", handle);
    const unload = (event: BeforeUnloadEvent) => { if (busy) { event.preventDefault(); event.returnValue = ""; } };
    window.addEventListener("beforeunload", unload);
    return () => { window.removeEventListener("keydown", handle); window.removeEventListener("beforeunload", unload); };
  }, [busy, close]);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault(); setError(""); setBusy(true); setPaused(false);
    const controller = new AbortController(); abortRef.current = controller;
    try {
      if (kind === "nifti" && !session) {
        if (!nifti) throw new Error("Choose a .nii or .nii.gz file.");
        const result = await api.importNifti(caseLabel.trim(), nifti, qcReport ?? undefined); onImported(result.case.case_id); return;
      }
      if (!files.length) throw new Error("Choose a DICOM folder or ZIP archive.");
      const current = session ?? await uploadApi.create(files);
      setSession(current); remember(current.upload_id);
      await transferUpload(current, files, (done, total) => setProgress([done, total]), controller.signal);
      setSession(await uploadApi.finalize(current.upload_id)); onCatalogUpdated?.();
    } catch (caught) {
      if (controller.signal.aborted) setPaused(true);
      else setError(caught instanceof Error ? caught.message : "Upload failed. The received files are retained for retry.");
    } finally { setBusy(false); abortRef.current = null; }
  };
  const retryFinalize = async () => {
    if (!session) return;
    setBusy(true); setError("");
    try { setSession(await uploadApi.finalize(session.upload_id)); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "Preparation could not be retried."); }
    finally { setBusy(false); }
  };
  const newUpload = () => { remember(null); setSession(null); setFiles([]); setProgress([0, 0]); setError(""); setPaused(false); };
  const transferForm = !session || session.status === "uploading";
  return <div className="modal-backdrop" role="presentation"><div className="modal" role="dialog" aria-modal="true" aria-labelledby="import-title" tabIndex={-1} ref={dialogRef}>
    <div className="modal-header"><div><span className="eyebrow">Local imaging workspace</span><h2 id="import-title">Import scan</h2></div><button className="icon-button" type="button" onClick={close} disabled={busy}><span aria-hidden="true">×</span><span className="sr-only">Close import dialog</span></button></div>
    {resuming ? <p role="status">Restoring the previous upload…</p> : transferForm ? <form onSubmit={event => void submit(event)}>
      <fieldset className="segmented-fieldset" disabled={busy}><legend>Input type</legend>{(["dicom", "zip", "nifti"] as const).filter(value => !session || value !== "nifti").map(value => <label key={value} className={kind === value ? "selected" : ""}><input type="radio" name="kind" checked={kind === value} onChange={() => { setKind(value); setFiles([]); }} />{value === "dicom" ? "DICOM folder" : value === "zip" ? "DICOM ZIP" : "NIfTI"}</label>)}</fieldset>
      {session && <p className="form-hint">Resume the saved upload by selecting the same files. Previously received bytes are checked before continuing.</p>}
      {kind === "nifti" && !session ? <><label className="field"><span>Neutral case label</span><input value={caseLabel} onChange={event => setCaseLabel(event.target.value)} pattern="Case[0-9]{3}" placeholder="Leave blank to generate" disabled={busy} /></label><label className="field"><span>NIfTI volume</span><input type="file" accept=".nii,.nii.gz" disabled={busy} onChange={event => setNifti(event.target.files?.[0] ?? null)} /></label><label className="field"><span>Matching QC report <small>(optional)</small></span><input type="file" accept=".json,application/json" disabled={busy} onChange={event => setQcReport(event.target.files?.[0] ?? null)} /></label><p className="form-hint">Without a matching valid QC report, the scan opens as UNVERIFIED. Start its count manually after review.</p></> : <><label className="field"><span>{kind === "zip" ? "DICOM ZIP archive" : "DICOM folder"}</span><input aria-label={kind === "zip" ? "DICOM ZIP archive" : "DICOM folder"} key={kind} ref={folderRef} type="file" multiple={kind === "dicom"} accept={kind === "zip" ? ".zip,application/zip" : undefined} disabled={busy} onChange={event => setFiles(Array.from(event.target.files ?? []))} /><small>{files.length ? `${files.length} files selected` : "Select a complete sinus CT study. Mixed studies are grouped automatically."}</small></label><p className="form-hint">All series stay in the catalog. The recommended verified series in each uploaded study is counted automatically. Images remain on this workstation.</p></>}
      {busy && <div role="status"><progress aria-label="Upload progress" max={progress[1] || 1} value={progress[0]} /><p>{progress[1] ? `${Math.round(progress[0] / progress[1] * 100)}% transferred and checked` : "Preparing upload…"}</p></div>}
      {paused && <p role="status">Upload paused. Received files are retained locally.</p>}
      {error && <div className="alert alert-error" role="alert">{error}</div>}
      <div className="modal-actions"><button className="button secondary" type="button" disabled={busy} onClick={close}>Close</button>{busy && kind !== "nifti" ? <button className="button" type="button" onClick={() => abortRef.current?.abort()}>Pause upload</button> : <button className="button primary" type="submit" disabled={busy}>{busy ? "Importing…" : session ? "Resume upload" : "Import scan"}</button>}</div>
    </form> : session && <section aria-label="Upload status"><p role="status"><strong>{session.status === "completed" ? "Upload cataloged" : session.status === "failed" ? "Upload needs attention" : session.status === "preparing" ? "Preparing verified volumes" : "Reading studies and series"}</strong></p><p>{session.files.length} uploaded files · {session.studies.length} studies. Counting continues locally after preparation; a completed upload does not mean the count succeeded.</p>
      {session.reason_code && <p role="alert">{friendly(session.reason_code)}</p>}
      <div className="series-list">{session.studies.map(study => <div className="series-option" key={study.study_id}><span><strong>{study.study_label} · {study.series_count} series</strong><small>Preparation: {friendly(study.preparation_status ?? "pending")}</small><small>Analysis: {friendly(study.analysis_status ?? "waiting for verification")}</small>{study.reason_code && <small>{friendly(study.reason_code)}</small>}{study.case_id && <button className="button compact" type="button" onClick={() => onImported(study.case_id!)}>Open {study.study_label}</button>}</span></div>)}</div>
      {session.issues.length > 0 && <details><summary>File outcomes ({session.issues.length})</summary><ul>{session.issues.map((issue, index) => <li key={index}>Item {(issue.item_index ?? index) + 1}: {issue.message ?? friendly(issue.reason_code ?? issue.status ?? "needs review")}</li>)}</ul></details>}
      {error && <div className="alert alert-error" role="alert">{error}</div>}
      <div className="modal-actions"><button className="button secondary" onClick={close} disabled={busy}>Return to cases</button>{session.status === "failed" && ["DISK_SPACE", "INDEXING_ERROR"].includes(session.reason_code ?? "") && <button className="button" onClick={() => void retryFinalize()} disabled={busy}>Retry preparation</button>}{["completed", "failed"].includes(session.status) && <button className="button" onClick={newUpload}>Import another study</button>}</div>
    </section>}
  </div></div>;
}
