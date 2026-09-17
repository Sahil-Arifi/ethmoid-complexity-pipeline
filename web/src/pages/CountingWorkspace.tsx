import { lazy, Suspense, useCallback, useEffect, useRef, useState } from "react";
import { api, ApiError } from "../api/client";
import { countingApi, type Cell, type Correction, type CountRun } from "../api/counting";
import type { CaseDetail } from "../api/types";
import type { ProtocolDefinition } from "../config/protocol";
import { BrandMark } from "../components/BrandMark";
import { StudySeriesSwitcher } from "../components/StudySeriesSwitcher";
import { qualityNotes } from "../components/catalog";
import { countFailureMessage } from "../components/countFailure";
import { NiiVueViewer, type NiiVueViewerHandle, type ViewerLocation, type ViewerTool, type ViewMode } from "../viewer/NiiVueViewer";
import "./counting.css";

type Props = { caseId: string; onBack: () => void; onOpenCase?: (caseId: string) => void; registerNavigationGuard: (guard: (() => Promise<boolean>) | null) => void };
const PreviousWorkspace = lazy(() => import("./Workspace").then(m => ({ default: m.Workspace })));
const message = (error: unknown) => error instanceof Error ? error.message : "The request failed.";

export function CountingWorkspace(props: Props) {
  const [previous, setPrevious] = useState(() => new URLSearchParams(window.location.search).get("view") === "annotations");
  const switchView = (value: boolean) => {
    const url = new URL(window.location.href);
    if (value) url.searchParams.set("view", "annotations"); else url.searchParams.delete("view");
    window.history.replaceState({}, "", url.pathname + url.search);
    setPrevious(value);
  };
  return previous ? <Suspense fallback={<main className="center-state">Opening previous annotations…</main>}>
    <PreviousWorkspace key={props.caseId} {...props} onBack={() => switchView(false)} />
  </Suspense> : <CountReview key={props.caseId} {...props} onPrevious={() => switchView(true)} />;
}

function CountReview({ caseId, onBack, onOpenCase, registerNavigationGuard, onPrevious }: Props & { onPrevious: () => void }) {
  const viewer = useRef<NiiVueViewerHandle>(null);
  const [detail, setDetail] = useState<CaseDetail | null>(null);
  const [protocol, setProtocol] = useState<ProtocolDefinition | null>(null);
  const [run, setRun] = useState<CountRun | null>(null);
  const [runs, setRuns] = useState<CountRun[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [switchingSeries, setSwitchingSeries] = useState(false);
  const [ready, setReady] = useState(false);
  const [selected, setSelected] = useState<number[]>([]);
  const [tool, setTool] = useState<ViewerTool>("navigate");
  const [viewMode, setViewMode] = useState<ViewMode>("multiplanar");
  const [center, setCenter] = useState(300);
  const [width, setWidth] = useState(3000);
  const [opacity, setOpacity] = useState(0.55);
  const [visible, setVisible] = useState(true);
  const [overlayKind, setOverlayKind] = useState<"air" | "cavities">("air");
  const [brush, setBrush] = useState(2);
  const [history, setHistory] = useState([false, false]);
  const [location, setLocation] = useState<ViewerLocation>({ voxel: "—", millimeters: "—" });
  const [edit, setEdit] = useState<{ type: "add" | "split" | "update"; id?: number; parentRevision: number } | null>(null);
  const [dirty, setDirty] = useState(false);
  const [side, setSide] = useState<"left" | "right">("left");
  const [showOriginal, setShowOriginal] = useState(false);
  const [notice, setNotice] = useState("");
  const keyRef = useRef<string | null>(null);
  const [viewChange, setViewChange] = useState(0);
  const savedViewRef = useRef<string | null>(null);
  const active = run?.status === "queued" || run?.status === "running";
  const completed = run?.status === "completed";
  const readOnly = detail?.state === "APPROVED" || detail?.review_status === "Approved";
  const canEdit = completed && !busy && !readOnly && !showOriginal && ready;
  const revision = showOriginal ? 0 : (run?.latest_revision ?? 0);
  const cells = run?.cells.filter(c => !c.retired && c.included) ?? [];
  const selectedCell = cells.find(c => c.id === selected[0]);
  const staleEdit = !!edit && edit.parentRevision !== run?.latest_revision;

  const guard = useCallback(async () => {
    if (switchingSeries) return false;
    if (edit && !window.confirm("Discard the pending cell correction?")) return false;
    if (ready && viewer.current && !readOnly) {
      try { await api.saveViewerState(caseId, viewer.current.getViewerState()); }
      catch (e) { setError(message(e)); return false; }
    }
    return true;
  }, [caseId, edit, readOnly, ready, switchingSeries]);
  useEffect(() => {
    registerNavigationGuard(guard);
    const unload = (event: BeforeUnloadEvent) => { if (edit) { event.preventDefault(); event.returnValue = ""; } };
    window.addEventListener("beforeunload", unload);
    return () => { registerNavigationGuard(null); window.removeEventListener("beforeunload", unload); };
  }, [edit, guard, registerNavigationGuard]);

  useEffect(() => {
    let cancelled = false;
    void Promise.all([api.getCase(caseId), api.getProtocol(), countingApi.list(caseId)])
      .then(async ([data, rules, result]) => {
        if (cancelled) return;
        setDetail(data); setProtocol(rules); setRuns(result.runs);
        const saved = data.viewer_state;
        if (saved) {
          setCenter(saved.window_center ?? 300); setWidth(saved.window_width ?? 3000);
          setViewMode(saved.view_mode ?? "multiplanar"); setBrush(saved.brush_size ?? 2);
          setOpacity(saved.overlay_opacity ?? 0.55);
        }
        if (result.runs[0]) {
          const latest = await countingApi.get(caseId, result.runs[0].run_id);
          if (!cancelled) setRun(latest);
        }
      }).catch(e => { if (!cancelled) setError(message(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [caseId]);

  useEffect(() => {
    if (!ready || readOnly) return;
    const signature = JSON.stringify([center, width, opacity, brush, viewMode, viewChange]);
    if (savedViewRef.current === null) { savedViewRef.current = signature; return; }
    if (savedViewRef.current === signature) return;
    const timer = setTimeout(() => {
      if (!viewer.current) return;
      void api.saveViewerState(caseId, viewer.current.getViewerState())
        .then(() => { savedViewRef.current = signature; })
        .catch(e => setError(message(e)));
    }, 750);
    return () => clearTimeout(timer);
  }, [brush, caseId, center, opacity, readOnly, ready, viewChange, viewMode, width]);

  useEffect(() => {
    if (!active || !run) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        const value = await countingApi.get(caseId, run.run_id);
        if (cancelled) return;
        setRun(value); setError("");
        if (value.status === "queued" || value.status === "running") timer = setTimeout(poll, 2000);
      } catch (e) {
        if (!cancelled) { setError(message(e)); timer = setTimeout(poll, 5000); }
      }
    };
    timer = setTimeout(poll, 1000);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [active, caseId, run?.run_id]);

  const clearEdit = () => { viewer.current?.clearSelectionMask(); setEdit(null); setDirty(false); setTool("navigate"); };
  const execute = async (action: () => Promise<void>) => {
    setBusy(true); setError(""); setNotice("");
    try { await action(); } catch (e) { setError(message(e)); }
    finally { setBusy(false); }
  };
  const start = () => execute(async () => {
    if (!(await guard())) return;
    keyRef.current ??= crypto.randomUUID();
    const value = await countingApi.start(caseId, keyRef.current);
    keyRef.current = null;
    clearEdit(); setSelected([]); setShowOriginal(false); setRun(value);
    setRuns(previous => [value, ...previous.filter(r => r.run_id !== value.run_id)]);
  });
  const save = (operation: Correction, mask?: Blob) => execute(async () => {
    if (!run || run.latest_revision === null) return;
    try {
      const value = await countingApi.correct(caseId, run.run_id, edit?.parentRevision ?? run.latest_revision, operation, mask);
      setRun(value); clearEdit(); setSelected([]); setNotice(`Correction saved as revision ${value.latest_revision}. The original prediction is preserved.`);
    } catch (e) {
      if (e instanceof ApiError && e.status === 409) {
        setRun(await countingApi.get(caseId, run.run_id));
        throw new Error("Another window saved a revision. Discard the pending correction, then review the latest cells before reapplying it.");
      }
      throw e;
    }
  });
  const prepare = (type: "add" | "split" | "update") => execute(async () => {
    if (!(await guard())) return;
    clearEdit();
    if (type === "update" && selectedCell && run) {
      await viewer.current?.prepareCellMask(countingApi.artifact(caseId, run.run_id, "cavities", revision), selectedCell.id);
    }
    setEdit({ type, id: type === "add" ? undefined : selectedCell?.id, parentRevision: run?.latest_revision ?? 0 });
    setSide(selectedCell?.side ?? "left"); setTool("draw"); setViewMode("multiplanar");
  });
  const applyMask = async () => {
    if (!edit || staleEdit || !viewer.current) return;
    try {
      const mask = await viewer.current.exportMask();
      await save({ type: edit.type, id: edit.id, ...(edit.type === "add" ? { side } : {}) }, mask);
    } catch (e) { setError(message(e)); }
  };
  const selectRun = (id: string) => execute(async () => {
    if (!(await guard())) return;
    const value = await countingApi.get(caseId, id);
    clearEdit(); setRun(value); setSelected([]); setShowOriginal(false);
  });
  const focus = (cell: Cell) => {
    setSelected([cell.id]); viewer.current?.focusWorld(cell.centroid_ras_mm);
  };
  const download = () => execute(async () => {
    if (!run) return;
    const blob = await countingApi.download(caseId, run.run_id, revision);
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url; link.download = `count_revision_${String(revision).padStart(6, "0")}.zip`;
    document.body.appendChild(link); link.click(); link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 30_000);
  });

  if (loading) return <main className="center-state">Opening 3D counting workspace…</main>;
  if (!detail || !protocol) return <main className="center-state"><p role="alert">{error || "The case could not be loaded."}</p><button className="button" onClick={onBack}>Back to cases</button></main>;
  const overlay = completed && run ? countingApi.artifact(caseId, run.run_id, overlayKind, revision) : undefined;
  const counts = showOriginal ? run?.original_counts : run?.counts;
  const scanNotes = detail.catalog ? qualityNotes(detail.catalog) : [...detail.warnings, "Original acquisition thickness is unconfirmed for this imported volume; voxel spacing alone does not establish the source resolution."];
  return <main className="count-workspace" inert={switchingSeries || undefined}>
    <header className="workspace-header">
      <div className="count-heading"><button className="button tertiary" onClick={() => void guard().then(ok => { if (ok) onBack(); })}>Cases</button><BrandMark /><div><span className="eyebrow">3D ethmoid cell count</span><h1>{detail.catalog ? `${detail.catalog.study_label} · ${detail.catalog.series_label}` : detail.case_label}</h1><small>Experimental · clinical accuracy not established</small></div></div>
      <div className="toolbar-group">
        <button className="button" disabled={busy || active || !!readOnly || detail.validation_status === "FAILED"} onClick={() => void start()}>{active ? "Counting…" : run ? "Run again" : "Run automatic count"}</button>
        {completed && run ? <a className="button secondary" aria-disabled={busy} href={countingApi.exportUrl(caseId, run.run_id, revision)} onClick={event => { event.preventDefault(); if (!busy) void download(); }}>Export revision {revision}</a> : <button className="button secondary" disabled>Export</button>}
      </div>
    </header>
    {detail.catalog && onOpenCase && <StudySeriesSwitcher key={caseId} caseId={caseId} context={detail.catalog} disabled={busy} beforeSwitch={guard} onOpenCase={onOpenCase} onSwitchBusy={setSwitchingSeries} />}
    <div className="scan-quality"><strong>{detail.validation_status === "VERIFIED" ? "Conversion verified" : detail.validation_status === "FAILED" ? "Volume verification failed" : "Conversion unverified"}.</strong> {detail.validation_status === "FAILED" ? "A count cannot be attempted on this volume." : "Ready to attempt a count does not mean a count has completed."}{scanNotes.length > 0 && <details open><summary>Scan quality notes before counting</summary><ul>{scanNotes.map((note, i) => <li key={i}>{note}</li>)}</ul></details>}</div>
    {error && <div className="alert danger" role="alert">{error}</div>}
    {notice && <div className="alert" role="status">{notice}</div>}
    <section className="viewer-toolbar" aria-label="Scan display">
      <div className="toolbar-group"><button className="tool-button" aria-pressed={viewMode === "multiplanar"} onClick={() => setViewMode("multiplanar")}>Axial · Coronal · Sagittal</button><button className="tool-button" aria-pressed={viewMode === "render"} disabled={!!edit} onClick={() => { setViewMode("render"); setTool("navigate"); }}>3D</button><button className="tool-button" aria-pressed={tool === "navigate"} onClick={() => setTool("navigate")}>Navigate</button><button className="tool-button" aria-pressed={tool === "window"} onClick={() => setTool("window")}>Window/level</button><button className="tool-button" onClick={() => viewer.current?.resetView()}>Reset view</button></div>
      <div className="toolbar-group"><label>Center <input aria-label="Window center" className="count-number" type="number" value={center} onChange={e => setCenter(Number(e.target.value))} /></label><label>Width <input aria-label="Window width" className="count-number" type="number" min="1" value={width} onChange={e => setWidth(Math.max(1, Number(e.target.value)))} /></label><button className="tool-button" onClick={() => { setCenter(300); setWidth(3000); }}>Bone · 300/3000</button></div>
      <div className="toolbar-group"><label><input type="checkbox" checked={visible} onChange={e => setVisible(e.target.checked)} /> Overlay</label><label><span className="sr-only">Overlay content</span><select aria-label="Overlay content" value={overlayKind} onChange={e => setOverlayKind(e.target.value as "air" | "cavities")}><option value="air">Air-filled portions</option><option value="cavities">Cell compartments</option></select></label><input aria-label="Overlay opacity" type="range" min="0" max="1" step="0.05" value={opacity} onChange={e => setOpacity(Number(e.target.value))} /></div>
    </section>
    <div className="count-body">
      <section className="count-viewer" aria-label="CT and numbered cells">
        <NiiVueViewer ref={viewer} caseId={caseId} volumeFileType={detail.volume_file_type} protocol={protocol} hasAnnotation={false} initialState={detail.viewer_state}
          tool={tool} selectedLabel={1} brushSize={brush} overlayOpacity={opacity} overlayVisible={visible} viewMode={viewMode}
          windowCenter={center} windowWidth={width} instanceOverlayUrl={overlay}
          onReady={() => setReady(true)} onError={setError} onDirty={() => setDirty(true)} onViewStateChange={() => setViewChange(value => value + 1)}
          onWindowChange={(c, w) => { setCenter(c); setWidth(w); }} onLocationChange={setLocation} onHistoryChange={(u, r) => setHistory([u, r])} />
        <div className="count-coordinates"><span>Voxel {location.voxel}</span><span>RAS mm {location.millimeters}</span><strong>{location.cellId ? `Cell #${location.cellId}` : "No cell at crosshair"}</strong></div>
      </section>
      <aside className="count-panel" aria-label="Count and review">
        <section><span className="eyebrow">{showOriginal ? "Original automatic result" : (revision > 0 ? `Corrected result · revision ${revision}` : "Automatic result")}</span>
          {counts ? <dl className="count-totals"><div><dt>Left</dt><dd>{counts.left}</dd></div><div><dt>Right</dt><dd>{counts.right}</dd></div><div><dt>Total</dt><dd data-testid="total-count">{counts.total}</dd></div></dl> : <p role="status">{active ? `${run?.status === "queued" ? "Queued" : "Analyzing the full CT volume"}…` : run?.status === "failed" ? "Count unavailable" : "Ready for an automatic 3D count."}</p>}
          {run?.status === "failed" && <p role="alert">{countFailureMessage(run.failure_code, run.message)}</p>}
          {!run && <p className="muted">Run once to identify cells throughout the volume. No outline or seed points are required.</p>}
          {active && <p className="muted">The job continues locally if you leave this page. Return to see its result.</p>}
          {completed && <p className="muted">Each candidate ID is counted once across the 3D volume. Review possible missed or merged cells before presenting the result.</p>}
          {!!run?.latest_revision && <label><input type="checkbox" checked={showOriginal} disabled={busy || !!edit} onChange={e => { setShowOriginal(e.target.checked); setSelected([]); }} /> Show original prediction</label>}
          {runs.length > 1 && <label className="count-field">Run history<select aria-label="Run history" disabled={busy} value={run?.run_id} onChange={e => void selectRun(e.target.value)}>{runs.map((r, index) => <option key={r.run_id} value={r.run_id}>Run {runs.length - index} · {new Date(r.created_at).toLocaleString()}</option>)}</select></label>}
        </section>
        {!!run?.warnings.length && <details><summary>Analysis notes ({run.warnings.length})</summary><ul>{run.warnings.map((text, i) => <li key={i}>{text}</li>)}</ul></details>}
        {readOnly && <p className="alert">This case is approved and locked. Return it to review in previous annotations to make changes.</p>}
        {completed && !showOriginal && <>
          <section><h2>Review cells</h2><p className="muted">Click a cell to locate it. Check two or more cells to merge them.</p>
            <div className="count-cells">{cells.map(cell => <div key={cell.id} className={selected.includes(cell.id) ? "count-cell selected" : "count-cell"}>
              <input type="checkbox" aria-label={`Select cell ${cell.id}`} disabled={!!edit || busy} checked={selected.includes(cell.id)} onChange={e => setSelected(ids => e.target.checked ? [...ids, cell.id] : ids.filter(id => id !== cell.id))} />
              <button disabled={!ready || !!edit} onClick={() => focus(cell)}><strong>#{cell.id} · {cell.side}</strong><span>{cell.uncertain ? "Needs review" : "Candidate"}{cell.origin === "manual" ? " · Manual" : ""}{cell.air_voxel_count === 0 ? " · No visible air" : ""}</span></button>
            </div>)}</div>
            {selected.length === 1 && selectedCell && <div className="count-cell-detail"><strong>Cell #{selectedCell.id}</strong>{selectedCell.reasons.map((r, i) => <p key={i}>{r.replaceAll("_", " ")}</p>)}<small>{selectedCell.voxel_count} compartment voxels · {selectedCell.air_voxel_count} air voxels</small></div>}
          </section>
          {!edit ? <section><h2>Correct the result</h2><div className="count-actions">
            <button className="button" disabled={!canEdit} onClick={() => void prepare("add")}>Add cell</button>
            <button className="button" disabled={!canEdit || selected.length !== 1} onClick={() => void save({ type: "remove", id: selected[0] })}>Remove cell</button>
            <button className="button" disabled={!canEdit || selected.length < 2} onClick={() => void save({ type: "merge", ids: selected })}>Merge selected</button>
            <button className="button" disabled={!canEdit || selected.length !== 1} onClick={() => void prepare("split")}>Split cell</button>
            <button className="button" disabled={!canEdit || selected.length !== 1} onClick={() => void prepare("update")}>Edit boundary</button>
            <button className="button" disabled={!canEdit || selected.length !== 1 || !selectedCell?.uncertain} onClick={() => void save({ type: "update", id: selected[0], uncertain: false, reasons: ["Reviewed manually"] })}>Mark reviewed</button>
          </div></section> : <section className="count-edit"><h2>{edit.type === "add" ? "Add a cell" : edit.type === "split" ? `Split cell #${edit.id}` : `Edit cell #${edit.id} boundary`}</h2>
            {staleEdit && <p role="alert">This draft targets an earlier revision. Discard it before continuing.</p>}
            <p>{edit.type === "split" ? "Paint the part that should become a separate cell. The remaining part stays as a second cell." : edit.type === "add" ? "Paint the new compartment, or place the crosshair at its center and add a count-only cell." : "Adjust the selected cell's compartment. Apply to save a new revision."}</p>
            {edit.type === "add" && <label className="count-field">Side<select aria-label="New cell side" value={side} onChange={e => setSide(e.target.value as "left" | "right")}><option value="left">Left</option><option value="right">Right</option></select></label>}
            <div className="count-actions">{(["draw", "erase", "fill"] as ViewerTool[]).map(t => <button className="tool-button" key={t} aria-pressed={tool === t} onClick={() => setTool(t)}>{t[0].toUpperCase() + t.slice(1)}</button>)}<button className="tool-button" disabled={!history[0]} onClick={() => viewer.current?.undo()}>Undo</button><button className="tool-button" disabled={!history[1]} onClick={() => viewer.current?.redo()}>Redo</button></div>
            <label className="count-field">Brush size<input aria-label="Brush size" type="range" min="1" max="15" value={brush} onChange={e => setBrush(Number(e.target.value))} /></label>
            <div className="count-actions"><button className="button primary" disabled={busy || staleEdit || !dirty} onClick={() => void applyMask()}>Apply painted correction</button>
              {edit.type === "add" && <button className="button" disabled={busy || staleEdit || dirty} onClick={() => void save({ type: "add", side, centroid_ras_mm: viewer.current!.getWorldPosition(), uncertain: true, reasons: ["Manual count-only cell; no compartment mask"] })}>Add at crosshair</button>}
              <button className="button tertiary" disabled={busy} onClick={() => void guard().then(ok => { if (ok) clearEdit(); })}>Discard correction</button></div>
          </section>}
        </>}
        {showOriginal && <p className="muted">The overlay and export show revision 0. Turn off “Show original prediction” to review or correct the latest result.</p>}
        <section className="count-provenance"><small>Automatic validation: {detail.validation_status}. Clinical eligibility is unconfirmed. Preliminary results are not reference counts.</small><button className="text-button" onClick={() => void guard().then(ok => { if (ok) onPrevious(); })}>Previous annotations</button></section>
      </aside>
    </div>
  </main>;
}
