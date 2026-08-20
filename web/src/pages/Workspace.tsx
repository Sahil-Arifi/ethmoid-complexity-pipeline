import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type {
  CaseDetail,
  QcChecklist,
  ReviewStatus,
  ViewerState,
} from "../api/types";
import {
  annotationLabels,
  type ProtocolDefinition,
} from "../config/protocol";
import {
  NiiVueViewer,
  type NiiVueViewerHandle,
  type ViewerLocation,
  type ViewerTool,
  type ViewMode,
} from "../viewer/NiiVueViewer";

type Props = {
  caseId: string;
  onBack: () => void;
  registerNavigationGuard: (guard: (() => Promise<boolean>) | null) => void;
};

const defaultQc: QcChecklist = {
  coverage: false,
  orientation: false,
  image_quality: false,
  slice_continuity: false,
};

const reviewOptions: ReviewStatus[] = ["Draft", "Ready for review", "Needs changes", "Approved"];

function allQcChecksPass(checklist: QcChecklist): boolean {
  return Object.values(checklist).every(Boolean);
}

function reviewSnapshot(checklist: QcChecklist, reviewerNotes: string, status: ReviewStatus) {
  return JSON.stringify([checklist, reviewerNotes, status]);
}

function savedControls(state?: ViewerState | null) {
  return {
    tool: state?.active_tool ?? ("draw" as ViewerTool),
    selectedLabel: state?.selected_label ?? 1,
    brushSize: state?.brush_size ?? 2,
    overlayOpacity: state?.overlay_opacity ?? 0.7,
    viewMode: state?.view_mode ?? ("multiplanar" as ViewMode),
    windowCenter: state?.window_center ?? 300,
    windowWidth: state?.window_width ?? 3000,
  };
}

export function Workspace({ caseId, onBack, registerNavigationGuard }: Props) {
  const viewerRef = useRef<NiiVueViewerHandle>(null);
  const viewerReadyRef = useRef(false);
  const changeVersionRef = useRef(0);
  const savedVersionRef = useRef(0);
  const savingRef = useRef(false);
  const viewerStateSaveEnabledRef = useRef(false);
  const savedReviewSnapshotRef = useRef(reviewSnapshot(defaultQc, "", "Draft"));
  const [detail, setDetail] = useState<CaseDetail | null>(null);
  const [protocol, setProtocol] = useState<ProtocolDefinition | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState("");
  const [viewerError, setViewerError] = useState("");
  const [viewerReady, setViewerReady] = useState(false);
  const [tool, setTool] = useState<ViewerTool>("draw");
  const [selectedLabel, setSelectedLabel] = useState(1);
  const [brushSize, setBrushSize] = useState(2);
  const [overlayOpacity, setOverlayOpacity] = useState(0.7);
  const [overlayVisible, setOverlayVisible] = useState(true);
  const [viewMode, setViewMode] = useState<ViewMode>("multiplanar");
  const [windowCenter, setWindowCenter] = useState(300);
  const [windowWidth, setWindowWidth] = useState(3000);
  const [location, setLocation] = useState<ViewerLocation>({ voxel: "—", millimeters: "—" });
  const [canUndo, setCanUndo] = useState(false);
  const [canRedo, setCanRedo] = useState(false);
  const [changeVersion, setChangeVersion] = useState(0);
  const [savedVersion, setSavedVersion] = useState(0);
  const [viewerStateVersion, setViewerStateVersion] = useState(0);
  const [saving, setSaving] = useState(false);
  const [saveMessage, setSaveMessage] = useState("No unsaved changes");
  const [actionError, setActionError] = useState("");
  const [qc, setQc] = useState<QcChecklist>(defaultQc);
  const [notes, setNotes] = useState("");
  const [reviewStatus, setReviewStatus] = useState<ReviewStatus>("Draft");
  const [reviewSaving, setReviewSaving] = useState(false);
  const [finalizing, setFinalizing] = useState(false);
  const isReadOnly = detail?.review_status === "Approved" || detail?.state === "APPROVED";
  const currentReviewSnapshot = reviewSnapshot(qc, notes, reviewStatus);
  const reviewDirty = currentReviewSnapshot !== savedReviewSnapshotRef.current;

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setLoadError("");
      try {
        const [result, loadedProtocol] = await Promise.all([
          api.getCase(caseId),
          api.getProtocol(),
        ]);
        if (cancelled) return;
        setDetail(result);
        setProtocol(loadedProtocol);
        setQc(result.qc_checklist ?? defaultQc);
        setNotes(result.reviewer_notes ?? "");
        setReviewStatus(result.review_status ?? "Draft");
        savedReviewSnapshotRef.current = reviewSnapshot(
          result.qc_checklist ?? defaultQc,
          result.reviewer_notes ?? "",
          result.review_status ?? "Draft",
        );
        const controls = savedControls(result.viewer_state);
        setTool(controls.tool);
        setSelectedLabel(controls.selectedLabel);
        setBrushSize(controls.brushSize);
        setOverlayOpacity(controls.overlayOpacity);
        setViewMode(controls.viewMode);
        setWindowCenter(controls.windowCenter);
        setWindowWidth(controls.windowWidth);
      } catch (caught) {
        if (!cancelled) {
          setLoadError(caught instanceof Error ? caught.message : "Case could not be loaded.");
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [caseId]);

  const markDirty = useCallback(() => {
    setChangeVersion((current) => {
      const next = current + 1;
      changeVersionRef.current = next;
      return next;
    });
    setSaveMessage("Unsaved changes");
  }, []);

  const saveAnnotation = useCallback(
    async (isAutosave = false): Promise<boolean> => {
      if (isReadOnly) return true;
      const viewer = viewerRef.current;
      if (!viewer || !viewerReadyRef.current || savingRef.current) return false;
      savingRef.current = true;
      setSaving(true);
      setActionError("");
      setSaveMessage(isAutosave ? "Autosaving…" : "Saving…");
      const versionAtStart = changeVersionRef.current;
      try {
        const [mask, viewerState] = await Promise.all([
          viewer.exportMask(),
          Promise.resolve(viewer.getViewerState()),
        ]);
        const revision = await api.saveAnnotation(caseId, mask, viewerState);
        setDetail((current) =>
          current
            ? {
                ...current,
                latest_revision: revision.revision,
                state: "ANNOTATING",
                review_status: "Draft",
                metrics: revision.metrics
                  ? revision.metrics.labels.map((item) => ({
                      label_value: item.value,
                      label_name: item.name,
                      voxel_count: item.voxel_count,
                      volume_mm3: item.volume_mm3,
                    }))
                  : current.metrics,
                updated_at: revision.created_at,
              }
            : current,
        );
        setSavedVersion(versionAtStart);
        savedVersionRef.current = versionAtStart;
        setReviewStatus("Draft");
        setSaveMessage(`Saved revision ${revision.revision}`);
        return true;
      } catch (caught) {
        const message = caught instanceof Error ? caught.message : "Annotation could not be saved.";
        setActionError(message);
        setSaveMessage("Save failed");
        return false;
      } finally {
        savingRef.current = false;
        setSaving(false);
      }
    },
    [caseId, isReadOnly],
  );

  useEffect(() => {
    if (!viewerReady || saving || changeVersion <= savedVersion) return;
    const timer = window.setTimeout(() => void saveAnnotation(true), 2000);
    return () => window.clearTimeout(timer);
  }, [changeVersion, saveAnnotation, savedVersion, saving, viewerReady]);

  const controlSnapshot = `${tool}|${selectedLabel}|${brushSize}|${overlayOpacity}|${viewMode}|${windowCenter}|${windowWidth}`;
  const previousControlSnapshot = useRef(controlSnapshot);
  useEffect(() => {
    if (previousControlSnapshot.current === controlSnapshot) return;
    previousControlSnapshot.current = controlSnapshot;
    if (viewerStateSaveEnabledRef.current) {
      setViewerStateVersion((current) => current + 1);
    }
  }, [controlSnapshot]);

  const saveViewerStateNow = useCallback(
    async (keepalive = false): Promise<boolean> => {
      const state = viewerRef.current?.getViewerState();
      if (!state) return true;
      try {
        await api.saveViewerState(caseId, state, keepalive);
        return true;
      } catch (caught) {
        setActionError(
          caught instanceof Error ? caught.message : "Viewer position could not be saved.",
        );
        return false;
      }
    },
    [caseId],
  );

  useEffect(() => {
    if (!viewerReady || viewerStateVersion === 0) return;
    const timer = window.setTimeout(() => {
      void saveViewerStateNow();
    }, 750);
    return () => window.clearTimeout(timer);
  }, [saveViewerStateNow, viewerReady, viewerStateVersion]);

  useEffect(() => {
    return () => {
      const state = viewerRef.current?.getViewerState();
      if (state) void api.saveViewerState(caseId, state, true);
    };
  }, [caseId]);

  useEffect(() => {
    if (changeVersion <= savedVersion && !reviewDirty) return;
    const warnBeforeUnload = (event: BeforeUnloadEvent) => {
      event.preventDefault();
      event.returnValue = "";
    };
    window.addEventListener("beforeunload", warnBeforeUnload);
    return () => window.removeEventListener("beforeunload", warnBeforeUnload);
  }, [changeVersion, reviewDirty, savedVersion]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (!(event.ctrlKey || event.metaKey)) return;
      if (isReadOnly) return;
      if (event.key.toLowerCase() === "s") {
        event.preventDefault();
        void saveAnnotation(false);
      } else if (event.key.toLowerCase() === "z" && event.shiftKey) {
        event.preventDefault();
        viewerRef.current?.redo();
      } else if (event.key.toLowerCase() === "z") {
        event.preventDefault();
        viewerRef.current?.undo();
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [isReadOnly, saveAnnotation]);

  const saveReview = useCallback(async (): Promise<boolean> => {
    if (isReadOnly) return true;
    setReviewSaving(true);
    setActionError("");
    try {
      const updated = await api.updateReview(caseId, {
        status: reviewStatus,
        qc_checklist: qc,
        reviewer_notes: notes,
      });
      setDetail(updated);
      setReviewStatus(updated.review_status);
      savedReviewSnapshotRef.current = reviewSnapshot(
        updated.qc_checklist,
        updated.reviewer_notes,
        updated.review_status,
      );
      return true;
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "Review could not be saved.");
      return false;
    } finally {
      setReviewSaving(false);
    }
  }, [caseId, isReadOnly, notes, qc, reviewStatus]);

  useEffect(() => {
    if (
      loading ||
      isReadOnly ||
      !reviewDirty ||
      reviewStatus === "Approved" ||
      reviewSaving ||
      finalizing
    ) {
      return;
    }
    const timer = window.setTimeout(() => {
      void saveReview();
    }, 1250);
    return () => window.clearTimeout(timer);
  }, [
    finalizing,
    isReadOnly,
    loading,
    reviewDirty,
    reviewSaving,
    reviewStatus,
    saveReview,
  ]);

  const flushBeforeLeave = useCallback(async (): Promise<boolean> => {
    if (changeVersionRef.current > savedVersionRef.current) {
      const saved = await saveAnnotation(false);
      if (!saved) return false;
    }
    if (
      !isReadOnly &&
      currentReviewSnapshot !== savedReviewSnapshotRef.current
    ) {
      const reviewSaved = await saveReview();
      if (!reviewSaved) return false;
    }
    return saveViewerStateNow();
  }, [currentReviewSnapshot, isReadOnly, saveAnnotation, saveReview, saveViewerStateNow]);

  useEffect(() => {
    registerNavigationGuard(flushBeforeLeave);
    return () => registerNavigationGuard(null);
  }, [flushBeforeLeave, registerNavigationGuard]);

  const leaveWorkspace = async () => {
    if (!(await flushBeforeLeave())) return;
    onBack();
  };

  const finalize = async () => {
    if (!detail) return;
    setFinalizing(true);
    setActionError("");
    try {
      if (changeVersionRef.current > savedVersion) {
        const saved = await saveAnnotation(false);
        if (!saved) return;
      }
      const reviewSaved = await saveReview();
      if (!reviewSaved) return;
      const approved = await api.finalize(caseId);
      setDetail(approved);
      setQc(approved.qc_checklist);
      setNotes(approved.reviewer_notes);
      setReviewStatus(approved.review_status);
      savedReviewSnapshotRef.current = reviewSnapshot(
        approved.qc_checklist,
        approved.reviewer_notes,
        approved.review_status,
      );
      setSaveMessage("Finalized for review");
    } catch (caught) {
      setActionError(caught instanceof Error ? caught.message : "Case could not be finalized.");
    } finally {
      setFinalizing(false);
    }
  };

  if (loading) return <main className="center-state">Loading case…</main>;
  if (loadError || !detail || !protocol) {
    return (
      <main className="center-state error-state">
        <h1>Case unavailable</h1>
        <p>{loadError || "The case was not found."}</p>
        <button className="button secondary" type="button" onClick={onBack}>Back to cases</button>
      </main>
    );
  }

  const canFinalize =
    !isReadOnly &&
    detail.validation_status !== "FAILED" &&
    Boolean(detail.latest_revision) &&
    allQcChecksPass(qc) &&
    !saving &&
    !reviewSaving &&
    !finalizing;
  const labels = annotationLabels(protocol);
  const activeTool =
    isReadOnly && (tool === "draw" || tool === "erase" || tool === "fill")
      ? "navigate"
      : tool;

  return (
    <main className="workspace">
      <header className="workspace-header">
        <div className="workspace-identity">
          <button className="text-button" type="button" onClick={() => void leaveWorkspace()}>← Cases</button>
          <div>
            <span className="eyebrow">Annotation workspace</span>
            <h1>{detail.case_label}</h1>
          </div>
          <span className={`status status-${detail.validation_status.toLowerCase()}`}>{detail.validation_status}</span>
          <span className="status status-neutral">{detail.review_status}</span>
        </div>
        <div className="header-actions">
          <span className={changeVersion > savedVersion ? "save-state dirty" : "save-state"} aria-live="polite">
            {saveMessage}
          </span>
          <button
            className="button"
            type="button"
            onClick={() => void saveAnnotation(false)}
            disabled={!viewerReady || saving || isReadOnly}
          >
            {saving ? "Saving…" : "Save"}
          </button>
          {detail.latest_revision ? (
            <a className="button secondary" href={api.exportUrl(caseId)} download>Export</a>
          ) : (
            <button className="button secondary" type="button" disabled title="Save an annotation before exporting.">Export</button>
          )}
        </div>
      </header>

      <section className="viewer-toolbar" aria-label="Viewer and annotation controls">
        <div className="toolbar-group">
          <span className="toolbar-label">View</span>
          <button
            className="tool-button"
            type="button"
            aria-pressed={viewMode === "multiplanar"}
            onClick={() => setViewMode("multiplanar")}
          >
            Axial · Coronal · Sagittal
          </button>
          <button
            className="tool-button"
            type="button"
            aria-pressed={viewMode === "render"}
            onClick={() => {
              setViewMode("render");
              setTool("navigate");
            }}
          >
            3D
          </button>
        </div>

        <div className="toolbar-group">
          <span className="toolbar-label">Tool</span>
          {(["navigate", "window", "draw", "erase", "fill"] as ViewerTool[]).map((item) => (
            <button
              key={item}
              className="tool-button"
              type="button"
              aria-pressed={activeTool === item}
              disabled={Boolean(isReadOnly && item !== "navigate" && item !== "window")}
              onClick={() => {
                setTool(item);
                if (item !== "navigate" && item !== "window") setViewMode("multiplanar");
              }}
            >
              {item === "window" ? "Window/level" : item[0].toUpperCase() + item.slice(1)}
            </button>
          ))}
        </div>

        <div className="toolbar-group">
          <span className="toolbar-label">History</span>
          <button className="tool-button" type="button" disabled={!canUndo || isReadOnly} onClick={() => viewerRef.current?.undo()} title="Undo (Ctrl+Z)">Undo</button>
          <button className="tool-button" type="button" disabled={!canRedo || isReadOnly} onClick={() => viewerRef.current?.redo()} title="Redo (Ctrl+Shift+Z)">Redo</button>
          <button className="tool-button" type="button" onClick={() => viewerRef.current?.resetView()}>Reset view</button>
        </div>
      </section>

      <section className="annotation-toolbar" aria-label="Drawing settings">
        <label>
          <span>Label</span>
          <select value={selectedLabel} disabled={isReadOnly} onChange={(event) => setSelectedLabel(Number(event.target.value))}>
            {labels.map((label) => (
              <option value={label.value} key={label.value}>{label.value}. {label.name}</option>
            ))}
          </select>
        </label>
        <label>
          <span>Brush <output>{brushSize} vox</output></span>
          <input type="range" min="1" max="12" value={brushSize} disabled={isReadOnly} onChange={(event) => setBrushSize(Number(event.target.value))} />
        </label>
        <label>
          <span>Overlay <output>{Math.round(overlayOpacity * 100)}%</output></span>
          <input type="range" min="0.1" max="1" step="0.05" value={overlayOpacity} onChange={(event) => setOverlayOpacity(Number(event.target.value))} />
        </label>
        <label className="inline-check">
          <input type="checkbox" checked={overlayVisible} onChange={(event) => setOverlayVisible(event.target.checked)} />
          Show overlay
        </label>
        <label className="numeric-field">
          <span>Center (HU)</span>
          <input type="number" value={windowCenter} onChange={(event) => setWindowCenter(Number(event.target.value))} />
        </label>
        <label className="numeric-field">
          <span>Width (HU)</span>
          <input type="number" min="1" value={windowWidth} onChange={(event) => setWindowWidth(Math.max(1, Number(event.target.value)))} />
        </label>
        <button className="tool-button preset" type="button" onClick={() => { setWindowCenter(300); setWindowWidth(3000); }}>Bone preset</button>
      </section>

      <div className="workspace-body">
        <section className="viewer-panel" aria-label="CT viewer">
          {viewerError ? <div className="viewer-error alert alert-error">{viewerError}</div> : null}
          <NiiVueViewer
            ref={viewerRef}
            caseId={caseId}
            volumeFileType={detail.volume_file_type}
            protocol={protocol}
            hasAnnotation={Boolean(detail.latest_revision)}
            initialState={detail.viewer_state}
            tool={activeTool}
            selectedLabel={selectedLabel}
            brushSize={brushSize}
            overlayOpacity={overlayOpacity}
            overlayVisible={overlayVisible}
            viewMode={viewMode}
            windowCenter={windowCenter}
            windowWidth={windowWidth}
            onReady={() => {
              viewerReadyRef.current = true;
              setViewerReady(true);
              setSaveMessage("No unsaved changes");
              window.queueMicrotask(() => {
                viewerStateSaveEnabledRef.current = true;
              });
            }}
            onError={setViewerError}
            onDirty={isReadOnly ? () => undefined : markDirty}
            onViewStateChange={() => {
              if (viewerStateSaveEnabledRef.current) {
                setViewerStateVersion((current) => current + 1);
              }
            }}
            onLocationChange={(next) => {
              setLocation(next);
              if (viewerStateSaveEnabledRef.current) {
                setViewerStateVersion((current) => current + 1);
              }
            }}
            onHistoryChange={(undo, redo) => { setCanUndo(undo); setCanRedo(redo); }}
          />
          <footer className="viewer-status">
            <span>Voxel: {location.voxel}</span>
            <span>World position: {location.millimeters} mm</span>
            <span>{viewMode === "render" ? "3D volume and mask surface" : "Linked radiological views"}</span>
          </footer>
        </section>

        <aside className="review-panel" aria-label="Case review">
          <div className="provisional-banner compact" role="note">
            <strong>{detail.annotation_disposition ?? protocol.training_eligibility}</strong>
            <span>Protocol {protocol.protocol_version}</span>
          </div>

          <section>
            <h2>Scan</h2>
            <dl className="metadata-grid">
              <div><dt>Source</dt><dd>{detail.source_type}</dd></div>
              <div><dt>Dimensions</dt><dd>{detail.shape.join(" × ")}</dd></div>
              <div><dt>Spacing</dt><dd>{detail.voxel_spacing_mm.join(" × ")} mm</dd></div>
              <div><dt>Orientation</dt><dd>{detail.orientation_codes.join(" · ")}</dd></div>
              <div><dt>Revision</dt><dd>{detail.latest_revision ?? "None"}</dd></div>
            </dl>
            {detail.warnings.length > 0 ? (
              <div className="warning-list"><strong>Warnings</strong><ul>{detail.warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></div>
            ) : null}
          </section>

          <section>
            <h2>Basic QC</h2>
            {(
              [
                ["coverage", "Complete ethmoid coverage"],
                ["orientation", "Orientation is correct"],
                ["image_quality", "Image quality is acceptable"],
                ["slice_continuity", "Slice continuity is intact"],
              ] as const
            ).map(([key, label]) => (
              <label className="check-row" key={key}>
                <input type="checkbox" checked={qc[key]} disabled={isReadOnly} onChange={(event) => setQc((current) => ({ ...current, [key]: event.target.checked }))} />
                {label}
              </label>
            ))}
          </section>

          <section>
            <h2>Review</h2>
            <label className="field">
              <span>Status</span>
              <select value={reviewStatus} onChange={(event) => setReviewStatus(event.target.value as ReviewStatus)} disabled={detail.review_status === "Approved"}>
                {reviewOptions.map((status) => (
                  <option
                    key={status}
                    disabled={status === "Approved" && detail.state !== "READY_FOR_REVIEW" && detail.state !== "NEEDS_CHANGES" && detail.state !== "APPROVED"}
                  >
                    {status}
                  </option>
                ))}
              </select>
            </label>
            <label className="field">
              <span>Reviewer notes</span>
              <textarea rows={4} value={notes} disabled={isReadOnly} onChange={(event) => setNotes(event.target.value)} placeholder="Record coverage, quality, or requested changes." />
            </label>
            <button className="button secondary full-width" type="button" disabled={reviewSaving || finalizing || isReadOnly} onClick={() => void saveReview()}>
              {reviewSaving ? "Saving review…" : "Save review"}
            </button>
          </section>

          <section>
            <h2>Label measurements</h2>
            {detail.metrics.length === 0 ? <p className="muted">Measurements appear after a valid annotation is saved.</p> : (
              <ul className="metric-list">
                {detail.metrics.map((metric) => (
                  <li key={metric.label_value}>
                    <span className="label-swatch" style={{ backgroundColor: labels.find((label) => label.value === metric.label_value)?.color }} />
                    <span><strong>{metric.label_name}</strong><small>{metric.voxel_count.toLocaleString()} voxels · {metric.volume_mm3.toFixed(1)} mm³</small></span>
                  </li>
                ))}
              </ul>
            )}
          </section>

          {actionError ? <div className="alert alert-error" role="alert">{actionError}</div> : null}
          <button className="button primary full-width" type="button" disabled={!canFinalize} onClick={() => void finalize()}>
            {finalizing ? "Validating…" : "Finalize for review"}
          </button>
          {!canFinalize && detail.review_status !== "Approved" ? (
            <p className="form-hint">Save an annotation, pass all four QC checks, and resolve failed validation before finalizing.</p>
          ) : null}
          <p className="no-score-note">This application does not calculate a complexity score or model prediction.</p>
        </aside>
      </div>
    </main>
  );
}
