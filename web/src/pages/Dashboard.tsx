import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { CaseSummary, StudiesResponse, StudySummary } from "../api/types";
import { BrandMark } from "../components/BrandMark";
import { FirstRunGuide } from "../components/FirstRunGuide";
import { ImportDialog } from "../components/ImportDialog";
import { canPrepare, chosenSeries, countLabel, orderedSeries, preparationLabel, qualityNotes, seriesDescription } from "../components/catalog";
import "../components/catalog.css";

type Props = { onOpenCase: (caseId: string) => void };

export function Dashboard({ onOpenCase }: Props) {
  const [data, setData] = useState<StudiesResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [showImport, setShowImport] = useState(false);
  const [guideOpen, setGuideOpen] = useState(false);
  const requestId = useRef(0);
  const mounted = useRef(true);
  const load = useCallback(async () => {
    const id = ++requestId.current;
    try {
      const result = await api.listStudies();
      if (mounted.current && id === requestId.current) { setData(result); setError(""); }
    } catch (caught) {
      if (mounted.current && id === requestId.current) setError(caught instanceof Error ? caught.message : "Studies could not be loaded.");
    } finally { if (mounted.current) setLoading(false); }
  }, []);
  useEffect(() => {
    mounted.current = true; void load();
    return () => { mounted.current = false; requestId.current++; };
  }, [load]);
  const progressing = data && (data.catalog.status === "indexing" || data.catalog.status === "preparing" || data.catalog.queued > 0 || data.catalog.preparing > 0);
  useEffect(() => {
    if (!progressing || busy) return;
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => { await load(); if (!stopped) timer = setTimeout(poll, 2000); };
    timer = setTimeout(poll, 2000);
    return () => { stopped = true; clearTimeout(timer); };
  }, [busy, load, progressing]);
  const act = async (study: StudySummary, action: () => Promise<void>) => {
    if (busy) return;
    requestId.current++; setBusy(study.study_id); setError("");
    try { await action(); await load(); }
    catch (caught) { if (mounted.current) setError(caught instanceof Error ? caught.message : "The study could not be updated."); }
    finally { if (mounted.current) setBusy(null); }
  };
  const deleteDraft = async (item: CaseSummary) => {
    if (!window.confirm(`Delete draft ${item.case_label}? This cannot be undone.`)) return;
    requestId.current++;
    try {
      await api.deleteCase(item.case_id);
      setData(current => current && ({ ...current, unlinked_cases: current.unlinked_cases.filter(candidate => candidate.case_id !== item.case_id) }));
    } catch (caught) { setError(caught instanceof Error ? caught.message : "Draft could not be deleted."); }
  };
  return <main className="dashboard">
    <FirstRunGuide open={guideOpen} onOpenChange={setGuideOpen} version="2" />
    <header className="dashboard-header">
      <div className="dashboard-brand"><BrandMark /><div><span className="eyebrow">Ethmoid Atlas · Local workspace</span><h1>Count ethmoid cells in 3D.</h1><p>Choose a study and series, run the count, then review and export.</p></div></div>
      <div className="dashboard-actions"><button className="button tertiary help-button" onClick={() => setGuideOpen(true)}><span aria-hidden="true">?</span> Help</button><button className="button primary" onClick={() => setShowImport(true)}><span aria-hidden="true">＋</span> Import scan</button></div>
    </header>
    <section className="workflow-strip" aria-label="Application workflow"><div><span>01</span><strong>Choose</strong><small>Study and CT series</small></div><i aria-hidden="true">→</i><div><span>02</span><strong>Count</strong><small>Unique cells across all slices</small></div><i aria-hidden="true">→</i><div><span>03</span><strong>Review</strong><small>Correct cells and export</small></div></section>
    <div className="provisional-banner" role="note"><span className="provisional-icon" aria-hidden="true">◇</span><span><strong>Experimental prototype.</strong> Preliminary cell counts require review. Clinical accuracy is not established.</span></div>
    {error && <div className="alert alert-error" role="alert"><span>{error}</span><button className="text-button" onClick={() => void load()}>Try again</button></div>}
    {loading && <div className="center-state">Loading studies…</div>}
    {data && data.catalog.status !== "unconfigured" && <section className="catalog-progress" aria-label="Scan preparation" role="status">
      <strong>{data.catalog.status === "indexing" ? "Reading the study catalog…" : progressing ? "Preparing CT volumes in the background" : data.catalog.status === "error" ? "Study catalog needs attention" : "Study catalog ready"}</strong>
      <p>{data.catalog.total_studies} studies · {data.catalog.total_series} series · {data.catalog.ready} ready · {data.catalog.preparing} preparing · {data.catalog.queued} queued · {data.catalog.unavailable} unavailable{data.catalog.not_prepared > 0 ? ` · ${data.catalog.not_prepared} not prepared` : ""}</p>
      <p>Preparation verifies the volume. It does not count cells. Counting starts only when you select Run automatic count.</p>
      {data.catalog.reason && <p>{data.catalog.reason}</p>}
    </section>}
    {data && data.studies.length > 0 && <section aria-labelledby="study-list-title"><div className="section-heading"><h2 id="study-list-title">Cases</h2><span>{data.studies.length} studies</span></div>
      <div className="case-table-wrap"><table className="case-table study-table"><thead><tr><th scope="col">Study</th><th scope="col">Selected series</th><th scope="col">Preparation / count</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead><tbody>
        {data.studies.map(study => {
          const selected = chosenSeries(study);
          const ready = selected?.preparation_status === "ready" && !!selected.case_id;
          const notes = selected ? qualityNotes(selected) : [];
          return <tr key={study.study_id} data-testid={`study-${study.study_id}`}><td><strong>{study.study_label}</strong><small>{study.series_count} series</small></td>
            <td><div className="series-details"><select aria-label={`Series for ${study.study_label}`} value={selected?.series_id ?? ""} disabled={!!busy || !study.series.length} onChange={event => { const id = event.target.value; void act(study, async () => { await api.selectStudySeries(study.study_id, id); }); }}>
              {orderedSeries(study.series).map(series => <option key={series.series_id} value={series.series_id}>{seriesDescription(series)}{series.preparation_status === "unavailable" ? " · Unavailable" : ""}</option>)}
            </select>{selected && <small>{selected.image_count} images{selected.spacing_mm ? ` · spacing ${selected.spacing_mm.join(" × ")} mm` : ""}</small>}{notes.length > 0 && <details className="series-warning"><summary>Scan quality notes ({notes.length})</summary><ul>{notes.map((note, i) => <li key={i}>{note}</li>)}</ul></details>}</div></td>
            <td className="study-state">{selected ? <><span className={`status ${ready ? "status-verified" : "status-neutral"}`}>{preparationLabel(selected)}</span><small>{countLabel(selected)}</small>{selected.reason && <small>{selected.reason}</small>}</> : <span>No series available</span>}</td>
            <td><div className="row-actions">{ready ? <button className="button compact" disabled={!!busy} onClick={() => onOpenCase(selected!.case_id!)}>Open series</button> : selected && canPrepare(selected) ? <button className="button compact" disabled={!!busy} onClick={() => void act(study, async () => { await api.prepareStudySeries(study.study_id, selected.series_id); })}>{selected.preparation_status === "unavailable" ? "Retry preparation" : "Prepare series"}</button> : selected?.preparation_status === "queued" ? <button className="button compact" disabled={!!busy} onClick={() => void act(study, async () => { await api.prepareStudySeries(study.study_id, selected.series_id); })}>Prepare next</button> : <button className="button compact" disabled>{selected?.preparation_status === "preparing" ? "Preparing…" : "Unavailable"}</button>}</div></td>
          </tr>;
        })}
      </tbody></table></div></section>}
    {data && data.unlinked_cases.length > 0 && <section aria-labelledby="imported-list-title"><div className="section-heading"><h2 id="imported-list-title">Imported scans</h2><span>{data.unlinked_cases.length} separate imports</span></div><p className="muted">These scans are not linked to the study catalog.</p><div className="case-table-wrap"><table className="case-table"><thead><tr><th scope="col">Scan</th><th scope="col">Source</th><th scope="col">Validation</th><th scope="col">Review</th><th scope="col">Last saved</th><th scope="col"><span className="sr-only">Actions</span></th></tr></thead><tbody>{data.unlinked_cases.map(item => <tr key={item.case_id}><td><button className="case-link" onClick={() => onOpenCase(item.case_id)}>{item.case_label}</button><small>{item.latest_revision ? `Annotation revision ${item.latest_revision}` : "Not annotated"}</small></td><td>{item.source_type}</td><td><span className={`status status-${item.validation_status.toLowerCase()}`}>{item.validation_status}</span></td><td>{item.review_status}</td><td>{new Date(item.updated_at).toLocaleString()}</td><td><div className="row-actions"><button className="button compact" onClick={() => onOpenCase(item.case_id)}>Open case</button>{item.latest_revision ? <a className="button compact secondary" href={api.exportUrl(item.case_id)} download>Export</a> : null}{item.review_status === "Draft" && item.state !== "APPROVED" && <button className="text-button danger" onClick={() => void deleteDraft(item)}>Delete draft</button>}</div></td></tr>)}</tbody></table></div></section>}
    {!loading && data && !data.studies.length && !data.unlinked_cases.length && <section className="empty-state"><h2>No scans available</h2><p>Import a verified DICOM series or a NIfTI volume to begin.</p><button className="button primary" onClick={() => setShowImport(true)}>Import first scan</button></section>}
    {showImport && <ImportDialog onClose={() => setShowImport(false)} onImported={caseId => { setShowImport(false); onOpenCase(caseId); }} />}
  </main>;
}
