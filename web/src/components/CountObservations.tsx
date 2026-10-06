import { useEffect, useState } from "react";
import { countingApi, type CountObservation, type ObservationInput } from "../api/counting";

type Point = [number, number, number];
type Props = {
  caseId: string; runId: string; revision: number; selectedIds: number[]; disabled: boolean;
  getPoint: () => Point | undefined; focusPoint: (point: Point) => void;
  onDraftChange: (dirty: boolean) => void;
  onSavingChange?: (saving: boolean) => void;
};
const kinds: Record<ObservationInput["kind"], string> = {
  missed: "Possible missed cell", merged: "Possible merged cells", extra: "Possible extra detection",
  side: "Side assignment question", other: "Other observation",
};
const pointText = (point: Point) => point.map(value => value.toFixed(2)).join(", ");

/** Review evidence stays separate from automatic predictions and count corrections. */
export function CountObservations({ caseId, runId, revision, selectedIds, disabled, getPoint, focusPoint, onDraftChange, onSavingChange }: Props) {
  const [observations, setObservations] = useState<CountObservation[]>([]);
  const [kind, setKind] = useState<ObservationInput["kind"]>("missed");
  const [note, setNote] = useState("");
  const [point, setPoint] = useState<Point | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const dirty = note.length > 0 || point !== null;
  useEffect(() => { onDraftChange(dirty); }, [dirty, onDraftChange]);
  useEffect(() => () => onDraftChange(false), [onDraftChange]);
  useEffect(() => {
    let cancelled = false;
    void countingApi.observations(caseId, runId).then(value => {
      if (!cancelled) { setObservations(value.observations); setError(""); }
    }).catch(value => { if (!cancelled) setError(value instanceof Error ? value.message : "Observations could not be loaded."); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [caseId, runId]);
  const save = async () => {
    if (disabled || busy || !note.trim() || (!selectedIds.length && !point)) return;
    setBusy(true); onSavingChange?.(true); setError(""); setNotice("");
    try {
      const saved = await countingApi.observe(caseId, runId, {
        revision, kind, candidate_ids: selectedIds, point_ras_mm: point, note: note.trim(),
      });
      setObservations(values => [...values, saved]); setNote(""); setPoint(null);
      setNotice("Observation saved. Counts and cell masks are unchanged.");
    } catch (value) { setError(value instanceof Error ? value.message : "The observation could not be saved."); }
    finally { setBusy(false); onSavingChange?.(false); }
  };
  const saved = observations.filter(value => value.revision === revision);
  return <section className="count-observations" aria-label="Diagnostic observations">
    <h2>Diagnostic observations</h2>
    <p className="muted">Record a suspected miss, merge, or side question. These notes preserve the result; they do not change the count or establish reference accuracy.</p>
    <small>Bound to this scan, run, and revision {revision}. Locations come from this CT, not the photos.</small>
    <label className="count-field">Observation type<select aria-label="Observation type" value={kind} disabled={disabled || busy}
      onChange={event => setKind(event.target.value as ObservationInput["kind"])}>
      {Object.entries(kinds).map(([value, label]) => <option key={value} value={value}>{label}</option>)}
    </select></label>
    <p className="muted">Selected candidates: {selectedIds.length ? selectedIds.map(id => `#${id}`).join(", ") : "None"}</p>
    <p className="muted">{point ? `Location: RAS mm ${pointText(point)}` : "Location not captured. Move the crosshair to the area to review."}</p>
    <div className="count-actions"><button className="button" disabled={disabled || busy} onClick={() => {
      const value = getPoint();
      if (value?.length === 3 && value.every(Number.isFinite)) { setPoint([...value]); setError(""); }
      else setError("The viewer location is unavailable. Open the CT and try again.");
    }}>Use crosshair location</button>{point && <button className="button tertiary" disabled={disabled || busy} onClick={() => setPoint(null)}>Clear location</button>}</div>
    <label className="count-field">Observation note<textarea aria-label="Observation note" maxLength={2000} rows={3} value={note}
      disabled={disabled || busy} onChange={event => setNote(event.target.value)} placeholder="Describe what the physician marked and what the model shows." /></label>
    <div className="count-actions"><button className="button" disabled={disabled || busy || loading || !note.trim() || (!point && !selectedIds.length)} onClick={() => void save()}>{busy ? "Saving observation…" : "Save observation"}</button>
      {dirty && <button className="button tertiary" disabled={busy} onClick={() => { setNote(""); setPoint(null); setError(""); }}>Discard observation</button>}</div>
    {error && <p role="alert">{error}</p>}{notice && <p role="status">{notice}</p>}
    {loading ? <p role="status">Loading observations…</p> : saved.length ? <ol className="count-observation-list">{saved.map(value => <li key={value.id}>
      <strong>{kinds[value.kind]}</strong><small> · Revision {value.revision}{value.candidate_ids.length ? ` · ${value.candidate_ids.map(id => `#${id}`).join(", ")}` : ""}</small>
      <p>{value.note}</p>{value.point_ras_mm && <><small>RAS mm {pointText(value.point_ras_mm)}</small><button className="text-button" disabled={busy} onClick={() => focusPoint(value.point_ras_mm!)}>Locate observation</button></>}
    </li>)}</ol> : <p className="muted">No observations saved for this revision.</p>}
    {!loading && !error && observations.length > 0 && <a className="text-button" href={countingApi.observationsExportUrl(caseId, runId)}>Export observations JSON</a>}
  </section>;
}
