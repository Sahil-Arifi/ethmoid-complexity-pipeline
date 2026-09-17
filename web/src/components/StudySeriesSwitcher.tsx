import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { CatalogContext, CatalogSeries } from "../api/types";
import { canPrepare, orderedSeries, preparationLabel, seriesDescription } from "./catalog";
import "./catalog.css";

type Props = {
  context: CatalogContext;
  disabled?: boolean;
  beforeSwitch: () => Promise<boolean>;
  onOpenCase: (caseId: string) => void;
  onSwitchBusy?: (busy: boolean) => void;
};

export function StudySeriesSwitcher({ context, disabled, beforeSwitch, onOpenCase, onSwitchBusy }: Props) {
  const [series, setSeries] = useState<CatalogSeries[]>([]);
  const [selected, setSelected] = useState(context.series_id);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [refresh, setRefresh] = useState(0);
  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const load = async () => {
      try {
        const result = await api.listStudySeries(context.study_id);
        if (cancelled) return;
        setSeries(result.series); setError("");
        if (result.series.some(item => item.preparation_status === "queued" || item.preparation_status === "preparing")) timer = setTimeout(load, 2000);
      } catch (caught) {
        if (!cancelled) { setError(caught instanceof Error ? caught.message : "Series could not be loaded."); timer = setTimeout(load, 5000); }
      }
    };
    void load();
    return () => { cancelled = true; clearTimeout(timer); };
  }, [context.study_id, refresh]);
  const target = series.find(item => item.series_id === selected);
  const open = async () => {
    if (!target?.case_id || target.preparation_status !== "ready" || busy || disabled) return;
    setBusy(true); onSwitchBusy?.(true); setError("");
    try {
      if (!(await beforeSwitch())) { setSelected(context.series_id); return; }
      // Persist only after the existing correction/annotation guard accepts.
      await api.selectStudySeries(context.study_id, target.series_id);
      onOpenCase(target.case_id);
    } catch (caught) { setError(caught instanceof Error ? caught.message : "The series could not be opened."); }
    finally { setBusy(false); onSwitchBusy?.(false); }
  };
  const prepare = async () => {
    if (!target || busy || disabled) return;
    setBusy(true); setError("");
    try { await api.prepareStudySeries(context.study_id, target.series_id); setRefresh(value => value + 1); }
    catch (caught) { setError(caught instanceof Error ? caught.message : "The series could not be prepared."); }
    finally { setBusy(false); }
  };
  return <section className="series-switcher" aria-label="Study series">
    <label><span>Series to open</span><select aria-label="Series to open" value={selected} disabled={busy || disabled || !series.length} onChange={event => setSelected(event.target.value)}>
      {!series.length && <option value={context.series_id}>{context.series_label}</option>}
      {orderedSeries(series).map(item => <option key={item.series_id} value={item.series_id}>{seriesDescription(item)}{item.preparation_status === "unavailable" ? " · Unavailable" : ""}</option>)}
    </select></label>
    {target?.preparation_status === "ready" ? <button className="button compact" disabled={busy || disabled || selected === context.series_id} onClick={() => void open()}>Open selected series</button>
      : target && (canPrepare(target) || target.preparation_status === "queued") ? <button className="button compact" disabled={busy || disabled} onClick={() => void prepare()}>{target.preparation_status === "unavailable" ? "Retry preparation" : target.preparation_status === "queued" ? "Prepare next" : "Prepare series"}</button> : null}
    <span>Viewing {context.series_label}</span>
    {target && selected !== context.series_id && <p className="series-feedback">{preparationLabel(target)}{target.reason ? `. ${target.reason}` : ""}</p>}
    {error && <p className="series-feedback" role="alert">{error}</p>}
  </section>;
}
