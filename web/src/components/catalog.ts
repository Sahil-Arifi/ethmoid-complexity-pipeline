import type { CatalogContext, CatalogSeries, StudySummary } from "../api/types";

export function orderedSeries(series: CatalogSeries[]): CatalogSeries[] {
  return [...series].sort((a, b) => Number(b.recommended) - Number(a.recommended) || a.series_label.localeCompare(b.series_label, undefined, { numeric: true }));
}
export function chosenSeries(study: StudySummary): CatalogSeries | undefined {
  return study.series.find(s => s.series_id === study.selected_series_id) ?? study.series.find(s => s.series_id === study.recommended_series_id) ?? orderedSeries(study.series)[0];
}
export function seriesDescription(series: CatalogSeries): string {
  const thickness = series.slice_thickness_mm === null ? "thickness unknown" : `${series.slice_thickness_mm} mm thickness`;
  const kind = { native: "Native CT", mpr: "Reformatted CT", scout: "Scout", projection: "Projection", unknown: "Scan" }[series.source_kind];
  return `${series.series_label} · ${thickness} · ${kind}${series.kernel ? ` · ${series.kernel}` : ""}${series.recommended ? " · Recommended" : ""}`;
}
export function preparationLabel(series: CatalogSeries): string {
  return { not_prepared: "Not prepared", queued: "Queued for preparation", preparing: "Preparing volume…", ready: "Conversion verified · Ready to attempt", unavailable: "Unavailable" }[series.preparation_status];
}
export function countLabel(series: CatalogSeries): string {
  if (series.count_outcome === "technical_error") return "Technical issue · Open to retry";
  if (series.count_outcome === "needs_review") return "Needs anatomical review";
  if (series.count_outcome === "unsupported_input") return "Unsupported input · Open for reason";
  if (series.count_status === "completed" && series.counts) return `Count complete · L ${series.counts.left} / R ${series.counts.right} / Total ${series.counts.total}`;
  return { not_run: "Not counted", queued: "Count queued", running: "Counting…", completed: "Count unavailable", failed: "Count withheld · Open for reason", cancelled: "Count cancelled" }[series.count_status];
}
export function qualityNotes(series: Pick<CatalogContext, "resolution_class" | "slice_thickness_mm" | "warnings">): string[] {
  const notes = [...series.warnings];
  if (series.resolution_class === "outside_protocol") notes.unshift(`This series is outside the thin native CT protocol${series.slice_thickness_mm === null ? "" : ` (slice thickness ${series.slice_thickness_mm} mm)`}. Small walls may be unclear; any count is exploratory.`);
  if (series.resolution_class === "unknown") notes.unshift("Scan resolution is unconfirmed. Review image quality before attempting a count.");
  return [...new Set(notes)];
}
export function canPrepare(series: CatalogSeries): boolean {
  return series.preparation_status === "not_prepared" || (series.preparation_status === "unavailable" && series.retryable);
}
