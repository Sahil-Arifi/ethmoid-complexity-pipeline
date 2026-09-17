import type { CatalogSeries, StudiesResponse, StudySummary } from "./api/types";

export const readySeries: CatalogSeries = {
  series_id: "series-thin", series_label: "Series007", case_id: "case-1", preparation_status: "ready", reason: null,
  spacing_mm: [0.5, 0.5, 0.625], slice_thickness_mm: 0.625, image_count: 224, kernel: "BONE", source_kind: "native",
  resolution_class: "within_protocol", warnings: [], recommended: true, retryable: false, count_status: "not_run", counts: null,
};
export const thickSeries: CatalogSeries = { ...readySeries, series_id: "series-thick", series_label: "Series001", case_id: "case-2", slice_thickness_mm: 2.5, resolution_class: "outside_protocol", recommended: false };
export const studyFixture: StudySummary = { study_id: "study-1", study_label: "Case003", series_count: 2, selected_series_id: "series-thin", recommended_series_id: "series-thin", series: [thickSeries, readySeries] };
export const catalogFixture: StudiesResponse = { studies: [studyFixture], catalog: { status: "ready", total_studies: 1, total_series: 2, ready: 2, queued: 0, preparing: 0, unavailable: 0, not_prepared: 0 }, unlinked_cases: [] };
