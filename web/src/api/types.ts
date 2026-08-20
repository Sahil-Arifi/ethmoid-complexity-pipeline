export type SourceType = "DICOM" | "NIFTI";

export type CaseState =
  | "IMPORTING"
  | "VALIDATION_FAILED"
  | "READY"
  | "ANNOTATING"
  | "READY_FOR_REVIEW"
  | "NEEDS_CHANGES"
  | "APPROVED";

export type ValidationStatus = "VERIFIED" | "UNVERIFIED" | "FAILED";

export type ReviewStatus =
  | "Draft"
  | "Ready for review"
  | "Needs changes"
  | "Approved";

export type QcChecklist = {
  coverage: boolean;
  orientation: boolean;
  image_quality: boolean;
  slice_continuity: boolean;
};

export type ViewerState = {
  crosshair?: [number, number, number];
  pan_zoom?: [number, number, number, number];
  view_mode?: "multiplanar" | "render";
  selected_label?: number;
  brush_size?: number;
  overlay_opacity?: number;
  active_tool?: "navigate" | "window" | "draw" | "erase" | "fill";
  window_center?: number;
  window_width?: number;
  render_azimuth?: number;
  render_elevation?: number;
  volume_scale?: number;
};

export type LabelMetric = {
  label_value: number;
  label_name: string;
  voxel_count: number;
  volume_mm3: number;
};

export type CaseSummary = {
  case_id: string;
  case_label: string;
  source_type: SourceType;
  state: CaseState;
  validation_status: ValidationStatus;
  review_status: ReviewStatus;
  updated_at: string;
  latest_revision?: number | null;
};

export type CaseDetail = CaseSummary & {
  warnings: string[];
  shape: [number, number, number];
  voxel_spacing_mm: [number, number, number];
  orientation_codes: [string, string, string];
  volume_file_type: "nii" | "nii.gz";
  qc_checklist: QcChecklist;
  reviewer_notes: string;
  viewer_state?: ViewerState | null;
  metrics: LabelMetric[];
  protocol_version?: string;
  annotation_disposition?: string;
};

export type SeriesSummary = {
  series_label: string;
  image_count: number;
  rows: number;
  columns: number;
  spacing_mm: number[];
  kernel?: string | null;
  description?: string | null;
  preflight_status: "PASS" | "FAIL";
  issues: string[];
  warnings: string[];
};

export type DicomImport = {
  import_id: string;
  status: string;
  series: SeriesSummary[];
};

export type AnnotationRevision = {
  revision: number;
  created_at: string;
  voxel_counts?: Record<string, number>;
  metrics?: {
    labels: Array<{
      value: number;
      name: string;
      voxel_count: number;
      volume_mm3: number;
    }>;
  };
};

export type CasesResponse = { cases: CaseSummary[] };

export type SeriesResponse = { series: SeriesSummary[] };

export type NiftiImportResponse = { case: CaseDetail };

export type ReviewUpdate = {
  status: ReviewStatus;
  qc_checklist: QcChecklist;
  reviewer_notes: string;
};

export type ApiErrorBody = {
  detail?: string | { msg?: string }[];
  message?: string;
};
