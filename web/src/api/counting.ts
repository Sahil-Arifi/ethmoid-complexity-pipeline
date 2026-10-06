import { ApiError, request } from "./client";

export type Counts = { left: number; right: number; total: number };
export type Cell = {
  id: number; side: "left" | "right"; centroid_ras_mm: [number, number, number];
  included: boolean; uncertain: boolean; reasons: string[]; origin: string;
  voxel_count: number; air_voxel_count: number; retired?: boolean; corrected?: boolean;
};
export type CountRun = {
  run_id: string; case_id: string; status: "queued" | "running" | "completed" | "failed" | "cancelled";
  latest_revision: number | null; revision: number | null;
  original_counts: Counts | null; counts: Counts | null; cells: Cell[];
  warnings: string[]; failure_code: string | null; message: string | null;
  method_version: string | null; protocol_version: string;
  created_at: string; updated_at: string;
  stage?: string | null;
  outcome?: "automatic_candidate_count" | "assisted_candidate_count" | "needs_review" | "unsupported_input" | "technical_error" | "cancelled" | null;
  assistance?: { mode?: string; boundary_id?: string; parent_run_id?: string } | null;
  scan_profile?: Record<string, unknown> | null;
  processing_plan?: Record<string, unknown> | null;
  model_identity?: Record<string, unknown> | null;
  research_status?: "NOT_ACCURACY_VALIDATED" | null;
  protocol_sha256?: string | null;
  protocol_snapshot_sha256?: string;
};
export type BoundaryRevision = { boundary_id: string; case_id?: string; parent_run_id?: string; created_at?: string; note?: string };
export type ObservationInput = {
  revision: number; kind: "missed" | "merged" | "extra" | "side" | "other";
  candidate_ids: number[]; point_ras_mm: [number, number, number] | null; note: string;
};
export type CountObservation = ObservationInput & { id: string; created_at: string };
export type Correction = {
  type: "add" | "remove" | "merge" | "split" | "update";
  id?: number; ids?: number[]; side?: "left" | "right";
  centroid_ras_mm?: [number, number, number]; included?: boolean;
  uncertain?: boolean; reasons?: string[];
};
const path = (caseId: string, runId?: string) =>
  `/cases/${encodeURIComponent(caseId)}/count-runs${runId ? `/${encodeURIComponent(runId)}` : ""}`;
export const countingApi = {
  list: (caseId: string) => request<{ runs: CountRun[] }>(path(caseId)),
  start: (caseId: string, key: string) => request<CountRun>(path(caseId), {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ idempotency_key: key }),
  }),
  get: (caseId: string, runId: string) => request<CountRun>(path(caseId, runId)),
  observations: (caseId: string, runId: string) => request<{ observations: CountObservation[] }>(`${path(caseId, runId)}/observations`),
  observe: (caseId: string, runId: string, value: ObservationInput) => request<CountObservation>(`${path(caseId, runId)}/observations`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(value),
  }),
  observationsExportUrl: (caseId: string, runId: string) => `/api${path(caseId, runId)}/observations/export`,
  retry: (caseId: string, runId: string, key: string) => request<CountRun>(`${path(caseId, runId)}/retry`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ idempotency_key: key }),
  }),
  boundaries: (caseId: string, runId: string) => request<{ boundaries: BoundaryRevision[] }>(`${path(caseId, runId)}/boundaries`),
  cancel: (caseId: string, runId: string) => request<CountRun>(`${path(caseId, runId)}/cancel`, { method: "POST" }),
  saveBoundary: (caseId: string, runId: string, mask: Blob, note: string) => {
    const body = new FormData();
    body.append("mask", mask, mask instanceof File ? mask.name : "boundary.nii");
    body.append("note", note);
    return request<BoundaryRevision>(`${path(caseId, runId)}/boundaries`, { method: "POST", body });
  },
  boundaryMask: (caseId: string, runId: string, boundaryId: string) =>
    `/api${path(caseId, runId)}/boundaries/${encodeURIComponent(boundaryId)}/mask`,
  assisted: (caseId: string, runId: string, boundaryId: string, key: string) => request<CountRun>(`${path(caseId, runId)}/assisted-runs`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ boundary_id: boundaryId, idempotency_key: key }),
  }),
  latest: (caseId: string, runId: string) => request<CountRun>(`${path(caseId, runId)}/revisions/latest`),
  correct: (caseId: string, runId: string, parent: number, operation: Correction, mask?: Blob) => {
    const metadata = { parent_revision: parent, operation };
    if (mask) {
      const body = new FormData();
      body.append("metadata", JSON.stringify(metadata)); body.append("mask", mask, "selection.nii");
      return request<CountRun>(`${path(caseId, runId)}/revisions`, { method: "POST", body });
    }
    return request<CountRun>(`${path(caseId, runId)}/revisions`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(metadata),
    });
  },
  artifact: (caseId: string, runId: string, kind: "air" | "cavities" | "region", revision: number) =>
    `/api${path(caseId, runId)}/artifacts/${kind}?revision=${revision}`,
  exportUrl: (caseId: string, runId: string, revision: number) =>
    `/api${path(caseId, runId)}/export?revision=${revision}`,
  download: async (caseId: string, runId: string, revision: number) => {
    const response = await fetch(`/api${path(caseId, runId)}/export?revision=${revision}`);
    if (!response.ok || !response.headers.get("content-type")?.startsWith("application/zip")) {
      throw new ApiError("The result could not be exported. Please try again.", response.status);
    }
    return response.blob();
  },
};
