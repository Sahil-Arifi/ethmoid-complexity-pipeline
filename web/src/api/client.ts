import type {
  AnnotationRevision,
  ApiErrorBody,
  CaseDetail,
  CasesResponse,
  DicomImport,
  NiftiImportResponse,
  ReviewUpdate,
  SeriesResponse,
  ViewerState,
  StudiesResponse,
  StudySummary,
  CatalogSeries,
  CatalogStatus,
} from "./types";
import { validateProtocol, type ProtocolDefinition } from "../config/protocol";

const API_ROOT = "/api";

export class ApiError extends Error {
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function responseMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as ApiErrorBody;
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      return body.detail.map((item) => item.msg ?? "Invalid request").join("; ");
    }
    return body.message ?? `Request failed (${response.status})`;
  } catch {
    return `Request failed (${response.status})`;
  }
}

export async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_ROOT}${path}`, init);
  if (!response.ok) throw new ApiError(await responseMessage(response), response.status);
  return (await response.json()) as T;
}

export const api = {
  listStudies(): Promise<StudiesResponse> {
    return request("/studies");
  },
  listStudySeries(studyId: string): Promise<{ series: CatalogSeries[] }> {
    return request(`/studies/${encodeURIComponent(studyId)}/series`);
  },
  catalogStatus(): Promise<CatalogStatus> {
    return request("/catalog/status");
  },
  selectStudySeries(studyId: string, seriesId: string): Promise<{ study: StudySummary }> {
    return request(`/studies/${encodeURIComponent(studyId)}`, {
      method: "PATCH", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ selected_series_id: seriesId }),
    });
  },
  prepareStudySeries(studyId: string, seriesId: string): Promise<{ series: CatalogSeries }> {
    return request(`/studies/${encodeURIComponent(studyId)}/series/${encodeURIComponent(seriesId)}/prepare`, { method: "POST" });
  },
  async getProtocol(): Promise<ProtocolDefinition> {
    return validateProtocol(await request<ProtocolDefinition>("/protocol"));
  },

  async listCases(): Promise<CasesResponse> {
    return request<CasesResponse>("/cases");
  },

  async getCase(caseId: string): Promise<CaseDetail> {
    return request<CaseDetail>(`/cases/${encodeURIComponent(caseId)}`);
  },

  async importNifti(
    caseLabel: string,
    file: File,
    qcReport?: File,
  ): Promise<NiftiImportResponse> {
    const form = new FormData();
    if (caseLabel.trim()) form.append("case_label", caseLabel.trim());
    form.append("file", file, file.name);
    if (qcReport) form.append("qc_report", qcReport, qcReport.name);
    return request<NiftiImportResponse>("/imports/nifti", {
      method: "POST",
      body: form,
    });
  },

  async importDicom(caseLabel: string, files: File[]): Promise<DicomImport> {
    const form = new FormData();
    if (caseLabel.trim()) form.append("case_label", caseLabel.trim());
    files.forEach((file) => form.append("files", file, file.name));
    return request<DicomImport>("/imports/dicom", { method: "POST", body: form });
  },

  async getImportSeries(importId: string): Promise<SeriesResponse> {
    return request<SeriesResponse>(`/imports/${encodeURIComponent(importId)}/series`);
  },

  async deleteImport(importId: string): Promise<void> {
    const response = await fetch(`${API_ROOT}/imports/${encodeURIComponent(importId)}`, {
      method: "DELETE",
    });
    if (!response.ok) throw new ApiError(await responseMessage(response), response.status);
  },

  async convertSeries(importId: string, seriesLabel: string): Promise<NiftiImportResponse> {
    return request<NiftiImportResponse>(
      `/imports/${encodeURIComponent(importId)}/series/${encodeURIComponent(seriesLabel)}/convert`,
      { method: "POST" },
    );
  },

  async saveAnnotation(
    caseId: string,
    mask: Blob,
    viewerState: ViewerState,
  ): Promise<AnnotationRevision> {
    const form = new FormData();
    form.append("mask", mask, "annotation.nii");
    form.append("viewer_state", JSON.stringify(viewerState));
    return request<AnnotationRevision>(`/cases/${encodeURIComponent(caseId)}/annotations`, {
      method: "PUT",
      body: form,
    });
  },

  async updateReview(caseId: string, update: ReviewUpdate): Promise<CaseDetail> {
    return request<CaseDetail>(`/cases/${encodeURIComponent(caseId)}/review`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(update),
    });
  },

  async saveViewerState(
    caseId: string,
    viewerState: ViewerState,
    keepalive = false,
  ): Promise<CaseDetail> {
    return request<CaseDetail>(`/cases/${encodeURIComponent(caseId)}/viewer-state`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(viewerState),
      keepalive,
    });
  },

  async finalize(caseId: string): Promise<CaseDetail> {
    return request<CaseDetail>(`/cases/${encodeURIComponent(caseId)}/finalize`, {
      method: "POST",
    });
  },

  async deleteCase(caseId: string): Promise<void> {
    const response = await fetch(`${API_ROOT}/cases/${encodeURIComponent(caseId)}`, {
      method: "DELETE",
    });
    if (!response.ok) throw new ApiError(await responseMessage(response), response.status);
  },

  volumeUrl(caseId: string): string {
    return `${API_ROOT}/cases/${encodeURIComponent(caseId)}/volume`;
  },

  latestAnnotationUrl(caseId: string): string {
    return `${API_ROOT}/cases/${encodeURIComponent(caseId)}/annotations/latest.nii.gz`;
  },

  exportUrl(caseId: string): string {
    return `${API_ROOT}/cases/${encodeURIComponent(caseId)}/export`;
  },
};
