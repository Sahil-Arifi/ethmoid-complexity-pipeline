import { request } from "./client";

export type UploadFile = { index: number; name: string; size: number; received_bytes: number; status: string; reason_code?: string | null };
export type UploadStudy = { study_id: string; study_label: string; series_count: number; recommended_series_id?: string | null; case_id?: string | null; preparation_status?: string; count_run_id?: string | null; analysis_status?: string; reason_code?: string | null };
export type UploadSession = {
  upload_id: string; status: "uploading" | "queued" | "indexing" | "preparing" | "completed" | "failed";
  files: UploadFile[]; studies: UploadStudy[];
  issues: { item_index?: number; reason_code?: string; message?: string; status?: string }[];
  limits: { max_expanded_bytes: number; max_files: number };
  reason_code?: string | null;
};
export const uploadName = (file: File) => file.webkitRelativePath || file.name;
export const uploadApi = {
  create: (files: File[]) => request<UploadSession>("/uploads", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ files: files.map(file => ({ name: uploadName(file), size: file.size })) }),
  }),
  get: (id: string) => request<UploadSession>(`/uploads/${encodeURIComponent(id)}`),
  chunk: (id: string, index: number, offset: number, bytes: Blob, signal?: AbortSignal) => request<UploadFile>(`/uploads/${encodeURIComponent(id)}/files/${index}?offset=${offset}`, {
    method: "PUT", headers: { "Content-Type": "application/octet-stream" }, body: bytes, signal,
  }),
  finalize: (id: string) => request<UploadSession>(`/uploads/${encodeURIComponent(id)}/finalize`, { method: "POST" }),
};

/** Replays existing prefix bytes so a resumed session cannot splice a different file into its source. */
export async function transferUpload(session: UploadSession, files: File[], progress: (done: number, total: number) => void, signal?: AbortSignal) {
  const selected = new Map(files.map(file => [uploadName(file), file]));
  if (selected.size !== files.length || files.length !== session.files.length) throw new Error("Select exactly the same folder or ZIP to resume this upload.");
  for (const item of session.files) {
    if (selected.get(item.name)?.size !== item.size) throw new Error("The selected files do not match this upload. Select the original folder or ZIP.");
  }
  const total = files.reduce((sum, file) => sum + file.size, 0);
  let done = 0;
  progress(done, total);
  for (const item of session.files) {
    const file = selected.get(item.name)!;
    if (signal?.aborted) throw new DOMException("Upload paused", "AbortError");
    if (file.size === 0) await uploadApi.chunk(session.upload_id, item.index, 0, file.slice(), signal);
    for (let offset = 0; offset < file.size; offset += 8 * 1024 * 1024) {
      if (signal?.aborted) throw new DOMException("Upload paused", "AbortError");
      const end = Math.min(offset + 8 * 1024 * 1024, file.size);
      await uploadApi.chunk(session.upload_id, item.index, offset, file.slice(offset, end), signal);
      done += end - offset; progress(done, total);
    }
  }
}
