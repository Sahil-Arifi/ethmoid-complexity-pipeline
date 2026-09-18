import { afterEach, describe, expect, it, vi } from "vitest";
import { transferUpload, uploadApi, type UploadSession } from "./uploads";

afterEach(() => vi.restoreAllMocks());
const makeSession = (file: File, received: number): UploadSession => ({ upload_id: "upload-1", status: "uploading", files: [{ index: 0, name: file.name, size: file.size, received_bytes: received, status: "uploading" }], studies: [], issues: [], limits: { max_expanded_bytes: 20 * 1024 ** 3, max_files: 50000 } });
describe("resumable DICOM transport", () => {
  it("rechecks completed prefix bytes before continuing a resumed file", async () => {
    const file = new File([new Uint8Array(9 * 1024 * 1024)], "study.zip");
    const session = makeSession(file, 8 * 1024 * 1024);
    const chunk = vi.spyOn(uploadApi, "chunk").mockResolvedValue(session.files[0]); const progress = vi.fn();
    await transferUpload(session, [file], progress);
    expect(chunk).toHaveBeenCalledTimes(2);
    expect(chunk.mock.calls.map(call => call[2])).toEqual([0, 8 * 1024 * 1024]);
    expect(chunk.mock.calls[0][3].size).toBe(8 * 1024 * 1024);
    expect(progress).toHaveBeenLastCalledWith(file.size, file.size);
  });
  it("stops at a changed prefix and never sends later bytes", async () => {
    const file = new File([new Uint8Array(9 * 1024 * 1024)], "study.zip");
    const chunk = vi.spyOn(uploadApi, "chunk").mockRejectedValue(new Error("Previously uploaded bytes do not match"));
    await expect(transferUpload(makeSession(file, 8 * 1024 * 1024), [file], vi.fn())).rejects.toThrow("do not match");
    expect(chunk).toHaveBeenCalledTimes(1);
  });
  it("rejects different manifests and pauses before transmission", async () => {
    const file = new File(["a"], "study.zip"); const chunk = vi.spyOn(uploadApi, "chunk");
    await expect(transferUpload(makeSession(file, 0), [new File(["ab"], "study.zip")], vi.fn())).rejects.toThrow("do not match");
    const abort = new AbortController(); abort.abort();
    await expect(transferUpload(makeSession(file, 0), [file], vi.fn(), abort.signal)).rejects.toThrow("Upload paused");
    expect(chunk).not.toHaveBeenCalled();
  });
  it("accounts for an empty source file with an explicit empty chunk", async () => {
    const file = new File([], "empty.dcm"); const session = makeSession(file, 0);
    const chunk = vi.spyOn(uploadApi, "chunk").mockResolvedValue(session.files[0]);
    await transferUpload(session, [file], vi.fn());
    expect(chunk).toHaveBeenCalledTimes(1); expect(chunk.mock.calls[0][3].size).toBe(0);
  });
});
