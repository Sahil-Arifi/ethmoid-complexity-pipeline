import { afterEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("API client", () => {
  it("loads the case dashboard", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ cases: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await expect(api.listCases()).resolves.toEqual({ cases: [] });
    expect(fetchMock).toHaveBeenCalledWith("/api/cases", undefined);
  });

  it("sends a NIfTI and optional QC report as multipart data", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ case: { case_id: "case-1" } }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);
    const volume = new File([new Uint8Array([1, 2])], "volume.nii", {
      type: "application/octet-stream",
    });
    const report = new File(["{}"], "qc.json", { type: "application/json" });

    await api.importNifti("Case001", volume, report);

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(init.method).toBe("POST");
    expect(init.body).toBeInstanceOf(FormData);
    const form = init.body as FormData;
    expect(form.get("case_label")).toBe("Case001");
    expect((form.get("file") as File).name).toBe("volume.nii");
    expect((form.get("qc_report") as File).name).toBe("qc.json");
  });

  it("omits a blank case label so the backend can generate one", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ case: { case_id: "generated-case" } }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await api.importNifti("", new File([new Uint8Array([1])], "volume.nii"));

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect((init.body as FormData).has("case_label")).toBe(false);
  });

  it("surfaces FastAPI validation errors", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(JSON.stringify({ detail: [{ msg: "Invalid affine" }] }), {
          status: 422,
          headers: { "Content-Type": "application/json" },
        }),
      ),
    );

    await expect(api.getCase("bad")).rejects.toEqual(
      expect.objectContaining<ApiError>({
        name: "ApiError",
        status: 422,
        message: "Invalid affine",
      }),
    );
  });

  it("uploads the annotation mask and serialized viewer state", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ revision: 3, created_at: "2026-08-20T12:00:00Z" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetchMock);

    await api.saveAnnotation("case-1", new Blob([new Uint8Array([1])]), {
      selected_label: 2,
      view_mode: "multiplanar",
    });

    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    const form = init.body as FormData;
    expect((form.get("mask") as File).name).toBe("annotation.nii");
    expect(JSON.parse(String(form.get("viewer_state")))).toEqual({
      selected_label: 2,
      view_mode: "multiplanar",
    });
  });
});
