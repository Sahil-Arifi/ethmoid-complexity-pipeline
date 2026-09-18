import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { transferUpload, uploadApi, type UploadSession } from "../api/uploads";
import { ImportDialog } from "./ImportDialog";

vi.mock("../api/client", () => ({ api: { importNifti: vi.fn() } }));
vi.mock("../api/uploads", () => ({ uploadApi: { create: vi.fn(), get: vi.fn(), finalize: vi.fn() }, transferUpload: vi.fn() }));
const session: UploadSession = { upload_id: "upload-1", status: "uploading", files: [{ index: 0, name: "1.dcm", size: 1, received_bytes: 0, status: "pending" }], studies: [], issues: [], limits: { max_expanded_bytes: 20 * 1024 ** 3, max_files: 50000 } };

describe("ImportDialog", () => {
  beforeEach(() => {
    vi.clearAllMocks(); localStorage.clear();
    vi.mocked(uploadApi.create).mockResolvedValue(session);
    vi.mocked(transferUpload).mockResolvedValue(undefined);
    vi.mocked(uploadApi.finalize).mockResolvedValue({ ...session, status: "preparing" });
  });

  it("keeps NIfTI import available without automatically counting unverified volumes", async () => {
    const user = userEvent.setup();
    vi.mocked(api.importNifti).mockResolvedValue({ case: { case_id: "nifti-case" } } as never);
    const imported = vi.fn(); render(<ImportDialog onClose={vi.fn()} onImported={imported} />);
    await user.click(screen.getByRole("radio", { name: "NIfTI" }));
    const volume = new File(["a"], "scan.nii");
    await user.type(screen.getByRole("textbox", { name: /Neutral case label/ }), "Case010");
    await user.upload(screen.getByLabelText(/NIfTI volume/), volume);
    await user.click(screen.getByRole("button", { name: "Import scan" }));
    await waitFor(() => expect(api.importNifti).toHaveBeenCalledWith("Case010", volume, undefined));
    expect(imported).toHaveBeenCalledWith("nifti-case");
    expect(uploadApi.create).not.toHaveBeenCalled();
  });

  it("contains keyboard focus and restores the previous control when closed", async () => {
    const user = userEvent.setup();
    const previous = document.createElement("button"); previous.textContent = "Open import";
    document.body.appendChild(previous); previous.focus();
    const rendered = render(<ImportDialog onClose={vi.fn()} onImported={vi.fn()} />);
    const dialog = await screen.findByRole("dialog", { name: "Import scan" });
    await waitFor(() => expect(dialog).toHaveFocus()); await user.tab({ shift: true });
    expect(screen.getByRole("button", { name: "Import scan" })).toHaveFocus(); await user.tab();
    expect(screen.getByRole("button", { name: "Close import dialog" })).toHaveFocus();
    rendered.unmount(); expect(previous).toHaveFocus(); previous.remove();
  });

  it("finalizes all selected DICOM files without discarding other series", async () => {
    const user = userEvent.setup(); const updated = vi.fn(); const imported = vi.fn();
    render(<ImportDialog onClose={vi.fn()} onImported={imported} onCatalogUpdated={updated} />);
    const input = screen.getByLabelText("DICOM folder", { selector: "input[type=file]" });
    expect(input).toHaveAttribute("webkitdirectory");
    const files = [new File(["a"], "1.dcm"), new File(["b"], "2.dcm")];
    await user.upload(input, files); await user.click(screen.getByRole("button", { name: "Import scan" }));
    expect(await screen.findByText("Preparing verified volumes")).toBeVisible();
    expect(uploadApi.create).toHaveBeenCalledWith(files);
    expect(transferUpload).toHaveBeenCalledWith(session, files, expect.any(Function), expect.any(AbortSignal));
    expect(uploadApi.finalize).toHaveBeenCalledWith("upload-1"); expect(updated).toHaveBeenCalled();
    expect(imported).not.toHaveBeenCalled();
    expect(localStorage.getItem("ethmoid.pending-upload")).toBe("upload-1");
  });

  it("resumes a saved ZIP session and preserves a failed transfer for retry", async () => {
    const user = userEvent.setup(); localStorage.setItem("ethmoid.pending-upload", "upload-1");
    vi.mocked(uploadApi.get).mockResolvedValue(session);
    vi.mocked(transferUpload).mockRejectedValueOnce(new Error("Connection interrupted"));
    render(<ImportDialog onClose={vi.fn()} onImported={vi.fn()} />);
    await screen.findByRole("button", { name: "Resume upload" });
    await user.click(screen.getByRole("radio", { name: "DICOM ZIP" }));
    const archive = new File(["z"], "study.zip", { type: "application/zip" });
    await user.upload(screen.getByLabelText("DICOM ZIP archive"), archive);
    await user.click(screen.getByRole("button", { name: "Resume upload" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Connection interrupted");
    expect(uploadApi.create).not.toHaveBeenCalled(); expect(uploadApi.finalize).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Resume upload" }));
    expect(await screen.findByText("Preparing verified volumes")).toBeVisible();
    expect(transferUpload).toHaveBeenLastCalledWith(session, [archive], expect.any(Function), expect.any(AbortSignal));
  });

  it("shows unavailable series and file reasons without claiming counting succeeded", async () => {
    localStorage.setItem("ethmoid.pending-upload", "upload-1");
    vi.mocked(uploadApi.get).mockResolvedValue({ ...session, status: "completed", studies: [{ study_id: "study-1", study_label: "Case001", series_count: 2, preparation_status: "unavailable", analysis_status: "unsupported_input", reason_code: "SCOUT_ONLY" }], issues: [{ item_index: 0, reason_code: "NON_DICOM" }] });
    render(<ImportDialog onClose={vi.fn()} onImported={vi.fn()} />);
    expect(await screen.findByText("Upload cataloged")).toBeVisible();
    expect(screen.getByText(/a completed upload does not mean the count succeeded/)).toBeVisible();
    expect(screen.getByText("Analysis: unsupported input")).toBeVisible();
    expect(screen.queryByRole("button", { name: "Open Case001" })).not.toBeInTheDocument();
  });
});
