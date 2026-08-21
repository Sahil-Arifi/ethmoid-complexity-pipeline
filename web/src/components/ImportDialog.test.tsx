import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { ImportDialog } from "./ImportDialog";

vi.mock("../api/client", () => ({
  api: {
    importNifti: vi.fn(),
    importDicom: vi.fn(),
    convertSeries: vi.fn(),
    deleteImport: vi.fn(),
  },
}));

describe("ImportDialog", () => {
  beforeEach(() => vi.clearAllMocks());

  it("imports a NIfTI with a neutral label", async () => {
    const user = userEvent.setup();
    vi.mocked(api.importNifti).mockResolvedValue({ case: { case_id: "nifti-case" } } as never);
    const imported = vi.fn();
    render(<ImportDialog onClose={vi.fn()} onImported={imported} />);

    const volume = new File([new Uint8Array([1])], "scan.nii", {
      type: "application/octet-stream",
    });
    const caseLabel = screen.getByRole("textbox", { name: /Neutral case label/ });
    await user.clear(caseLabel);
    await user.type(caseLabel, "Case010");
    await user.upload(screen.getByLabelText(/NIfTI volume/), volume);
    await user.click(screen.getByRole("button", { name: "Import scan" }));

    await waitFor(() => expect(api.importNifti).toHaveBeenCalledWith("Case010", volume, undefined));
    expect(imported).toHaveBeenCalledWith("nifti-case");
  });

  it("contains keyboard focus and restores the previous control when closed", async () => {
    const user = userEvent.setup();
    const previous = document.createElement("button");
    previous.textContent = "Open import";
    document.body.appendChild(previous);
    previous.focus();
    const rendered = render(<ImportDialog onClose={vi.fn()} onImported={vi.fn()} />);

    const dialog = await screen.findByRole("dialog", { name: "Import scan" });
    await waitFor(() => expect(dialog).toHaveFocus());
    await user.tab({ shift: true });
    expect(screen.getByRole("button", { name: "Import scan" })).toHaveFocus();
    await user.tab();
    expect(screen.getByRole("button", { name: "Close import dialog" })).toHaveFocus();

    rendered.unmount();
    expect(previous).toHaveFocus();
    previous.remove();
  });

  it("requires series selection when a DICOM folder contains multiple series", async () => {
    const user = userEvent.setup();
    vi.mocked(api.importDicom).mockResolvedValue({
      import_id: "import-1",
      status: "AWAITING_SERIES_SELECTION",
      series: [
        { series_label: "Series001", image_count: 10, rows: 512, columns: 512, spacing_mm: [0.6, 0.6, 1], preflight_status: "PASS", issues: [], warnings: [] },
        { series_label: "Series002", image_count: 224, rows: 512, columns: 512, spacing_mm: [0.625, 0.625, 0.625], kernel: "BONEPLUS", preflight_status: "PASS", issues: [], warnings: [] },
      ],
    });
    vi.mocked(api.convertSeries).mockResolvedValue({ case: { case_id: "dicom-case" } } as never);
    const imported = vi.fn();
    render(<ImportDialog onClose={vi.fn()} onImported={imported} />);

    await user.click(screen.getByRole("radio", { name: "DICOM folder" }));
    const fileInput = screen.getByLabelText(/DICOM folder/, { selector: "input[type=file]" });
    expect(fileInput).toHaveAttribute("webkitdirectory");
    expect(fileInput).toHaveAttribute("directory");
    const files = [new File(["a"], "1.dcm"), new File(["b"], "2.dcm")];
    await user.upload(fileInput, files);
    await user.click(screen.getByRole("button", { name: "Import scan" }));

    expect(await screen.findByRole("heading", { name: "Select one DICOM series" })).toBeVisible();
    await user.click(screen.getByRole("radio", { name: /Series002/ }));
    await user.click(screen.getByRole("button", { name: "Convert selected series" }));

    await waitFor(() => expect(api.convertSeries).toHaveBeenCalledWith("import-1", "Series002"));
    expect(imported).toHaveBeenCalledWith("dicom-case");
  });

  it("does not convert a sole series that failed metadata preflight", async () => {
    const user = userEvent.setup();
    vi.mocked(api.importDicom).mockResolvedValue({
      import_id: "failed-import",
      status: "FAILED",
      series: [{
        series_label: "Series001",
        image_count: 10,
        rows: 0,
        columns: 0,
        spacing_mm: [],
        preflight_status: "FAIL",
        issues: ["Duplicate slice positions"],
        warnings: [],
      }],
    });
    render(<ImportDialog onClose={vi.fn()} onImported={vi.fn()} />);

    await user.click(screen.getByRole("radio", { name: "DICOM folder" }));
    await user.upload(
      screen.getByLabelText(/DICOM folder/, { selector: "input[type=file]" }),
      new File(["a"], "1.dcm"),
    );
    await user.click(screen.getByRole("button", { name: "Import scan" }));

    expect(await screen.findByText("Duplicate slice positions")).toBeVisible();
    expect(screen.getByRole("button", { name: "Convert selected series" })).toBeDisabled();
    expect(api.convertSeries).not.toHaveBeenCalled();
  });
});
