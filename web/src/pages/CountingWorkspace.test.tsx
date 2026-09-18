import { forwardRef, useEffect, useImperativeHandle } from "react";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError } from "../api/client";
import { countingApi, type CountRun } from "../api/counting";
import type { CaseDetail } from "../api/types";
import { CountingWorkspace } from "./CountingWorkspace";
import { readySeries, studyFixture, thickSeries } from "../testCatalog";

const handles = vi.hoisted(() => ({ focus: vi.fn(), clear: vi.fn(), prepare: vi.fn(), region: vi.fn() }));
vi.mock("../api/client", async importOriginal => ({ ...(await importOriginal<object>()), api: { getCase: vi.fn(), getProtocol: vi.fn(), saveViewerState: vi.fn(), listStudySeries: vi.fn(), selectStudySeries: vi.fn(), prepareStudySeries: vi.fn() } }));
vi.mock("../api/counting", async importOriginal => ({ ...(await importOriginal<object>()), countingApi: {
  ...(await importOriginal<{ countingApi: object }>()).countingApi,
  list: vi.fn(), get: vi.fn(), start: vi.fn(), correct: vi.fn(), retry: vi.fn(), cancel: vi.fn(), saveBoundary: vi.fn(), assisted: vi.fn(),
} }));
vi.mock("../viewer/NiiVueViewer", () => ({ NiiVueViewer: forwardRef(function Mock(props: { onReady: () => void; onDirty: () => void; instanceOverlayUrl?: string }, ref) {
  useEffect(() => props.onReady(), []);
  useImperativeHandle(ref, () => ({ focusWorld: handles.focus, clearSelectionMask: handles.clear,
    prepareCellMask: handles.prepare, prepareRegionMask: handles.region, getWorldPosition: () => [-12, 5, 7], getViewerState: () => ({ window_center: 300, window_width: 3000 }),
    exportMask: async () => new Blob(["synthetic mask"]),
  }));
  return <div><output data-testid="overlay-url">{props.instanceOverlayUrl}</output><button onClick={props.onDirty}>Paint fixture</button></div>;
}) }));

const detail: CaseDetail = { case_id: "case-1", case_label: "Case901", source_type: "NIFTI", state: "READY", validation_status: "UNVERIFIED", review_status: "Draft", updated_at: "2026-09-01", warnings: [], shape: [10, 10, 10], voxel_spacing_mm: [1, 1, 1], orientation_codes: ["R", "A", "S"], volume_file_type: "nii", qc_checklist: { coverage: false, orientation: false, image_quality: false, slice_continuity: false }, reviewer_notes: "", metrics: [] };
const result: CountRun = { run_id: "run-1", case_id: "case-1", status: "completed", latest_revision: 0, revision: 0, counts: { left: 1, right: 1, total: 2 }, original_counts: { left: 1, right: 1, total: 2 }, cells: [
  { id: 300, side: "left", centroid_ras_mm: [-12, 5, 7], included: true, uncertain: true, reasons: ["Incomplete septum"], origin: "automatic", voxel_count: 22, air_voxel_count: 0 },
  { id: 301, side: "right", centroid_ras_mm: [12, 5, 7], included: true, uncertain: false, reasons: [], origin: "automatic", voxel_count: 25, air_voxel_count: 10 },
], warnings: [], failure_code: null, message: null, method_version: "synthetic", protocol_version: "experimental", created_at: "2026-09-01T12:00:00Z", updated_at: "2026-09-01T12:00:00Z" };

describe("3D counting workspace", () => {
  beforeEach(() => {
    vi.clearAllMocks(); window.history.replaceState({}, "", "/cases/case-1");
    vi.mocked(api.getCase).mockResolvedValue(detail);
    vi.mocked(api.saveViewerState).mockResolvedValue(detail);
    vi.mocked(api.getProtocol).mockResolvedValue({ schema_version: 1, protocol_version: "synthetic", training_eligibility: "PROVISIONAL_NOT_FOR_TRAINING", labels: [] });
    vi.mocked(countingApi.list).mockResolvedValue({ runs: [result] });
    vi.mocked(countingApi.get).mockResolvedValue(result);
    vi.mocked(countingApi.correct).mockResolvedValue({ ...result, latest_revision: 1, revision: 1 });
  });
  function mount() { const register = vi.fn(); const back = vi.fn(); render(<CountingWorkspace caseId="case-1" onBack={back} registerNavigationGuard={register} />); return { register, back }; }

  it("runs without outlines or seeds and prevents duplicate in-flight starts", async () => {
    vi.mocked(countingApi.list).mockResolvedValue({ runs: [] });
    vi.mocked(countingApi.start).mockResolvedValue({ ...result, status: "running", counts: null, original_counts: null, cells: [], latest_revision: null });
    mount();
    const start = await screen.findByRole("button", { name: "Run automatic count" });
    expect(screen.queryByRole("button", { name: "Add cell" })).not.toBeInTheDocument();
    await userEvent.click(start);
    expect(countingApi.start).toHaveBeenCalledWith("case-1", expect.any(String));
    expect(screen.getByRole("button", { name: "Counting…" })).toBeDisabled();
    expect(screen.queryByTestId("total-count")).not.toBeInTheDocument();
  });

  it("displays failure as unavailable, never zero", async () => {
    vi.mocked(countingApi.get).mockResolvedValue({ ...result, status: "failed", counts: null, original_counts: null, cells: [], latest_revision: null, failure_code: "LOCALIZATION_FAILED", message: "Bony boundary unavailable." });
    mount();
    expect(await screen.findByText("Count unavailable")).toBeVisible();
    expect(screen.getByText(/Bony boundary unavailable/)).toBeVisible();
    expect(screen.queryByTestId("total-count")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export" })).toBeDisabled();
  });

  it("supports drawing anatomical regions after localization failure and starts an explicitly assisted child", async () => {
    const failed: CountRun = { ...result, status: "failed", outcome: "needs_review", counts: null, original_counts: null, cells: [], latest_revision: null, failure_code: "SPHENOID_UNRESOLVED" };
    vi.mocked(countingApi.get).mockResolvedValue(failed);
    vi.mocked(countingApi.saveBoundary).mockResolvedValue({ boundary_id: "boundary-1" });
    vi.mocked(countingApi.assisted).mockResolvedValue({ ...failed, run_id: "assisted-1", status: "queued", outcome: null, assistance: { boundary_id: "boundary-1", parent_run_id: "run-1" } });
    mount(); await userEvent.click(await screen.findByRole("button", { name: "Review anatomical boundaries" }));
    expect(screen.getByRole("combobox", { name: "Anatomical region" })).toHaveValue("1");
    expect(screen.getByTestId("overlay-url")).toBeEmptyDOMElement();
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Anatomical region" }), "2");
    fireEvent.click(screen.getByRole("button", { name: "Paint fixture" }));
    expect(screen.getByRole("button", { name: "Run assisted count" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Save anatomical boundaries" }));
    expect(countingApi.saveBoundary).toHaveBeenCalledWith("case-1", "run-1", expect.any(Blob), "");
    await userEvent.click(screen.getByRole("button", { name: "Run assisted count" }));
    expect(countingApi.assisted).toHaveBeenCalledWith("case-1", "run-1", "boundary-1", expect.any(String));
    expect(screen.getByText("Assisted result")).toBeVisible();
    expect(screen.getByRole("button", { name: "Review parent attempt" })).toBeVisible();
    expect(countingApi.correct).not.toHaveBeenCalled();
  });

  it("invalidates a saved boundary when it is painted again and guards navigation", async () => {
    vi.mocked(countingApi.saveBoundary).mockResolvedValue({ boundary_id: "boundary-1" });
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false); const { back } = mount();
    await userEvent.click(await screen.findByRole("button", { name: "Review anatomical boundaries" }));
    expect(handles.region).toHaveBeenCalledWith(expect.stringContaining("/artifacts/region?revision=0"));
    await userEvent.click(screen.getByRole("button", { name: "Save anatomical boundaries" }));
    expect(screen.getByRole("button", { name: "Run assisted count" })).toBeEnabled();
    fireEvent.click(screen.getByRole("button", { name: "Paint fixture" }));
    expect(screen.getByRole("button", { name: "Run assisted count" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Cases" }));
    expect(confirm).toHaveBeenCalled(); expect(back).not.toHaveBeenCalled();
    expect(screen.queryByRole("button", { name: "Add cell" })).not.toBeInTheDocument(); confirm.mockRestore();
  });

  it("never displays an imported region mask before source alignment is validated", async () => {
    vi.mocked(countingApi.get).mockResolvedValue({ ...result, status: "failed", counts: null, cells: [], failure_code: "SPHENOID_UNRESOLVED" });
    vi.mocked(countingApi.saveBoundary).mockRejectedValue(new ApiError("Boundary geometry does not match source", 422));
    mount(); await userEvent.click(await screen.findByRole("button", { name: "Review anatomical boundaries" }));
    await userEvent.upload(screen.getByLabelText("Import anatomical boundary mask"), new File(["bad"], "regions.nii"));
    expect(await screen.findByText("Boundary geometry does not match source")).toBeVisible();
    expect(handles.region).not.toHaveBeenCalled();
    expect(countingApi.assisted).not.toHaveBeenCalled();
  });

  it("separates technical retries from anatomical review and exposes cancellation", async () => {
    const failed: CountRun = { ...result, status: "failed", outcome: "technical_error", counts: null, cells: [], failure_code: "INTERRUPTED" };
    vi.mocked(countingApi.get).mockResolvedValue(failed);
    vi.mocked(countingApi.retry).mockResolvedValue({ ...failed, run_id: "retry-1", status: "running", outcome: null, stage: "localizing" });
    vi.mocked(countingApi.cancel).mockResolvedValue({ ...failed, run_id: "retry-1", status: "failed", outcome: "cancelled" });
    mount(); await userEvent.click(await screen.findByRole("button", { name: "Retry interrupted analysis" }));
    expect(countingApi.retry).toHaveBeenCalledWith("case-1", "run-1");
    expect(screen.getByRole("list", { name: "Analysis progress" })).toHaveTextContent("Locating anatomy");
    await userEvent.click(screen.getByRole("button", { name: "Cancel count" }));
    expect(countingApi.cancel).toHaveBeenCalledWith("case-1", "retry-1");
    expect(screen.getByText("Count cancelled")).toBeVisible(); expect(screen.queryByTestId("total-count")).not.toBeInTheDocument();
  });

  it("locates IDs above 255 in physical coordinates and preserves original counts", async () => {
    vi.mocked(countingApi.get).mockResolvedValue({ ...result, latest_revision: 2, counts: { left: 2, right: 1, total: 3 } });
    mount();
    await userEvent.click(await screen.findByRole("button", { name: /#300 · left/ }));
    expect(handles.focus).toHaveBeenCalledWith([-12, 5, 7]);
    expect(screen.getByText(/No visible air/)).toBeVisible();
    expect(screen.getByTestId("total-count")).toHaveTextContent("3");
    await userEvent.click(screen.getByRole("checkbox", { name: "Show original prediction" }));
    expect(screen.getByTestId("total-count")).toHaveTextContent("2");
    expect(screen.getByTestId("overlay-url")).toHaveTextContent("revision=0");
    expect(screen.getByRole("link", { name: "Export revision 0" })).toHaveAttribute("href", expect.stringContaining("revision=0"));
    expect(screen.queryByRole("button", { name: "Remove cell" })).not.toBeInTheDocument();
  });

  it("submits a split mask bound to the selected cell and parent revision", async () => {
    mount();
    await userEvent.click(await screen.findByRole("button", { name: /#300 · left/ }));
    await userEvent.click(screen.getByRole("button", { name: "Split cell" }));
    fireEvent.click(screen.getByRole("button", { name: "Paint fixture" }));
    await userEvent.click(screen.getByRole("button", { name: "Apply painted correction" }));
    await waitFor(() => expect(countingApi.correct).toHaveBeenCalledWith("case-1", "run-1", 0, { type: "split", id: 300 }, expect.any(Blob)));
    expect(await screen.findByText(/Correction saved as revision 1/)).toBeVisible();
    expect(handles.clear).toHaveBeenCalled();
  });

  it("loads an editable binary cell mask and guards pending corrections", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const { back } = mount();
    await userEvent.click(await screen.findByRole("button", { name: /#300 · left/ }));
    await userEvent.click(screen.getByRole("button", { name: "Edit boundary" }));
    expect(handles.prepare).toHaveBeenCalledWith(expect.stringContaining("/artifacts/cavities?revision=0"), 300);
    await userEvent.click(screen.getByRole("button", { name: "Cases" }));
    expect(confirm).toHaveBeenCalled(); expect(back).not.toHaveBeenCalled();
    confirm.mockRestore();
  });

  it("never rebases a stale painted draft after a revision conflict", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(true);
    mount();
    await userEvent.click(await screen.findByRole("button", { name: /#300 · left/ }));
    await userEvent.click(screen.getByRole("button", { name: "Edit boundary" }));
    fireEvent.click(screen.getByRole("button", { name: "Paint fixture" }));
    vi.mocked(countingApi.correct).mockRejectedValueOnce(new ApiError("Stale revision", 409));
    vi.mocked(countingApi.get).mockResolvedValue({ ...result, latest_revision: 1, revision: 1 });
    await userEvent.click(screen.getByRole("button", { name: "Apply painted correction" }));
    expect(await screen.findByText(/This draft targets an earlier revision/)).toBeVisible();
    expect(screen.getByRole("button", { name: "Apply painted correction" })).toBeDisabled();
    expect(countingApi.correct).toHaveBeenCalledWith("case-1", "run-1", 0, { type: "update", id: 300 }, expect.any(Blob));
    expect(countingApi.correct).toHaveBeenCalledTimes(1);
    await userEvent.click(screen.getByRole("button", { name: "Discard correction" }));
    await userEvent.click(screen.getByRole("button", { name: /#300 · left/ }));
    await userEvent.click(screen.getByRole("button", { name: "Edit boundary" }));
    expect(handles.prepare).toHaveBeenLastCalledWith(expect.stringContaining("revision=1"), 300);
    fireEvent.click(screen.getByRole("button", { name: "Paint fixture" }));
    expect(screen.getByRole("button", { name: "Apply painted correction" })).not.toBeDisabled();
    confirm.mockRestore();
  });

  it("adds an explicitly manual count-only cell at the crosshair", async () => {
    mount();
    await userEvent.click(await screen.findByRole("button", { name: "Add cell" }));
    await userEvent.click(screen.getByRole("button", { name: "Add at crosshair" }));
    expect(countingApi.correct).toHaveBeenCalledWith("case-1", "run-1", 0, expect.objectContaining({ type: "add", side: "left", centroid_ras_mm: [-12, 5, 7], uncertain: true }), undefined);
  });

  it("explains a withheld posterior boundary count without presenting a raw code", async () => {
    vi.mocked(countingApi.get).mockResolvedValue({ ...result, status: "failed", counts: null, original_counts: null, cells: [], latest_revision: null, failure_code: "SPHENOID_AMBIGUOUS", message: "The count did not complete." });
    mount();
    expect(await screen.findByText(/boundary between the ethmoid cells and the sphenoid sinus/)).toBeVisible();
    expect(screen.queryByText("SPHENOID_AMBIGUOUS")).not.toBeInTheDocument();
    expect(screen.queryByTestId("total-count")).not.toBeInTheDocument();
  });

  it("opens the current volume of the same series only after the retained-volume edit guard accepts", async () => {
    const context = { study_id: "study-1", study_label: "Case003", series_id: readySeries.series_id, series_label: readySeries.series_label, spacing_mm: readySeries.spacing_mm, slice_thickness_mm: 0.625, resolution_class: readySeries.resolution_class, warnings: ["This retained volume belongs to an earlier source version."] };
    vi.mocked(api.getCase).mockResolvedValue({ ...detail, catalog: context });
    vi.mocked(api.listStudySeries).mockResolvedValue({ series: [{ ...readySeries, case_id: "current-volume" }] });
    vi.mocked(api.selectStudySeries).mockResolvedValue({ study: studyFixture });
    const open = vi.fn(); const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const props = { caseId: "case-1", onBack: vi.fn(), onOpenCase: open, registerNavigationGuard: vi.fn() };
    const mounted = render(<CountingWorkspace {...props} />);
    await userEvent.click(await screen.findByRole("button", { name: "Add cell" }));
    const openCurrent = screen.getByRole("button", { name: "Open selected series" });
    expect(openCurrent).toBeEnabled();
    await userEvent.click(openCurrent);
    expect(confirm).toHaveBeenCalled();
    expect(api.selectStudySeries).not.toHaveBeenCalled();
    expect(open).not.toHaveBeenCalled();
    expect(screen.getByRole("heading", { name: "Add a cell" })).toBeVisible();
    expect(screen.getByTestId("total-count")).toHaveTextContent("2");
    confirm.mockReturnValue(true);
    await userEvent.click(openCurrent);
    expect(api.selectStudySeries).toHaveBeenCalledWith("study-1", readySeries.series_id);
    expect(open).toHaveBeenCalledWith("current-volume");
    vi.mocked(api.getCase).mockResolvedValue({ ...detail, case_id: "current-volume", catalog: { ...context, warnings: [] } });
    vi.mocked(countingApi.list).mockResolvedValue({ runs: [] });
    mounted.rerender(<CountingWorkspace {...props} caseId="current-volume" />);
    await screen.findByRole("button", { name: "Run automatic count" });
    await waitFor(() => expect(screen.getByRole("button", { name: "Open selected series" })).toBeDisabled());
    expect(screen.queryByTestId("total-count")).not.toBeInTheDocument();
    expect(screen.getByTestId("overlay-url")).toBeEmptyDOMElement();
    expect(screen.queryByRole("heading", { name: "Add a cell" })).not.toBeInTheDocument();
    expect(countingApi.start).not.toHaveBeenCalled();
    expect(countingApi.correct).not.toHaveBeenCalled();
    confirm.mockRestore();
  });

  it("shows the current same-series preparation failure while reviewing retained history", async () => {
    const reason = "The source files changed. Prepare this series again.";
    vi.mocked(api.getCase).mockResolvedValue({ ...detail, catalog: { study_id: "study-1", study_label: "Case003", series_id: readySeries.series_id, series_label: readySeries.series_label, spacing_mm: readySeries.spacing_mm, slice_thickness_mm: 0.625, resolution_class: readySeries.resolution_class, warnings: [] } });
    vi.mocked(api.listStudySeries).mockResolvedValue({ series: [{ ...readySeries, case_id: null, preparation_status: "unavailable", retryable: true, reason }] });
    render(<CountingWorkspace caseId="case-1" onBack={vi.fn()} onOpenCase={vi.fn()} registerNavigationGuard={vi.fn()} />);
    expect(await screen.findByText(`Unavailable. ${reason}`)).toBeVisible();
    expect(screen.getByRole("button", { name: "Retry preparation" })).toBeEnabled();
    expect(screen.getByTestId("total-count")).toHaveTextContent("2");
    expect(countingApi.start).not.toHaveBeenCalled();
  });

  it("guards series selection before persisting it and clears masks/counts on the next case", async () => {
    const context = { study_id: "study-1", study_label: "Case003", series_id: "series-thin", series_label: "Series007", spacing_mm: readySeries.spacing_mm, slice_thickness_mm: 0.625, resolution_class: readySeries.resolution_class, warnings: [] };
    vi.mocked(api.getCase).mockResolvedValue({ ...detail, catalog: context });
    vi.mocked(api.listStudySeries).mockResolvedValue({ series: [readySeries, thickSeries] });
    vi.mocked(api.selectStudySeries).mockResolvedValue({ study: studyFixture });
    const open = vi.fn(); const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    const props = { caseId: "case-1", onBack: vi.fn(), onOpenCase: open, registerNavigationGuard: vi.fn() };
    const mounted = render(<CountingWorkspace {...props} />);
    expect(await screen.findByRole("heading", { name: "Case003 · Series007" })).toBeVisible();
    await userEvent.click(await screen.findByRole("button", { name: "Add cell" }));
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Series to open" }), "series-thick");
    await userEvent.click(screen.getByRole("button", { name: "Open selected series" }));
    expect(confirm).toHaveBeenCalled(); expect(api.selectStudySeries).not.toHaveBeenCalled(); expect(open).not.toHaveBeenCalled();
    expect(screen.getByRole("combobox", { name: "Series to open" })).toHaveValue("series-thin");
    confirm.mockReturnValue(true);
    await userEvent.selectOptions(screen.getByRole("combobox", { name: "Series to open" }), "series-thick");
    await userEvent.click(screen.getByRole("button", { name: "Open selected series" }));
    expect(api.selectStudySeries).toHaveBeenCalledWith("study-1", "series-thick"); expect(open).toHaveBeenCalledWith("case-2");
    vi.mocked(api.getCase).mockResolvedValue({ ...detail, case_id: "case-2", catalog: { ...context, series_id: "series-thick", series_label: "Series001", slice_thickness_mm: 2.5, resolution_class: "outside_protocol" } });
    vi.mocked(countingApi.list).mockResolvedValue({ runs: [] });
    mounted.rerender(<CountingWorkspace {...props} caseId="case-2" />);
    expect(await screen.findByRole("heading", { name: "Case003 · Series001" })).toBeVisible();
    expect(screen.queryByTestId("total-count")).not.toBeInTheDocument();
    expect(screen.getByTestId("overlay-url")).toBeEmptyDOMElement();
    expect(screen.queryByRole("heading", { name: "Add a cell" })).not.toBeInTheDocument();
    expect(screen.getByText(/outside the thin native CT protocol \(slice thickness 2.5 mm\)/)).toBeVisible();
    expect(countingApi.start).not.toHaveBeenCalled(); confirm.mockRestore();
  });
});
