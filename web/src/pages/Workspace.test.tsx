import { forwardRef, useEffect, useImperativeHandle } from "react";
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import type { CaseDetail } from "../api/types";
import { Workspace } from "./Workspace";

const viewerReadiness = vi.hoisted(() => ({
  automatic: true,
  notify: () => {},
}));

vi.mock("../api/client", () => ({
  api: {
    getCase: vi.fn(),
    getProtocol: vi.fn(),
    saveAnnotation: vi.fn(),
    saveViewerState: vi.fn(),
    updateReview: vi.fn(),
    finalize: vi.fn(),
    exportUrl: (caseId: string) => `/api/cases/${caseId}/export`,
  },
}));

vi.mock("../viewer/NiiVueViewer", () => ({
  NiiVueViewer: forwardRef(function MockViewer(
    props: {
      onReady: () => void;
      onDirty: () => void;
      onWindowChange: (center: number, width: number) => void;
      windowCenter: number;
      windowWidth: number;
    },
    ref,
  ) {
    useImperativeHandle(ref, () => ({
      exportMask: async () => new Blob([new Uint8Array([1, 2, 3])]),
      getViewerState: () => ({
        view_mode: "multiplanar",
        selected_label: 1,
        window_center: props.windowCenter,
        window_width: props.windowWidth,
      }),
      undo: vi.fn(),
      redo: vi.fn(),
      canUndo: () => false,
      canRedo: () => false,
      resetView: vi.fn(),
    }));
    useEffect(() => {
      viewerReadiness.notify = props.onReady;
      if (viewerReadiness.automatic) props.onReady();
    }, []);
    return (
      <div>
        <div role="img" aria-label="Mock linked medical viewer" />
        <output data-testid="mock-window">{props.windowCenter}/{props.windowWidth}</output>
        <button type="button" onClick={props.onDirty}>Simulate drawing</button>
        <button type="button" onClick={() => props.onWindowChange(120, 1800)}>Simulate windowing</button>
      </div>
    );
  }),
}));

const caseDetail: CaseDetail = {
  case_id: "case-1",
  case_label: "Case003",
  source_type: "DICOM",
  state: "ANNOTATING",
  validation_status: "VERIFIED",
  review_status: "Draft",
  updated_at: "2026-08-20T12:00:00Z",
  latest_revision: 2,
  warnings: [],
  shape: [512, 512, 224],
  voxel_spacing_mm: [0.625, 0.625, 0.625],
  orientation_codes: ["R", "A", "S"],
  volume_file_type: "nii",
  qc_checklist: {
    coverage: false,
    orientation: false,
    image_quality: false,
    slice_continuity: false,
  },
  reviewer_notes: "",
  viewer_state: null,
  metrics: [],
};

describe("Workspace", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    viewerReadiness.automatic = true;
    window.localStorage.setItem("ethmoid:first-run-guide", "2");
    vi.mocked(api.getCase).mockResolvedValue(caseDetail);
    vi.mocked(api.getProtocol).mockResolvedValue({
      schema_version: 1,
      protocol_version: "ethmoid-provisional-v1",
      training_eligibility: "PROVISIONAL_NOT_FOR_TRAINING",
      labels: [
        { value: 0, name: "Background", color: "#000000", laterality: null, region: null },
        { value: 1, name: "Left anterior ethmoid", color: "#2F80ED", laterality: "left", region: "anterior_ethmoid" },
        { value: 2, name: "Right anterior ethmoid", color: "#EB5757", laterality: "right", region: "anterior_ethmoid" },
        { value: 3, name: "Left posterior ethmoid", color: "#56CCF2", laterality: "left", region: "posterior_ethmoid" },
        { value: 4, name: "Right posterior ethmoid", color: "#F2994A", laterality: "right", region: "posterior_ethmoid" },
      ],
    });
    vi.mocked(api.saveAnnotation).mockResolvedValue({
      revision: 3,
      created_at: "2026-08-20T12:30:00Z",
      metrics: {
        labels: [
          {
            value: 1,
            name: "Left anterior ethmoid",
            voxel_count: 8,
            volume_mm3: 1.25,
          },
        ],
      },
    });
    vi.mocked(api.saveViewerState).mockResolvedValue(caseDetail);
    vi.mocked(api.updateReview).mockResolvedValue(caseDetail);
    vi.mocked(api.finalize).mockResolvedValue({
      ...caseDetail,
      review_status: "Ready for review",
      state: "READY_FOR_REVIEW",
      qc_checklist: {
        coverage: true,
        orientation: true,
        image_quality: true,
        slice_continuity: true,
      },
    });
  });

  it("exposes the function-first viewer and all four provisional labels", async () => {
    render(<Workspace caseId="case-1" onBack={vi.fn()} registerNavigationGuard={vi.fn()} />);
    expect(await screen.findByRole("heading", { name: "Case003" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Axial · Coronal · Sagittal" })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: "3D" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Draw" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Erase" })).toBeVisible();
    expect(screen.getByRole("button", { name: "Fill" })).toBeVisible();
    expect(screen.getByRole("option", { name: "1. Left anterior ethmoid" })).toBeVisible();
    expect(screen.getByRole("option", { name: "4. Right posterior ethmoid" })).toBeVisible();
    expect(screen.getByText("PROVISIONAL_NOT_FOR_TRAINING")).toBeVisible();
    expect(screen.getByText(/do not calculate a complexity score/i)).toBeVisible();
  });

  it("saves a uint8 NIfTI mask revision with viewer state", async () => {
    const user = userEvent.setup();
    render(<Workspace caseId="case-1" onBack={vi.fn()} registerNavigationGuard={vi.fn()} />);
    await screen.findByRole("heading", { name: "Case003" });
    await user.click(screen.getByRole("button", { name: "Save" }));

    await waitFor(() => expect(api.saveAnnotation).toHaveBeenCalledTimes(1));
    const [, mask, state] = vi.mocked(api.saveAnnotation).mock.calls[0];
    expect(mask).toBeInstanceOf(Blob);
    expect(state).toEqual({
      view_mode: "multiplanar",
      selected_label: 1,
      window_center: 300,
      window_width: 3000,
    });
    expect(await screen.findByText("Saved revision 3")).toBeVisible();
    expect(screen.getByText("8 voxels · 1.3 mm³")).toBeVisible();
  });

  it("offers synchronized brightness and contrast sliders with an exact bone preset", async () => {
    const user = userEvent.setup();
    render(<Workspace caseId="case-1" onBack={vi.fn()} registerNavigationGuard={vi.fn()} />);
    await screen.findByRole("heading", { name: "Case003" });

    const brightness = screen.getByRole("slider", { name: "Brightness" });
    const contrast = screen.getByRole("slider", { name: "Contrast" });
    expect(brightness).toHaveValue("300");
    expect(contrast).toHaveValue("3000");
    expect(screen.getByLabelText("Brightness exact value")).toHaveValue(300);
    expect(screen.getByLabelText("Contrast exact value")).toHaveValue(3000);

    // The viewer enables persistence after its ready microtask. Exercise the
    // debounce explicitly instead of racing a wall-clock timeout under load.
    await act(async () => { await Promise.resolve(); });
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      fireEvent.change(brightness, { target: { value: "-500" } });
      fireEvent.change(contrast, { target: { value: "1200" } });
      expect(screen.getByTestId("mock-window")).toHaveTextContent("-500/1200");
      expect(api.saveViewerState).not.toHaveBeenCalled();
      await act(async () => { await vi.advanceTimersByTimeAsync(751); });
      expect(api.saveViewerState).toHaveBeenCalledWith(
        "case-1",
        expect.objectContaining({ window_center: -500, window_width: 1200 }),
        false,
      );
    } finally {
      vi.useRealTimers();
    }

    await user.click(screen.getByRole("button", { name: "Simulate windowing" }));
    expect(brightness).toHaveValue("120");
    expect(contrast).toHaveValue("1800");

    await user.click(screen.getByRole("button", { name: "Bone · 300/3000" }));
    expect(brightness).toHaveValue("300");
    expect(contrast).toHaveValue("3000");
  });

  it("restores saved brightness and contrast before rendering the viewer", async () => {
    vi.mocked(api.getCase).mockResolvedValue({
      ...caseDetail,
      viewer_state: {
        view_mode: "multiplanar",
        crosshair: [0.5, 0.5, 0.5],
        pan_zoom: [0, 0, 0, 1],
        selected_label: 1,
        brush_size: 4,
        overlay_opacity: 0.65,
        active_tool: "navigate",
        window_center: -240,
        window_width: 1450,
        render_azimuth: 110,
        render_elevation: 20,
        volume_scale: 1,
      },
    });

    render(<Workspace caseId="case-1" onBack={vi.fn()} registerNavigationGuard={vi.fn()} />);
    await screen.findByRole("heading", { name: "Case003" });
    expect(screen.getByRole("slider", { name: "Brightness" })).toHaveValue("-240");
    expect(screen.getByRole("slider", { name: "Contrast" })).toHaveValue("1450");
    expect(screen.getByTestId("mock-window")).toHaveTextContent("-240/1450");
  });

  it("clamps old saved display values and rejects an invalid width", async () => {
    vi.mocked(api.getCase).mockResolvedValue({
      ...caseDetail,
      viewer_state: {
        view_mode: "multiplanar",
        crosshair: [0.5, 0.5, 0.5],
        pan_zoom: [0, 0, 0, 1],
        selected_label: 1,
        brush_size: 4,
        overlay_opacity: 0.65,
        active_tool: "navigate",
        window_center: 9000,
        window_width: 0,
        render_azimuth: 110,
        render_elevation: 20,
        volume_scale: 1,
      },
    });

    render(<Workspace caseId="case-1" onBack={vi.fn()} registerNavigationGuard={vi.fn()} />);
    await screen.findByRole("heading", { name: "Case003" });
    expect(screen.getByRole("slider", { name: "Brightness" })).toHaveValue("2000");
    expect(screen.getByRole("slider", { name: "Contrast" })).toHaveValue("3000");
  });

  it("requires all QC checks before final approval", async () => {
    const user = userEvent.setup();
    render(<Workspace caseId="case-1" onBack={vi.fn()} registerNavigationGuard={vi.fn()} />);
    await screen.findByRole("heading", { name: "Case003" });
    const finalize = screen.getByRole("button", { name: "Finalize for review" });
    expect(finalize).toBeDisabled();

    await user.click(screen.getByRole("checkbox", { name: "Complete ethmoid coverage" }));
    await user.click(screen.getByRole("checkbox", { name: "Orientation is correct" }));
    await user.click(screen.getByRole("checkbox", { name: "Image quality is acceptable" }));
    await user.click(screen.getByRole("checkbox", { name: "Slice continuity is intact" }));
    expect(finalize).toBeEnabled();
    await user.click(finalize);

    await waitFor(() => expect(api.updateReview).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.finalize).toHaveBeenCalledWith("case-1"));
    expect(await screen.findByText("Finalized for review")).toBeVisible();
  });

  it("preserves finalization feedback when the viewer finishes loading later", async () => {
    viewerReadiness.automatic = false;
    const user = userEvent.setup();
    render(<Workspace caseId="case-1" onBack={vi.fn()} registerNavigationGuard={vi.fn()} />);
    await screen.findByRole("heading", { name: "Case003" });
    expect(screen.getByRole("button", { name: "Save" })).toBeDisabled();
    for (const label of [
      "Complete ethmoid coverage",
      "Orientation is correct",
      "Image quality is acceptable",
      "Slice continuity is intact",
    ]) {
      await user.click(screen.getByRole("checkbox", { name: label }));
    }
    await user.click(screen.getByRole("button", { name: "Finalize for review" }));
    expect(await screen.findByText("Finalized for review")).toBeVisible();

    act(() => viewerReadiness.notify());

    expect(screen.getByRole("button", { name: "Save" })).toBeEnabled();
    expect(screen.getByText("Finalized for review")).toBeVisible();
    expect(screen.getByText("Ready for review", { selector: "span" })).toBeVisible();
    expect(api.finalize).toHaveBeenCalledTimes(1);
  });

  it("registers a navigation guard that saves changed review fields", async () => {
    const user = userEvent.setup();
    let guard: (() => Promise<boolean>) | null = null;
    const registerNavigationGuard = vi.fn((next: (() => Promise<boolean>) | null) => {
      guard = next;
    });
    render(
      <Workspace
        caseId="case-1"
        onBack={vi.fn()}
        registerNavigationGuard={registerNavigationGuard}
      />,
    );
    await screen.findByRole("heading", { name: "Case003" });
    await user.click(screen.getByRole("button", { name: "Simulate drawing" }));
    await user.type(screen.getByRole("textbox", { name: "Reviewer notes" }), "Needs review");
    await waitFor(() => expect(guard).not.toBeNull());

    let allowed = false;
    await act(async () => {
      allowed = await guard!();
    });

    expect(allowed).toBe(true);
    expect(api.saveAnnotation).toHaveBeenCalledTimes(1);
    expect(api.updateReview).toHaveBeenCalledWith("case-1", {
      status: "Draft",
      qc_checklist: caseDetail.qc_checklist,
      reviewer_notes: "Needs review",
    });
    expect(api.saveViewerState).toHaveBeenCalled();
  });

  it("makes approved cases read-only while retaining navigation and export", async () => {
    vi.mocked(api.getCase).mockResolvedValue({
      ...caseDetail,
      state: "APPROVED",
      review_status: "Approved",
    });
    render(<Workspace caseId="case-1" onBack={vi.fn()} registerNavigationGuard={vi.fn()} />);
    await screen.findByRole("heading", { name: "Case003" });

    expect(screen.getByRole("button", { name: "Draw" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Erase" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Fill" })).toBeDisabled();
    expect(screen.getByRole("button", { name: /^Save$/ })).toBeDisabled();
    expect(screen.getByRole("textbox", { name: "Reviewer notes" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Save review" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Finalize for review" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "3D" })).toBeEnabled();
    expect(screen.getByRole("link", { name: "Export" })).toBeVisible();
  });
});
