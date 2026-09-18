import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import type { CountRun } from "../api/counting";
import { ScanProcessingDetails } from "./ScanProcessingDetails";

const run: CountRun = {
  run_id: "synthetic", case_id: "synthetic-case", status: "completed", latest_revision: 0, revision: 0,
  original_counts: { left: 1, right: 1, total: 2 }, counts: { left: 1, right: 1, total: 2 }, cells: [],
  warnings: [], failure_code: null, message: null, method_version: "research-native-0.5.0",
  protocol_version: "ethmoid-count-research-v5", created_at: "2026-01-01", updated_at: "2026-01-01",
  research_status: "NOT_ACCURACY_VALIDATED",
  model_identity: { localization: "research_geometric_baseline", direct_ethmoid_model: "not_used" },
  scan_profile: { native_spacing_mm: [0.5, 0.5, 2], slice_thickness_mm: 2, slice_interval_mm: 2, reconstruction_kind: "native" },
  processing_plan: { acquisition_policy: "experimental_resolution", counting_grid: "original_native" },
};

describe("processing provenance", () => {
  it("identifies research geometry without claiming a registered segmentation model", () => {
    render(<ScanProcessingDetails run={run} />);
    expect(screen.getByText("Experimental geometric localization")).toBeInTheDocument();
    expect(screen.getByText(/More completed counts do not establish accuracy/)).toBeInTheDocument();
    expect(screen.getByText(/A directly trained ethmoid segmentation model has not been promoted/)).toBeInTheDocument();
    expect(screen.queryByText("Registered anatomical segmentation model")).not.toBeInTheDocument();
    expect(screen.getByText(/research-native-0.5.0/)).toHaveTextContent("ethmoid-count-research-v5");
    expect(screen.getByText("Original scan voxels")).toBeInTheDocument();
  });

  it("retains research context when anatomy could not produce a count", () => {
    render(<ScanProcessingDetails run={{ ...run, status: "failed", counts: null, original_counts: null, outcome: "needs_review" }} />);
    expect(screen.getByText("Experimental geometric localization")).toBeInTheDocument();
    expect(screen.queryByText("Registered anatomical segmentation model")).not.toBeInTheDocument();
  });

  it("continues to describe unchanged v3 assisted boundaries", () => {
    render(<ScanProcessingDetails run={{ ...run, research_status: null, method_version: "scan-adaptive-native-0.3.0",
      protocol_version: "ethmoid-count-prototype-v3", assistance: { boundary_id: "synthetic-boundary", parent_run_id: "v5-parent" },
      model_identity: { localization: "reviewer_native_region", direct_ethmoid_model: "unavailable_without_promoted_weights" } }} />);
    expect(screen.getByText("Clinician-assisted boundaries")).toBeInTheDocument();
    expect(screen.queryByText(/Research method:/)).not.toBeInTheDocument();
    expect(screen.getByText(/scan-adaptive-native-0.3.0/)).toHaveTextContent("ethmoid-count-prototype-v3");
  });

  it("continues to describe the existing automatic geometric baseline", () => {
    render(<ScanProcessingDetails run={{ ...run, research_status: null,
      model_identity: { localization: "v2_geometric_baseline", direct_ethmoid_model: "unavailable_without_promoted_weights" } }} />);
    expect(screen.getByText("Existing geometric baseline")).toBeInTheDocument();
    expect(screen.queryByText("Registered anatomical segmentation model")).not.toBeInTheDocument();
  });
});
