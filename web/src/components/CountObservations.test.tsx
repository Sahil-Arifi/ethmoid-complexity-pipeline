import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { countingApi } from "../api/counting";
import { CountObservations } from "./CountObservations";

vi.mock("../api/counting", () => ({ countingApi: { observations: vi.fn(), observe: vi.fn(), observationsExportUrl: () => "/diagnostic-export" } }));

const context = { caseId: "synthetic-case", runId: "synthetic-run", revision: 0, selectedIds: [300], disabled: false,
  getPoint: () => [-12, 5, 7] as [number, number, number], focusPoint: vi.fn(), onDraftChange: vi.fn() };
const observation = { id: "observation-1", revision: 0, kind: "merged" as const, candidate_ids: [300],
  point_ras_mm: [-12, 5, 7] as [number, number, number], note: "Two physician-marked cells appear within this region.", created_at: "2026-10-05T12:00:00Z" };

describe("diagnostic count observations", () => {
  beforeEach(() => { vi.clearAllMocks(); vi.mocked(countingApi.observations).mockResolvedValue({ observations: [] });
    vi.mocked(countingApi.observe).mockResolvedValue(observation); });

  it("saves an observation linked to the displayed revision without submitting a correction", async () => {
    render(<CountObservations {...context} />);
    await screen.findByText("No observations saved for this revision.");
    await userEvent.selectOptions(screen.getByLabelText("Observation type"), "merged");
    await userEvent.type(screen.getByLabelText("Observation note"), observation.note);
    await userEvent.click(screen.getByRole("button", { name: "Use crosshair location" }));
    await userEvent.click(screen.getByRole("button", { name: "Save observation" }));
    expect(countingApi.observe).toHaveBeenCalledWith(context.caseId, context.runId, {
      revision: 0, kind: "merged", candidate_ids: [300], point_ras_mm: [-12, 5, 7], note: observation.note,
    });
    expect(await screen.findByText(observation.note)).toBeVisible();
    expect(context.onDraftChange).toHaveBeenLastCalledWith(false);
    await userEvent.click(screen.getByRole("button", { name: "Locate observation" }));
    expect(context.focusPoint).toHaveBeenCalledWith([-12, 5, 7]);
  });

  it("requires a point or selected candidate, preserves a failed draft, and keeps approved cases read-only", async () => {
    const view = render(<CountObservations {...context} selectedIds={[]} />);
    await screen.findByText("No observations saved for this revision.");
    await userEvent.type(screen.getByLabelText("Observation note"), "Possible missed cell");
    expect(screen.getByRole("button", { name: "Save observation" })).toBeDisabled();
    await userEvent.click(screen.getByRole("button", { name: "Use crosshair location" }));
    vi.mocked(countingApi.observe).mockRejectedValue(new Error("Source changed. Review the current source."));
    await userEvent.click(screen.getByRole("button", { name: "Save observation" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("Source changed");
    expect(screen.getByLabelText("Observation note")).toHaveValue("Possible missed cell");
    view.rerender(<CountObservations {...context} disabled />);
    expect(screen.getByRole("button", { name: "Save observation" })).toBeDisabled();
    expect(screen.getByLabelText("Observation note")).toBeDisabled();
  });

  it("filters saved observations to the displayed revision and does not substitute coordinates from photos", async () => {
    vi.mocked(countingApi.observations).mockResolvedValue({ observations: [observation, { ...observation, id: "other", revision: 1, note: "Later revision" }] });
    render(<CountObservations {...context} />);
    expect(await screen.findByText(observation.note)).toBeVisible();
    expect(screen.queryByText("Later revision")).not.toBeInTheDocument();
    await waitFor(() => expect(countingApi.observations).toHaveBeenCalledWith(context.caseId, context.runId));
    expect(screen.getByText(/Location not captured/)).toBeVisible();
    expect(screen.getByRole("link", { name: "Export observations JSON" })).toHaveAttribute("href", "/diagnostic-export");
  });
});
