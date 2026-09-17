import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { Dashboard } from "./Dashboard";
import { catalogFixture, readySeries, studyFixture, thickSeries } from "../testCatalog";

vi.mock("../api/client", () => ({
  api: {
    listStudies: vi.fn(),
    selectStudySeries: vi.fn(),
    prepareStudySeries: vi.fn(),
    deleteCase: vi.fn(),
    exportUrl: (caseId: string) => `/api/cases/${caseId}/export`,
  },
}));

describe("Dashboard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.setItem("ethmoid:first-run-guide", "2");
    vi.mocked(api.listStudies).mockResolvedValue({
      studies: [], catalog: { ...catalogFixture.catalog, status: "unconfigured" },
      unlinked_cases: [
        {
          case_id: "case-1",
          case_label: "Case003",
          source_type: "DICOM",
          state: "ANNOTATING",
          validation_status: "VERIFIED",
          review_status: "Draft",
          updated_at: "2026-08-20T12:00:00Z",
          latest_revision: 2,
        },
      ],
    });
  });

  it("groups reconstructions once and preserves the selected series ahead of its recommendation", async () => {
    vi.mocked(api.listStudies).mockResolvedValue({ ...catalogFixture, studies: [{ ...studyFixture, selected_series_id: thickSeries.series_id }] });
    const openCase = vi.fn(); render(<Dashboard onOpenCase={openCase} />);
    const select = await screen.findByRole("combobox", { name: "Series for Case003" });
    expect(select).toHaveValue("series-thick");
    expect(within(select).getAllByRole("option")[0]).toHaveValue("series-thin");
    expect(screen.getAllByTestId("study-study-1")).toHaveLength(1);
    expect(screen.getByText("Conversion verified · Ready to attempt")).toBeVisible();
    expect(screen.getByText("Not counted")).toBeVisible();
    await userEvent.click(screen.getByRole("button", { name: "Open series" }));
    expect(openCase).toHaveBeenCalledWith("case-2");
    expect(api.prepareStudySeries).not.toHaveBeenCalled();
  });

  it("persists series selection and keeps unavailable scans selectable with a reason", async () => {
    const unavailable = { ...thickSeries, case_id: null, preparation_status: "unavailable" as const, source_kind: "scout" as const, reason: "A scout is not a 3D volume.", retryable: false };
    const data = { ...catalogFixture, studies: [{ ...studyFixture, series: [readySeries, unavailable] }] };
    vi.mocked(api.listStudies).mockResolvedValueOnce(data).mockResolvedValue({ ...data, studies: [{ ...data.studies[0], selected_series_id: unavailable.series_id }] });
    vi.mocked(api.selectStudySeries).mockResolvedValue({ study: data.studies[0] });
    render(<Dashboard onOpenCase={vi.fn()} />);
    await userEvent.selectOptions(await screen.findByRole("combobox"), unavailable.series_id);
    await waitFor(() => expect(api.selectStudySeries).toHaveBeenCalledWith("study-1", "series-thick"));
    expect(await screen.findByText("A scout is not a 3D volume.")).toBeVisible();
    expect(screen.getByRole("button", { name: "Unavailable" })).toBeDisabled();
    expect(screen.queryByRole("button", { name: "Retry preparation" })).not.toBeInTheDocument();
  });

  it("retries preparation and polls progress without starting a count", async () => {
    const unavailable = { ...readySeries, case_id: null, preparation_status: "unavailable" as const, retryable: true, reason: "Preparation was interrupted." };
    const queued = { ...unavailable, preparation_status: "queued" as const, reason: null };
    const data = { ...catalogFixture, studies: [{ ...studyFixture, series: [unavailable] }] };
    vi.mocked(api.listStudies).mockResolvedValueOnce(data).mockResolvedValueOnce({ ...data, catalog: { ...data.catalog, status: "preparing", queued: 1 }, studies: [{ ...studyFixture, series: [queued] }] }).mockResolvedValue(catalogFixture);
    vi.mocked(api.prepareStudySeries).mockResolvedValue({ series: queued });
    render(<Dashboard onOpenCase={vi.fn()} />);
    await userEvent.click(await screen.findByRole("button", { name: "Retry preparation" }));
    expect(api.prepareStudySeries).toHaveBeenCalledWith("study-1", "series-thin");
    expect(await screen.findByText("Preparing CT volumes in the background")).toBeVisible();
    expect(await screen.findByRole("button", { name: "Open series" }, { timeout: 3500 })).toBeEnabled();
    expect(screen.getByText("Not counted")).toBeVisible();
  });

  it("shows validation and annotation status and opens the case", async () => {
    const user = userEvent.setup();
    const openCase = vi.fn();
    render(<Dashboard onOpenCase={openCase} />);

    expect(await screen.findByRole("button", { name: "Case003" })).toBeVisible();
    expect(screen.getByText("VERIFIED")).toBeVisible();
    expect(screen.getByText("Annotation revision 2")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Open case" }));
    expect(openCase).toHaveBeenCalledWith("case-1");
  });

  it("deletes only after confirmation", async () => {
    const user = userEvent.setup();
    vi.spyOn(window, "confirm").mockReturnValue(true);
    render(<Dashboard onOpenCase={vi.fn()} />);
    await screen.findByText("Case003");
    await user.click(screen.getByRole("button", { name: "Delete draft" }));

    await waitFor(() => expect(api.deleteCase).toHaveBeenCalledWith("case-1"));
    expect(screen.queryByText("Case003")).not.toBeInTheDocument();
  });
});
