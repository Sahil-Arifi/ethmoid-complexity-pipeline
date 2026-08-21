import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { api } from "../api/client";
import { Dashboard } from "./Dashboard";

vi.mock("../api/client", () => ({
  api: {
    listCases: vi.fn(),
    deleteCase: vi.fn(),
    exportUrl: (caseId: string) => `/api/cases/${caseId}/export`,
  },
}));

describe("Dashboard", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.setItem("ethmoid:first-run-guide", "2");
    vi.mocked(api.listCases).mockResolvedValue({
      cases: [
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

  it("shows validation and annotation status and opens the case", async () => {
    const user = userEvent.setup();
    const openCase = vi.fn();
    render(<Dashboard onOpenCase={openCase} />);

    expect(await screen.findByRole("button", { name: "Case003" })).toBeVisible();
    expect(screen.getByText("VERIFIED")).toBeVisible();
    expect(screen.getByText("Annotation revision 2")).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Continue annotation" }));
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
