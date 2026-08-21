import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it } from "vitest";
import { useState } from "react";
import {
  FIRST_RUN_GUIDE_STORAGE_KEY,
  FirstRunGuide,
} from "./FirstRunGuide";

describe("FirstRunGuide", () => {
  beforeEach(() => window.localStorage.clear());

  it("walks through four steps and remembers the completed version", async () => {
    const user = userEvent.setup();
    const { unmount } = render(<FirstRunGuide version="test-v1" />);

    const dialog = screen.getByRole("dialog", { name: "Import a scan" });
    await waitFor(() => expect(dialog).toHaveFocus());
    expect(screen.getByText("Step 1 of 4")).toBeVisible();
    expect(screen.getByRole("button", { name: "Back" })).toBeDisabled();

    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByRole("heading", { name: "Review every view" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByRole("heading", { name: "Mark the ethmoid regions" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Back" }));
    expect(screen.getByRole("heading", { name: "Review every view" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Next" }));
    await user.click(screen.getByRole("button", { name: "Next" }));
    expect(screen.getByRole("heading", { name: "Check, review, and export" })).toBeVisible();
    await user.click(screen.getByRole("button", { name: "Finish" }));

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(window.localStorage.getItem(FIRST_RUN_GUIDE_STORAGE_KEY)).toBe("test-v1");

    unmount();
    render(<FirstRunGuide version="test-v1" />);
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
  });

  it("reopens for a new guide version", () => {
    window.localStorage.setItem(FIRST_RUN_GUIDE_STORAGE_KEY, "old-version");
    render(<FirstRunGuide version="new-version" />);

    expect(screen.getByRole("dialog", { name: "Import a scan" })).toBeVisible();
  });

  it("can be reopened by a parent Help button and restores focus when skipped", async () => {
    const user = userEvent.setup();
    window.localStorage.setItem(FIRST_RUN_GUIDE_STORAGE_KEY, "controlled-v1");

    function Harness() {
      const [open, setOpen] = useState(false);
      return (
        <>
          <button type="button" onClick={() => setOpen(true)}>Help</button>
          <FirstRunGuide
            open={open}
            onOpenChange={setOpen}
            version="controlled-v1"
          />
        </>
      );
    }

    render(<Harness />);
    const help = screen.getByRole("button", { name: "Help" });
    await user.click(help);
    expect(screen.getByRole("dialog", { name: "Import a scan" })).toBeVisible();

    await user.click(screen.getByRole("button", { name: "Skip guide" }));
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(help).toHaveFocus();
  });

  it("traps keyboard focus and treats Escape as skipping", async () => {
    const user = userEvent.setup();
    render(<FirstRunGuide version="keyboard-v1" />);
    const dialog = screen.getByRole("dialog");
    await waitFor(() => expect(dialog).toHaveFocus());

    await user.keyboard("{Shift>}{Tab}{/Shift}");
    expect(screen.getByRole("button", { name: "Next" })).toHaveFocus();
    await user.keyboard("{Tab}");
    expect(screen.getByRole("button", { name: "Skip guide" })).toHaveFocus();
    await user.keyboard("{Escape}");

    expect(screen.queryByRole("dialog")).not.toBeInTheDocument();
    expect(window.localStorage.getItem(FIRST_RUN_GUIDE_STORAGE_KEY)).toBe("keyboard-v1");
  });
});
