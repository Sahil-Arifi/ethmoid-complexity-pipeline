import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { HelpTip } from "./HelpTip";

describe("HelpTip", () => {
  it("shows a body-portal tooltip on hover and links it to the trigger", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <HelpTip label="Explain brightness" content="Changes how light or dark the CT appears." />,
    );
    const trigger = screen.getByRole("button", { name: "Explain brightness" });

    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
    await user.hover(trigger);

    const tooltip = screen.getByRole("tooltip");
    expect(tooltip).toHaveTextContent("Changes how light or dark the CT appears.");
    expect(trigger).toHaveAttribute("aria-describedby", tooltip.id);
    expect(container).not.toContainElement(tooltip);

    fireEvent.mouseLeave(trigger);
    fireEvent.mouseEnter(tooltip);
    expect(screen.getByRole("tooltip")).toBeVisible();
    fireEvent.mouseLeave(tooltip);
    await new Promise((resolve) => window.setTimeout(resolve, 150));
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });

  it("supports focus, click pinning, outside dismissal, and Escape", async () => {
    const user = userEvent.setup();
    render(
      <div>
        <HelpTip label="Explain labels" content="Select a region before drawing." />
        <button type="button">Outside</button>
      </div>,
    );
    const trigger = screen.getByRole("button", { name: "Explain labels" });

    fireEvent.focus(trigger);
    expect(screen.getByRole("tooltip")).toBeVisible();
    await user.keyboard("{Escape}");
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();

    fireEvent.blur(trigger);
    fireEvent.click(trigger);
    expect(screen.getByRole("tooltip")).toBeVisible();
    fireEvent.pointerDown(screen.getByRole("button", { name: "Outside" }));
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument();
  });
});
