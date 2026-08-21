import {
  useCallback,
  useEffect,
  useId,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";

type HelpTipProps = {
  content: ReactNode;
  label: string;
  id?: string;
  className?: string;
  children?: ReactNode;
};

type TooltipPosition = {
  left: number;
  top: number;
  placement: "top" | "bottom";
};

const VIEWPORT_GUTTER = 12;
const TRIGGER_GAP = 10;

export function HelpTip({
  content,
  label,
  id,
  className = "",
  children = "?",
}: HelpTipProps) {
  const generatedId = useId();
  const tooltipId = id ?? `help-tip-${generatedId}`;
  const triggerRef = useRef<HTMLButtonElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const hoverCloseTimerRef = useRef<number | null>(null);
  const [hovered, setHovered] = useState(false);
  const [tooltipHovered, setTooltipHovered] = useState(false);
  const [focused, setFocused] = useState(false);
  const [pinned, setPinned] = useState(false);
  const [dismissed, setDismissed] = useState(false);
  const [position, setPosition] = useState<TooltipPosition>({
    left: VIEWPORT_GUTTER,
    top: VIEWPORT_GUTTER,
    placement: "bottom",
  });

  const visible = !dismissed && (hovered || tooltipHovered || focused || pinned);

  const cancelHoverClose = useCallback(() => {
    if (hoverCloseTimerRef.current === null) return;
    window.clearTimeout(hoverCloseTimerRef.current);
    hoverCloseTimerRef.current = null;
  }, []);

  const scheduleHoverClose = useCallback(() => {
    cancelHoverClose();
    hoverCloseTimerRef.current = window.setTimeout(() => {
      setHovered(false);
      setTooltipHovered(false);
      hoverCloseTimerRef.current = null;
    }, 120);
  }, [cancelHoverClose]);

  useEffect(() => cancelHoverClose, [cancelHoverClose]);

  const updatePosition = useCallback(() => {
    const trigger = triggerRef.current;
    const tooltip = tooltipRef.current;
    if (!trigger || !tooltip) return;

    const triggerBox = trigger.getBoundingClientRect();
    const tooltipBox = tooltip.getBoundingClientRect();
    const availableAbove = triggerBox.top - TRIGGER_GAP;
    const placement = availableAbove >= tooltipBox.height ? "top" : "bottom";
    const unclampedLeft = triggerBox.left + triggerBox.width / 2 - tooltipBox.width / 2;
    const maximumLeft = Math.max(
      VIEWPORT_GUTTER,
      window.innerWidth - tooltipBox.width - VIEWPORT_GUTTER,
    );
    const left = Math.min(Math.max(unclampedLeft, VIEWPORT_GUTTER), maximumLeft);
    const top = placement === "top"
      ? triggerBox.top - tooltipBox.height - TRIGGER_GAP
      : triggerBox.bottom + TRIGGER_GAP;

    setPosition({ left, top, placement });
  }, []);

  useLayoutEffect(() => {
    if (visible) updatePosition();
  }, [updatePosition, visible]);

  useEffect(() => {
    if (!visible) return;

    const dismissWithEscape = (event: KeyboardEvent) => {
      if (event.key !== "Escape") return;
      setPinned(false);
      setDismissed(true);
    };
    const dismissOutside = (event: PointerEvent) => {
      const target = event.target;
      if (
        target instanceof Node &&
        (triggerRef.current?.contains(target) || tooltipRef.current?.contains(target))
      ) return;
      setPinned(false);
    };

    window.addEventListener("keydown", dismissWithEscape);
    window.addEventListener("resize", updatePosition);
    window.addEventListener("scroll", updatePosition, true);
    document.addEventListener("pointerdown", dismissOutside);
    return () => {
      window.removeEventListener("keydown", dismissWithEscape);
      window.removeEventListener("resize", updatePosition);
      window.removeEventListener("scroll", updatePosition, true);
      document.removeEventListener("pointerdown", dismissOutside);
    };
  }, [updatePosition, visible]);

  const togglePinned = () => {
    if (pinned) {
      setPinned(false);
      setDismissed(true);
      return;
    }
    setPinned(true);
    setDismissed(false);
  };

  const rootClassName = ["help-tip", className].filter(Boolean).join(" ");

  return (
    <span className={rootClassName}>
      <button
        ref={triggerRef}
        className="help-tip-trigger"
        type="button"
        aria-label={label}
        aria-describedby={visible ? tooltipId : undefined}
        aria-expanded={visible}
        onMouseEnter={() => {
          cancelHoverClose();
          setHovered(true);
          setDismissed(false);
        }}
        onMouseLeave={scheduleHoverClose}
        onFocus={() => {
          setFocused(true);
          setDismissed(false);
        }}
        onBlur={() => setFocused(false)}
        onClick={togglePinned}
      >
        {children}
      </button>
      {visible && typeof document !== "undefined"
        ? createPortal(
            <div
              ref={tooltipRef}
              id={tooltipId}
              className="help-tip-popover"
              role="tooltip"
              data-placement={position.placement}
              onMouseEnter={() => {
                cancelHoverClose();
                setTooltipHovered(true);
              }}
              onMouseLeave={scheduleHoverClose}
              style={{
                left: `${position.left}px`,
                position: "fixed",
                top: `${position.top}px`,
              }}
            >
              {content}
            </div>,
            document.body,
          )
        : null}
    </span>
  );
}
