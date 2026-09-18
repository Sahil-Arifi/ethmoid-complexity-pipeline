import { useCallback, useEffect, useId, useRef, useState } from "react";

export const FIRST_RUN_GUIDE_VERSION = "1";
export const FIRST_RUN_GUIDE_STORAGE_KEY = "ethmoid:first-run-guide";

type FirstRunGuideProps = {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  version?: string;
  storageKey?: string;
};

type GuideStep = {
  title: string;
  description: string;
};

const GUIDE_STEPS: readonly GuideStep[] = [
  {
    title: "Import a scan",
    description: "Choose a DICOM folder or ZIP. The app groups every study and series, verifies volumes and starts the recommended series automatically. NIfTI import is also available.",
  },
  {
    title: "Run the automatic count",
    description: "Automatic analysis counts each candidate cell once across the full CT volume. Open another verified series and select Run to analyze it. Progress and any unresolved anatomy remain visible.",
  },
  {
    title: "Review numbered cells",
    description: "Locate and correct cells in linked CT views. If automatic localization needs help, review both anatomical regions and run a separately labeled assisted count.",
  },
  {
    title: "Export the reviewed result",
    description: "Export counts and labeled masks with a record of the original prediction and corrections. Preliminary counts still need clinical validation.",
  },
];

function completedVersion(storageKey: string) {
  try {
    return window.localStorage.getItem(storageKey);
  } catch {
    return null;
  }
}

function storeCompletedVersion(storageKey: string, version: string) {
  try {
    window.localStorage.setItem(storageKey, version);
  } catch {
    // The guide still closes when storage is unavailable or blocked.
  }
}

function focusableElements(root: HTMLElement) {
  return Array.from(
    root.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ),
  ).filter((element) => !element.hasAttribute("hidden"));
}

export function FirstRunGuide({
  open,
  onOpenChange,
  version = FIRST_RUN_GUIDE_VERSION,
  storageKey = FIRST_RUN_GUIDE_STORAGE_KEY,
}: FirstRunGuideProps) {
  const headingId = `${useId()}-guide-title`;
  const descriptionId = `${useId()}-guide-description`;
  const dialogRef = useRef<HTMLDivElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const autoOpenedTokenRef = useRef<string | null>(null);
  const wasOpenRef = useRef(false);
  const [stepIndex, setStepIndex] = useState(0);
  const [internalOpen, setInternalOpen] = useState(
    () => completedVersion(storageKey) !== version,
  );
  const controlled = open !== undefined;
  const visible = controlled ? open : internalOpen;
  const step = GUIDE_STEPS[stepIndex];
  const lastStep = stepIndex === GUIDE_STEPS.length - 1;
  const completionToken = `${storageKey}\u0000${version}`;

  useEffect(() => {
    if (completedVersion(storageKey) === version) return;
    if (autoOpenedTokenRef.current === completionToken) return;

    autoOpenedTokenRef.current = completionToken;
    if (controlled) onOpenChange?.(true);
    else setInternalOpen(true);
  }, [completionToken, controlled, onOpenChange, storageKey, version]);

  useEffect(() => {
    if (visible && !wasOpenRef.current) setStepIndex(0);
    wasOpenRef.current = visible;
  }, [visible]);

  useEffect(() => {
    if (!visible) return;

    previousFocusRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const frame = window.requestAnimationFrame(() => dialogRef.current?.focus());
    return () => {
      window.cancelAnimationFrame(frame);
      previousFocusRef.current?.focus();
    };
  }, [visible]);

  const closeGuide = useCallback(() => {
    storeCompletedVersion(storageKey, version);
    if (!controlled) setInternalOpen(false);
    onOpenChange?.(false);
  }, [controlled, onOpenChange, storageKey, version]);

  useEffect(() => {
    if (!visible) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeGuide();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;

      const focusable = focusableElements(dialogRef.current);
      if (focusable.length === 0) {
        event.preventDefault();
        dialogRef.current.focus();
        return;
      }

      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      const active = document.activeElement;
      if (event.shiftKey && (active === first || active === dialogRef.current)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && active === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => document.removeEventListener("keydown", handleKeyDown);
  }, [closeGuide, visible]);

  if (!visible) return null;

  return (
    <div className="guide-backdrop" role="presentation">
      <div
        ref={dialogRef}
        className="first-run-guide"
        role="dialog"
        aria-modal="true"
        aria-labelledby={headingId}
        aria-describedby={descriptionId}
        tabIndex={-1}
      >
        <header className="first-run-guide-header">
          <span className="eyebrow">Quick start</span>
          <span className="first-run-guide-count" aria-live="polite">
            Step {stepIndex + 1} of {GUIDE_STEPS.length}
          </span>
        </header>

        <div className="first-run-guide-progress" aria-hidden="true">
          {GUIDE_STEPS.map((item, index) => (
            <span
              className={index <= stepIndex ? "complete" : ""}
              key={item.title}
            />
          ))}
        </div>

        <section className="first-run-guide-content" aria-live="polite">
          <h2 id={headingId}>{step.title}</h2>
          <p id={descriptionId}>{step.description}</p>
        </section>

        <footer className="first-run-guide-actions">
          <button className="button tertiary" type="button" onClick={closeGuide}>
            Skip guide
          </button>
          <div>
            <button
              className="button secondary"
              type="button"
              disabled={stepIndex === 0}
              onClick={() => setStepIndex((current) => Math.max(0, current - 1))}
            >
              Back
            </button>
            {lastStep ? (
              <button className="button primary" type="button" onClick={closeGuide}>
                Finish
              </button>
            ) : (
              <button
                className="button primary"
                type="button"
                onClick={() => setStepIndex((current) => Math.min(GUIDE_STEPS.length - 1, current + 1))}
              >
                Next
              </button>
            )}
          </div>
        </footer>
      </div>
    </div>
  );
}
