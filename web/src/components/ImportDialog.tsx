import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { SeriesSummary } from "../api/types";

type Props = {
  onClose: () => void;
  onImported: (caseId: string) => void;
};

type ImportKind = "nifti" | "dicom";

function focusableElements(root: HTMLElement) {
  return Array.from(
    root.querySelectorAll<HTMLElement>(
      'button:not([disabled]), [href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
    ),
  ).filter((element) => !element.hasAttribute("hidden"));
}

export function ImportDialog({ onClose, onImported }: Props) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const dicomInputRef = useRef<HTMLInputElement>(null);
  const previousFocusRef = useRef<HTMLElement | null>(null);
  const [kind, setKind] = useState<ImportKind>("nifti");
  const [caseLabel, setCaseLabel] = useState("");
  const [nifti, setNifti] = useState<File | null>(null);
  const [qcReport, setQcReport] = useState<File | null>(null);
  const [dicomFiles, setDicomFiles] = useState<File[]>([]);
  const [importId, setImportId] = useState<string | null>(null);
  const [series, setSeries] = useState<SeriesSummary[]>([]);
  const [selectedSeries, setSelectedSeries] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const closeDialog = useCallback(() => {
    if (!importId) {
      onClose();
      return;
    }
    setBusy(true);
    setError("");
    void api
      .deleteImport(importId)
      .then(onClose)
      .catch((caught) => {
        setError(caught instanceof Error ? caught.message : "Temporary import cleanup failed.");
      })
      .finally(() => setBusy(false));
  }, [importId, onClose]);

  useEffect(() => {
    if (kind === "dicom") {
      dicomInputRef.current?.setAttribute("webkitdirectory", "");
      dicomInputRef.current?.setAttribute("directory", "");
    }
  }, [kind]);

  useEffect(() => {
    previousFocusRef.current = document.activeElement instanceof HTMLElement
      ? document.activeElement
      : null;
    const frame = window.requestAnimationFrame(() => dialogRef.current?.focus());
    return () => {
      window.cancelAnimationFrame(frame);
      previousFocusRef.current?.focus();
    };
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape" && !busy) {
        event.preventDefault();
        closeDialog();
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
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [busy, closeDialog]);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError("");
    setBusy(true);
    try {
      if (kind === "nifti") {
        if (!nifti) throw new Error("Choose a .nii or .nii.gz file.");
        const result = await api.importNifti(caseLabel.trim(), nifti, qcReport ?? undefined);
        onImported(result.case.case_id);
        return;
      }
      if (dicomFiles.length === 0) throw new Error("Choose a DICOM folder.");
      const result = await api.importDicom(caseLabel.trim(), dicomFiles);
      if (result.series.length === 1 && result.series[0].preflight_status === "PASS") {
        const converted = await api.convertSeries(result.import_id, result.series[0].series_label);
        onImported(converted.case.case_id);
        return;
      }
      setImportId(result.import_id);
      setSeries(result.series);
      setSelectedSeries(
        result.series.find((candidate) => candidate.preflight_status === "PASS")?.series_label ?? "",
      );
      if (result.series.length === 1 && result.series[0].preflight_status === "FAIL") {
        setError("The only DICOM series failed preflight and cannot be converted.");
      }
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Import failed.");
    } finally {
      setBusy(false);
    }
  };

  const convertSelected = async () => {
    if (!importId || !selectedSeries) return;
    setBusy(true);
    setError("");
    try {
      const result = await api.convertSeries(importId, selectedSeries);
      onImported(result.case.case_id);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Conversion failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="modal-backdrop" role="presentation">
      <div
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-labelledby="import-title"
        tabIndex={-1}
        ref={dialogRef}
      >
        <div className="modal-header">
          <div>
            <span className="eyebrow">New case</span>
            <h2 id="import-title">Import scan</h2>
          </div>
          <button className="icon-button" type="button" onClick={closeDialog} disabled={busy}>
            <span aria-hidden="true">×</span>
            <span className="sr-only">Close import dialog</span>
          </button>
        </div>

        {importId && series.length > 0 ? (
          <section aria-labelledby="series-title">
            <h3 id="series-title">Select one DICOM series</h3>
            <p className="muted">Only series that pass metadata preflight can be converted.</p>
            <div className="series-list">
              {series.map((item) => (
                <label className="series-option" key={item.series_label}>
                  <input
                    type="radio"
                    name="series"
                    value={item.series_label}
                    checked={selectedSeries === item.series_label}
                    onChange={() => setSelectedSeries(item.series_label)}
                    disabled={item.preflight_status !== "PASS"}
                  />
                  <span>
                    <strong>{item.series_label} · {item.preflight_status}</strong>
                    <small>
                      {item.image_count} images · {item.rows}×{item.columns} · spacing{" "}
                      {item.spacing_mm.join(" × ")} mm
                      {item.kernel ? ` · ${item.kernel}` : ""}
                    </small>
                    {item.description ? <small>{item.description}</small> : null}
                    {item.warnings.map((warning) => <small className="series-warning" key={warning}>{warning}</small>)}
                    {item.issues.map((issue) => <small className="series-issue" key={issue}>{issue}</small>)}
                  </span>
                </label>
              ))}
            </div>
            {error ? <div className="alert alert-error" role="alert">{error}</div> : null}
            <div className="modal-actions">
              <button type="button" className="button secondary" onClick={closeDialog} disabled={busy}>
                Cancel
              </button>
              <button
                type="button"
                className="button primary"
                onClick={() => void convertSelected()}
                disabled={busy || !selectedSeries}
              >
                {busy ? "Validating and converting…" : "Convert selected series"}
              </button>
            </div>
          </section>
        ) : (
          <form onSubmit={(event) => void submit(event)}>
            <fieldset className="segmented-fieldset">
              <legend>Input type</legend>
              <label className={kind === "nifti" ? "selected" : ""}>
                <input
                  type="radio"
                  name="kind"
                  value="nifti"
                  checked={kind === "nifti"}
                  onChange={() => setKind("nifti")}
                />
                NIfTI
              </label>
              <label className={kind === "dicom" ? "selected" : ""}>
                <input
                  type="radio"
                  name="kind"
                  value="dicom"
                  checked={kind === "dicom"}
                  onChange={() => setKind("dicom")}
                />
                DICOM folder
              </label>
            </fieldset>

            <label className="field">
              <span>Neutral case label</span>
              <input
                value={caseLabel}
                onChange={(event) => setCaseLabel(event.target.value)}
                pattern="Case[0-9]{3}"
                placeholder="Leave blank to generate (for example, Case003)"
                title="Leave blank to generate a label, or use Case followed by exactly 3 digits."
                autoComplete="off"
              />
              <small>Leave blank and the application will generate the next neutral case label.</small>
            </label>

            {kind === "nifti" ? (
              <>
                <label className="field">
                  <span>NIfTI volume</span>
                  <input
                    type="file"
                    accept=".nii,.nii.gz"
                    onChange={(event) => setNifti(event.target.files?.[0] ?? null)}
                  />
                </label>
                <label className="field">
                  <span>Matching QC report <small>(optional)</small></span>
                  <input
                    type="file"
                    accept=".json,application/json"
                    onChange={(event) => setQcReport(event.target.files?.[0] ?? null)}
                  />
                </label>
                <p className="form-hint">Without a matching valid QC report, the scan opens as UNVERIFIED.</p>
              </>
            ) : (
              <label className="field">
                <span>DICOM folder</span>
                <input
                  ref={dicomInputRef}
                  type="file"
                  multiple
                  onChange={(event) => setDicomFiles(Array.from(event.target.files ?? []))}
                />
                <small>{dicomFiles.length > 0 ? `${dicomFiles.length} files selected` : "Choose the folder containing one or more CT series."}</small>
              </label>
            )}

            {error ? <div className="alert alert-error" role="alert">{error}</div> : null}
            <div className="modal-actions">
              <button type="button" className="button secondary" onClick={closeDialog} disabled={busy}>
                Cancel
              </button>
              <button type="submit" className="button primary" disabled={busy}>
                {busy ? "Importing and validating…" : "Import scan"}
              </button>
            </div>
          </form>
        )}
      </div>
    </div>
  );
}
