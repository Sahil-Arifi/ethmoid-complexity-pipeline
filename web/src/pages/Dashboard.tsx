import { useCallback, useEffect, useState } from "react";
import { api } from "../api/client";
import type { CaseSummary } from "../api/types";
import { ImportDialog } from "../components/ImportDialog";

type Props = { onOpenCase: (caseId: string) => void };

function friendlyDate(value: string): string {
  const parsed = new Date(value);
  return Number.isNaN(parsed.valueOf()) ? value : parsed.toLocaleString();
}

function canDelete(item: CaseSummary): boolean {
  return item.review_status === "Draft" && item.state !== "APPROVED";
}

export function Dashboard({ onOpenCase }: Props) {
  const [cases, setCases] = useState<CaseSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [showImport, setShowImport] = useState(false);

  const loadCases = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const result = await api.listCases();
      setCases(result.cases);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Cases could not be loaded.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadCases();
  }, [loadCases]);

  const deleteDraft = async (item: CaseSummary) => {
    if (!window.confirm(`Delete draft ${item.case_label}? This cannot be undone.`)) return;
    try {
      await api.deleteCase(item.case_id);
      setCases((current) => current.filter((candidate) => candidate.case_id !== item.case_id));
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Draft could not be deleted.");
    }
  };

  return (
    <main className="dashboard">
      <header className="dashboard-header">
        <div>
          <span className="eyebrow">Local workspace</span>
          <h1>Ethmoid annotation</h1>
          <p>Import, review, and create provisional training annotations.</p>
        </div>
        <button className="button primary" type="button" onClick={() => setShowImport(true)}>
          Import scan
        </button>
      </header>

      <div className="provisional-banner" role="note">
        <strong>Provisional protocol.</strong> Annotations are not approved for model training.
      </div>

      {error ? (
        <div className="alert alert-error">
          <span>{error}</span>
          <button className="text-button" type="button" onClick={() => void loadCases()}>
            Try again
          </button>
        </div>
      ) : null}

      {loading ? <div className="center-state">Loading cases…</div> : null}

      {!loading && cases.length === 0 ? (
        <section className="empty-state">
          <h2>No scans imported</h2>
          <p>Import a verified DICOM series or a NIfTI volume to begin.</p>
          <button className="button primary" type="button" onClick={() => setShowImport(true)}>
            Import first scan
          </button>
        </section>
      ) : null}

      {cases.length > 0 ? (
        <section aria-labelledby="case-list-title">
          <div className="section-heading">
            <h2 id="case-list-title">Cases</h2>
            <span>{cases.length} total</span>
          </div>
          <div className="case-table-wrap">
            <table className="case-table">
              <thead>
                <tr>
                  <th scope="col">Case</th>
                  <th scope="col">Source</th>
                  <th scope="col">Validation</th>
                  <th scope="col">Review</th>
                  <th scope="col">Last saved</th>
                  <th scope="col"><span className="sr-only">Actions</span></th>
                </tr>
              </thead>
              <tbody>
                {cases.map((item) => (
                  <tr key={item.case_id}>
                    <td>
                      <button className="case-link" type="button" onClick={() => onOpenCase(item.case_id)}>
                        {item.case_label}
                      </button>
                      <small>{item.latest_revision ? `Annotation revision ${item.latest_revision}` : "Not annotated"}</small>
                    </td>
                    <td>{item.source_type}</td>
                    <td><span className={`status status-${item.validation_status.toLowerCase()}`}>{item.validation_status}</span></td>
                    <td><span className="status status-neutral">{item.review_status}</span></td>
                    <td>{friendlyDate(item.updated_at)}</td>
                    <td>
                      <div className="row-actions">
                        <button className="button compact" type="button" onClick={() => onOpenCase(item.case_id)}>
                          {item.latest_revision ? "Continue annotation" : "Open"}
                        </button>
                        {item.latest_revision ? (
                          <a className="button compact secondary" href={api.exportUrl(item.case_id)} download>
                            Export
                          </a>
                        ) : null}
                        {canDelete(item) ? (
                          <button className="text-button danger" type="button" onClick={() => void deleteDraft(item)}>
                            Delete draft
                          </button>
                        ) : null}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {showImport ? (
        <ImportDialog
          onClose={() => setShowImport(false)}
          onImported={(caseId) => {
            setShowImport(false);
            onOpenCase(caseId);
          }}
        />
      ) : null}
    </main>
  );
}
