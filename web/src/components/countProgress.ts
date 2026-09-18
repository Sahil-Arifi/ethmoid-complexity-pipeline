import type { CountRun } from "../api/counting";

export function countStage(run: CountRun): string {
  const labels: Record<string, string> = {
    queued: "Waiting for the local worker", preparing: "Preparing verified inputs", profile: "Reading scan characteristics",
    profiling: "Reading scan characteristics", localization: "Locating anatomy", localizing: "Locating anatomy",
    counting: "Separating and counting cells", inference: "Analyzing the scan", verification: "Verifying source alignment and results",
    verifying: "Verifying source alignment and results", publishing: "Saving verified results", completed: "Analysis finished",
    cancelled: "Cancelled", retry_wait: "Waiting to retry a technical interruption",
  };
  return labels[run.stage ?? run.status] ?? "Processing the CT volume";
}
export function countOutcome(run: CountRun): string {
  if (run.status === "cancelled" || run.outcome === "cancelled") return "Count cancelled";
  if (run.status === "queued" || run.status === "running") return countStage(run);
  const labels: Record<string, string> = {
    automatic_candidate_count: "Automatic candidate count", assisted_candidate_count: "Assisted candidate count",
    needs_review: "Anatomical review needed", unsupported_input: "Unsupported input", technical_error: "Technical issue",
  };
  return (run.outcome && labels[run.outcome]) || (run.status === "completed" ? "Candidate count available" : "Count unavailable");
}
