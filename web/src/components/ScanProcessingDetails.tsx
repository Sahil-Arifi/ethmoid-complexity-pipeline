import type { CountRun } from "../api/counting";

const number = (value: unknown) => typeof value === "number" && Number.isFinite(value) ? `${Number(value.toFixed(3))} mm` : "Unknown";
export function ScanProcessingDetails({ run }: { run: CountRun }) {
  if (!run.scan_profile && !run.processing_plan) return null;
  const profile = run.scan_profile ?? {};
  const plan = run.processing_plan ?? {};
  const diagnostics = profile.image_diagnostics as Record<string, unknown> | undefined;
  const quality = profile.quality as Record<string, unknown> | undefined;
  const directModel = run.model_identity?.direct_ethmoid_model;
  const hasDirectModel = typeof directModel === "string" && directModel.length > 0 && !["unavailable_without_promoted_weights", "not_used"].includes(directModel);
  const research = run.research_status === "NOT_ACCURACY_VALIDATED";
  const spacing = Array.isArray(profile.native_spacing_mm) ? profile.native_spacing_mm.map(number).join(" × ") : "Unknown";
  const kind = { native: "Native CT", mpr: "Reformatted CT", unknown: "Unconfirmed" }[String(profile.reconstruction_kind)] ?? "Unconfirmed";
  return <details><summary>How this scan was processed</summary><dl className="scan-processing-details">
    <dt>Slice thickness</dt><dd>{number(profile.slice_thickness_mm)}</dd>
    <dt>Slice interval</dt><dd>{number(profile.slice_interval_mm)}</dd>
    <dt>Original voxel spacing</dt><dd>{spacing}</dd>
    <dt>Reconstruction</dt><dd>{kind}{quality?.oblique_source ? " · Oblique" : ""}</dd>
    <dt>Acquisition policy</dt><dd>{plan.acquisition_policy === "thin_resolution" ? "Thin reconstruction" : plan.acquisition_policy === "experimental_resolution" ? "Experimental resolution; small walls may be unclear" : "Resolution unconfirmed"}</dd>
    <dt>Anatomical localization</dt><dd>{run.assistance?.boundary_id ? "Clinician-assisted boundaries" : hasDirectModel ? "Registered anatomical segmentation model" : research ? "Experimental geometric localization" : "Existing geometric baseline"}</dd>
    <dt>Counting coordinates</dt><dd>Original scan voxels</dd>
  </dl><p className="muted">{hasDirectModel ? "The registered model's output requires clinical review." : "A directly trained ethmoid segmentation model has not been promoted. Clinical counting accuracy remains unmeasured."}</p>
    {research && <p className="muted">Research method: candidate counts require review. More completed counts do not establish accuracy.</p>}
    {typeof diagnostics?.air_sample_mad_hu === "number" && <p className="muted">Sampled air-intensity variation: {diagnostics.air_sample_mad_hu.toFixed(1)} HU. This descriptive measurement does not establish image quality.</p>}
    <small>Method: {run.method_version ?? "Pending"} · Protocol: {run.protocol_version}</small>
  </details>;
}
