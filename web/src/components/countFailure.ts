export function countFailureMessage(code: string | null, fallback: string | null): string {
  const reasons: Record<string, string> = {
    SPHENOID_AMBIGUOUS: "The boundary between the ethmoid cells and the sphenoid sinus at the back is ambiguous. The model withheld the count rather than include the wrong spaces.",
    SPHENOID_UNRESOLVED: "The model could not locate the boundary with the sphenoid sinus at the back. No count is available for this scan.",
    ORBIT_WALL_UNRESOLVED: "The bony wall beside the eye could not be located reliably, so the model could not define the cell region.",
    ORBIT_FLOOR_UNRESOLVED: "The bony floor beneath the eye could not be located reliably, so the model could not define the lower boundary.",
    ROOF_UNRESOLVED: "The upper bony boundary could not be located reliably. The count was withheld.",
    INSUFFICIENT_COVERAGE: "Too much visible air in the proposed cell region remained unassigned. The model withheld an incomplete count.",
    TARGET_CLIPPED: "The required anatomy appears to extend beyond the scan or the localized region. A complete count could not be supported.",
    RESOLUTION_INELIGIBLE: "The method used for this run rejected the scan's resolution. This saved run has no count.",
    OBLIQUE_GRID_UNSUPPORTED: "The method used for this run could not analyze the tilted image grid. The scan can still be reviewed.",
    LOCALIZATION_ORIENTATION: "The model could not reliably orient the anatomy in this scan. No count is available.",
    LOCALIZER_NOT_READY: "The local anatomical models are not ready. Complete the inference runtime setup before retrying.",
    LOCALIZER_FAILED: "The local anatomical models did not finish. The run has no count; retry after checking the local runtime.",
    INTERRUPTED: "The local counting job was interrupted before it finished. Run the count again when the application is ready.",
    SOURCE_UNAVAILABLE: "The source volume is unavailable. Restore the local source before retrying.",
    INVALID_OR_UNAVAILABLE_RESULT: "The output did not pass the source, geometry, or result checks. No count is available.",
    EMPTY_REGION: "No supported cell region could be localized in this scan.",
    NO_COMPARTMENTS: "No reliable bony compartments could be separated in the localized region.",
    NO_SUPPORTED_CELLS: "The proposed compartments did not meet the model's checks. No count is available.",
  };
  return (code && reasons[code]) || fallback || "The scan did not provide enough reliable anatomy for a count. Review the scan quality and analysis notes.";
}
