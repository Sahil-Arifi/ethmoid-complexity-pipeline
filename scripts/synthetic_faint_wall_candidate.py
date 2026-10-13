"""One frozen synthetic-only candidate; production callers do not import it."""

import numpy as np
from scipy import ndimage as ndi

from ethmoid_pipeline.counting.engine import CountingError, compartment_stage

METHOD = "strong-connected-exclusion-50-v1"
REASON = "experimental_weak_barrier_requires_review"


def infer_exclusion(hu, roi):
    values, region = np.asarray(hu), np.asarray(roi)
    if (values.ndim != 3 or values.shape != region.shape or region.dtype != bool
            or not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values)
            or not np.isfinite(values).all()):
        raise CountingError("INVALID_VOLUME", "Finite real 3D HU values and a matching boolean ROI are required.")
    if not region.any():
        raise CountingError("EMPTY_REGION", "The synthetic region must not be empty.")
    strong = region & (values >= 200)
    reached = ndi.binary_propagation(strong, mask=region & (values >= 50),
                                     structure=ndi.generate_binary_structure(3, 1))
    mask = reached & ~strong
    mask.setflags(write=False)
    return mask


def count_with_faint_wall(hu, roi, affine):
    mask = infer_exclusion(hu, roi)
    cavities, air, cells, warnings = compartment_stage(hu, roi, affine, excluded=mask)
    if mask.any():
        for cell in cells:
            cell["reasons"].append(REASON)
            cell["uncertain"] = True
        warnings.append("Experimental weak-barrier exclusion; this is not a clinical count.")
    return cavities, air, cells, warnings, mask
