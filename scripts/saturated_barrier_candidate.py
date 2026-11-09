"""Falsifiable synthetic hypothesis; the float32 aperture counterexample rejects promotion.

This module is deliberately isolated from production. Saturated stored samples do
not prove that the exact, prequantization observation equals a material endpoint.
"""
import numpy as np
from scipy import ndimage as ndi

from canonical_ridge_candidate import COMMUNICATION_REASON, count_with_canonical_ridges

MODEL = 'CONDITIONAL_EXACT_TWO_MATERIAL_MODEL'
WARNING = ('Conditional hypothesis assumes known air/bone endpoints, noiseless unquantized '
           'box-average observations and full enclosure; these assumptions are not inferred '
           'from HU. Float32 rounding admits an opposite-count tiny-aperture counterexample; '
           'this inference rule must not be promoted.')
FACE = ndi.generate_binary_structure(3, 1)
FULL = ndi.generate_binary_structure(3, 3)


def _eligible(hu, roi, output):
    diagnostics = output['diagnostics']
    if (diagnostics['status'] != 'withheld'
            or diagnostics['withheld_reasons'] != [COMMUNICATION_REASON]
            or roi.dtype != bool or not roi.all()
            or not np.all((hu == -900) | (hu == 800))
            or len(output['cells']) != 2):
        return False
    free = roi & (hu == -900)
    six, six_count = ndi.label(free, FACE)
    _, full_count = ndi.label(free, FULL)
    if six_count != 2 or full_count != 2:
        return False
    boundary = roi & ~ndi.binary_erosion(roi, structure=FACE, border_value=0)
    if np.any(free & boundary):
        return False
    cavities = output['cavities']
    registry = {cell['id'] for cell in output['cells']}
    if len(registry) != 2 or set(np.unique(cavities)) - {0} != registry:
        return False
    matched = set()
    for identifier in (1, 2):
        component = six == identifier
        labels = set(np.unique(cavities[component]))
        if len(labels) != 1 or not labels.issubset(registry):
            return False
        cell_id = labels.pop()
        if cell_id in matched or not np.array_equal(cavities == cell_id, component):
            return False
        matched.add(cell_id)
    return matched == registry


def count_with_saturated_barrier(hu, roi, affine):
    """Preserve canonical results except for the explicitly conditional inference."""
    output = count_with_canonical_ridges(hu, roi, affine)
    recovered = _eligible(np.asarray(hu), np.asarray(roi), output)
    diagnostics = output['diagnostics']
    diagnostics['baseline_method'] = diagnostics['method']
    diagnostics.update(method='saturated-barrier-conditional-v1', conditional_model=MODEL,
                       conditional_recovery=recovered, clinical_accuracy='NOT_MEASURED')
    if recovered:
        cells = output['cells']
        diagnostics.update(status='accepted', experimental_count_available=True,
                           withheld_reasons=[], counts={
                               'left': sum(cell['side'] == 'left' for cell in cells),
                               'right': sum(cell['side'] == 'right' for cell in cells),
                               'total': len(cells)})
    output['warnings'] = [*output['warnings'], WARNING]
    return output
