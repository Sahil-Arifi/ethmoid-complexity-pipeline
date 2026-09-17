"""In-memory, metadata-free presentation reports for a validated count revision.

CT views are sampled in physical RAS planes, including for oblique/native-flipped
volumes. This is display interpolation only: labels and counts are not changed.
The caller must keep returned medical-image bytes in its protected local export.
"""
from __future__ import annotations

from io import BytesIO
from itertools import product
import math
from pathlib import Path
import re
from typing import Any

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


class CountReportError(ValueError):
    """An invalid report input, with no protected source details in the message."""


EXPERIMENTAL_NOTE = "Experimental prototype. Clinical accuracy is unmeasured; counts require expert review."
PALETTE = ((48, 165, 219), (239, 177, 64), (103, 190, 130), (213, 125, 185), (161, 148, 220), (229, 119, 96))
INK = (24, 38, 49)
MUTED = (81, 99, 111)
PAPER = (248, 250, 251)
PLANE_AXES = {"Axial": (0, 1, 2), "Coronal": (0, 2, 1), "Sagittal": (1, 2, 0)}
PLANE_LABELS = {"Axial": ("R", "L", "A", "P"), "Coronal": ("R", "L", "S", "I"), "Sagittal": ("A", "P", "S", "I")}


def _counts(value: Any) -> dict[str, int]:
    if not isinstance(value, dict) or any(type(value.get(key)) is not int or value[key] < 0 for key in ("left", "right", "total")):
        raise CountReportError("A completed, internally consistent count is required for a report.")
    selected = {key: value[key] for key in ("left", "right", "total")}
    if selected["total"] != selected["left"] + selected["right"]:
        raise CountReportError("The supplied count totals are inconsistent.")
    return selected


def _validate_result(result: dict, case_label: str) -> tuple[list[dict], dict, dict | None, int]:
    if not re.fullmatch(r"Case[0-9]{3}", case_label):
        raise CountReportError("Reports require a neutral Case### label.")
    if not isinstance(result, dict) or result.get("status", "completed") != "completed":
        raise CountReportError("A failed or unfinished run cannot produce a count report.")
    revision = result.get("revision", 0)
    if type(revision) is not int or revision < 0:
        raise CountReportError("The count revision is invalid.")
    current = _counts(result.get("corrected_counts", result.get("counts")))
    if result.get("counts") is not None and _counts(result["counts"]) != current:
        raise CountReportError("Current and corrected counts disagree.")
    original = _counts(result["original_counts"]) if result.get("original_counts") is not None else None
    records = result.get("cells")
    if not isinstance(records, list) or len(records) > 65535:
        raise CountReportError("The cell registry is invalid.")
    cells, identities = [], set()
    for record in records:
        identity = record.get("id") if isinstance(record, dict) else None
        if type(identity) is not int or not 1 <= identity <= 65535 or identity in identities:
            raise CountReportError("Cell identifiers must be unique positive instance labels.")
        if record.get("side") not in {"left", "right"} or record.get("origin") not in {"automatic", "manual"}:
            raise CountReportError("Cell side or origin is invalid.")
        if type(record.get("included")) is not bool or type(record.get("uncertain")) is not bool:
            raise CountReportError("Cell inclusion and uncertainty must be explicit.")
        if record.get("retired") is True and record["included"]:
            raise CountReportError("A retired cell cannot be included in the current count.")
        try:
            point = np.asarray(record["centroid_ras_mm"], dtype=float)
        except (KeyError, TypeError, ValueError):
            raise CountReportError("Cell coordinates are invalid.") from None
        if point.shape != (3,) or not np.isfinite(point).all():
            raise CountReportError("Cell coordinates are invalid.")
        # Copy only an explicit allowlist: notes, paths, identifiers, timestamps,
        # and arbitrary report/header strings never reach the exported document.
        cells.append({"id": identity, "side": record["side"], "origin": record["origin"], "included": record["included"], "uncertain": record["uncertain"], "retired": record.get("retired") is True, "point": point})
        identities.add(identity)
    observed = {side: sum(cell["included"] and cell["side"] == side for cell in cells) for side in ("left", "right")}
    observed["total"] = observed["left"] + observed["right"]
    if observed != current:
        raise CountReportError("The supplied count does not match the included cell registry.")
    return sorted(cells, key=lambda cell: cell["id"]), current, original, revision


def _load_volumes(source_path: Path, cavity_path: Path, air_path: Path, cells: list[dict]):
    source = nib.load(source_path, mmap=True)
    cavity_image = nib.load(cavity_path, mmap=True)
    air_image = nib.load(air_path, mmap=True)
    if len(source.shape) != 3 or any(size < 2 for size in source.shape) or not np.isfinite(source.affine).all():
        raise CountReportError("A finite three-dimensional CT grid is required.")
    if abs(float(np.linalg.det(source.affine[:3, :3]))) < 1e-9:
        raise CountReportError("The CT physical geometry is singular.")
    for mask in (cavity_image, air_image):
        if mask.shape != source.shape or not np.allclose(mask.affine, source.affine, rtol=0, atol=1e-5):
            raise CountReportError("Report masks must match the native CT shape and affine.")
    hu = source.get_fdata(dtype=np.float32, caching="unchanged")
    cavity = np.asanyarray(cavity_image.dataobj)
    air = np.asanyarray(air_image.dataobj)
    if not np.isfinite(hu).all():
        raise CountReportError("The CT contains nonfinite intensities.")
    for mask in (cavity, air):
        if not np.isfinite(mask).all() or np.any(mask < 0) or np.any(mask > 65535) or not np.array_equal(mask, np.rint(mask)):
            raise CountReportError("Report masks must contain finite nonnegative integer instance labels.")
    cavity, air = cavity.astype(np.uint16, copy=False), air.astype(np.uint16, copy=False)
    known = {0, *(cell["id"] for cell in cells)}
    if not set(np.unique(cavity)).issubset(known) or np.any((air != 0) & (air != cavity)):
        raise CountReportError("The cavity and air labels do not agree with the cell registry.")
    inverse = np.linalg.inv(source.affine)
    for cell in cells:
        index = nib.affines.apply_affine(inverse, cell["point"])
        if np.any(index < -0.5) or np.any(index > np.asarray(source.shape) - 0.5):
            raise CountReportError("A cell location lies outside the source volume.")
    return hu, cavity, air, np.asarray(source.affine, dtype=float)


def _physical_scene(shape: tuple, affine: np.ndarray, cavity: np.ndarray, cells: list[dict]):
    corners = np.asarray(list(product(*[(-0.5, size - 0.5) for size in shape])))
    physical = nib.affines.apply_affine(affine, corners)
    volume_bounds = np.column_stack((physical.min(axis=0), physical.max(axis=0)))
    included = [cell for cell in cells if cell["included"]]
    if not included:
        return volume_bounds.mean(axis=1), volume_bounds, None
    points = np.asarray([cell["point"] for cell in included])
    median = np.median(points, axis=0)
    reference = min(included, key=lambda cell: (float(np.linalg.norm(cell["point"] - median)), cell["id"]))
    present = np.isin(cavity, [cell["id"] for cell in included])
    index = np.where(present)
    bounds_points = [*points]
    if len(index[0]):
        box = list(product(*[(int(axis.min()) - 0.5, int(axis.max()) + 0.5) for axis in index]))
        bounds_points.extend(nib.affines.apply_affine(affine, box))
    points = np.asarray(bounds_points)
    lower = np.maximum(volume_bounds[:, 0], points.min(axis=0) - 12.0)
    upper = np.minimum(volume_bounds[:, 1], points.max(axis=0) + 12.0)
    return reference["point"], np.column_stack((lower, upper)), reference["id"]


def _plane_coordinates(plane: str, center: np.ndarray, bounds: np.ndarray, max_width: int, max_height: int):
    horizontal, vertical, fixed = PLANE_AXES[plane]
    spans = bounds[:, 1] - bounds[:, 0]
    mm_per_pixel = max(float(spans[horizontal]) / max_width, float(spans[vertical]) / max_height)
    width = max(2, int(math.ceil(spans[horizontal] / mm_per_pixel)))
    height = max(2, int(math.ceil(spans[vertical] / mm_per_pixel)))
    # All horizontal and vertical axes run from high to low RAS coordinates:
    # radiological R->L, A->P, and S->I conventions are explicit on every view.
    u = bounds[horizontal, 1] - (np.arange(width) + 0.5) * mm_per_pixel
    v = bounds[vertical, 1] - (np.arange(height) + 0.5) * mm_per_pixel
    grid = np.broadcast_to(np.asarray(center, dtype=float), (height, width, 3)).copy()
    grid[:, :, horizontal] = u[None, :]
    grid[:, :, vertical] = v[:, None]
    return grid, mm_per_pixel


def _sample_trilinear(data: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    valid = np.all((coordinates >= -0.5) & (coordinates <= np.asarray(data.shape) - 0.5), axis=-1)
    points = np.clip(coordinates, 0, np.asarray(data.shape) - 1)
    lower = np.floor(points).astype(np.int64)
    upper = np.minimum(lower + 1, np.asarray(data.shape) - 1)
    fraction = points - lower
    output = np.zeros(coordinates.shape[:-1], dtype=np.float32)
    for bits in product((0, 1), repeat=3):
        indices = tuple(upper[..., axis] if bits[axis] else lower[..., axis] for axis in range(3))
        weight = np.prod(np.stack([fraction[..., axis] if bits[axis] else 1 - fraction[..., axis] for axis in range(3)]), axis=0)
        output += np.asarray(data[indices] * weight, dtype=np.float32)
    return np.where(valid, output, -1024.0)


def _sample_labels(data: np.ndarray, coordinates: np.ndarray) -> np.ndarray:
    indices = np.rint(coordinates).astype(np.int64)
    valid = np.all((indices >= 0) & (indices < np.asarray(data.shape)), axis=-1)
    indices = np.clip(indices, 0, np.asarray(data.shape) - 1)
    values = data[tuple(indices[..., axis] for axis in range(3))]
    return np.where(valid, values, 0).astype(np.uint16)


def _font(size: int):
    return ImageFont.load_default(size=size)


def _label(draw: ImageDraw.ImageDraw, text: str, anchor: tuple[int, int], color: tuple, width: int, height: int, occupied: list):
    font = _font(25)
    text_box = draw.textbbox((0, 0), text, font=font)
    box_width, box_height = text_box[2] + 12, 34
    candidates = [(8, -18), (-box_width - 8, -18), (8, 18), (-box_width - 8, 18), (25, -55), (-box_width - 25, -55), (25, 48)]
    best = None
    for dx, dy in candidates:
        x = max(25, min(width - box_width - 25, anchor[0] + dx))
        y = max(30, min(height - box_height - 62, anchor[1] + dy))
        box = (x, y, x + box_width, y + box_height)
        collisions = sum(not (box[2] < old[0] or box[0] > old[2] or box[3] < old[1] or box[1] > old[3]) for old in occupied)
        if best is None or collisions < best[0]:
            best = (collisions, box)
        if collisions == 0:
            break
    box = best[1]
    draw.line((anchor, ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)), fill=color, width=2)
    draw.rounded_rectangle(box, radius=5, fill=(17, 25, 30), outline=color, width=2)
    draw.text((box[0] + 6, box[1] + 2), text, font=font, fill=(255, 255, 255))
    occupied.append(box)


def _view(plane: str, hu, cavity, air, affine, cells, center, bounds) -> Image.Image:
    panel = Image.new("RGB", (700, 820), PAPER)
    draw = ImageDraw.Draw(panel)
    draw.text((24, 18), plane.upper(), font=_font(32), fill=INK)
    grid, mm_per_pixel = _plane_coordinates(plane, center, bounds, 650, 650)
    coordinates = nib.affines.apply_affine(np.linalg.inv(affine), grid)
    grey = np.uint8(np.clip((_sample_trilinear(hu, coordinates) + 1200.0) / 3000.0, 0, 1) * 255)
    ids, air_ids = _sample_labels(cavity, coordinates), _sample_labels(air, coordinates)
    rgb = np.repeat(grey[:, :, None], 3, axis=2)
    included = [cell for cell in cells if cell["included"]]
    for cell in included:
        identity = cell["id"]
        color = np.asarray(PALETTE[(identity - 1) % len(PALETTE)])
        observed = air_ids == identity
        rgb[observed] = np.uint8(rgb[observed] * 0.58 + color * 0.42)
        selected = ids == identity
        padded = np.pad(selected, 1)
        edge = selected & ~(padded[:-2, 1:-1] & padded[2:, 1:-1] & padded[1:-1, :-2] & padded[1:-1, 2:])
        rgb[edge] = color
    view = Image.fromarray(rgb)
    pixels = ImageDraw.Draw(view)
    occupied = []
    horizontal, vertical, fixed = PLANE_AXES[plane]
    for cell in included:
        locations = np.argwhere(ids == cell["id"])
        if len(locations):
            midpoint = locations.mean(axis=0)
            anchor_y, anchor_x = locations[np.argmin(np.sum((locations - midpoint) ** 2, axis=1))]
        elif cell["origin"] == "manual" and abs(float(cell["point"][fixed] - center[fixed])) <= max(np.linalg.norm(affine[:3, :3], axis=0)) / 2:
            anchor_x = int((bounds[horizontal, 1] - cell["point"][horizontal]) / mm_per_pixel)
            anchor_y = int((bounds[vertical, 1] - cell["point"][vertical]) / mm_per_pixel)
            if not (0 <= anchor_x < view.width and 0 <= anchor_y < view.height):
                continue
            pixels.ellipse((anchor_x - 5, anchor_y - 5, anchor_x + 5, anchor_y + 5), outline=PALETTE[(cell["id"] - 1) % len(PALETTE)], width=2)
        else:
            continue
        _label(pixels, str(cell["id"]) + ("?" if cell["uncertain"] else ""), (int(anchor_x), int(anchor_y)), PALETTE[(cell["id"] - 1) % len(PALETTE)], view.width, view.height, occupied)
    left, right, top, bottom = PLANE_LABELS[plane]
    orientation_font = _font(25)
    for text, position in ((left, (7, view.height // 2)), (right, (view.width - 26, view.height // 2)), (top, (view.width // 2, 5)), (bottom, (view.width // 2, view.height - 29))):
        pixels.text(position, text, font=orientation_font, fill="white", stroke_width=1, stroke_fill="black")
    bar_mm = next((size for size in (20, 10, 5, 1) if size / mm_per_pixel < view.width / 3), 1)
    bar_pixels = max(1, int(round(bar_mm / mm_per_pixel)))
    pixels.line((24, view.height - 23, 24 + bar_pixels, view.height - 23), fill="white", width=4)
    pixels.text((24, view.height - 52), f"{bar_mm} mm", font=_font(22), fill="white", stroke_width=1, stroke_fill="black")
    x, y = (700 - view.width) // 2, 80 + (650 - view.height) // 2
    panel.paste(view, (x, y))
    draw.text((24, 760), f"{left} / {right}     {top} / {bottom}", font=_font(23), fill=MUTED)
    return panel


def _png_document(views: Image.Image, case: str, counts: dict, original: dict | None, revision: int, reference: int | None) -> bytes:
    document = Image.new("RGB", (2100, 1240), PAPER)
    draw = ImageDraw.Draw(document)
    draw.text((35, 25), "3D ETHMOID CELL COUNT", font=_font(52), fill=INK)
    draw.text((35, 92), f"{case}  |  Revision {revision}  |  " + ("Original automatic result" if revision == 0 else "Reviewed result"), font=_font(28), fill=MUTED)
    for index, key in enumerate(("left", "right", "total")):
        x = 35 + index * 690
        draw.rounded_rectangle((x, 145, x + 655, 280), radius=10, fill=(230, 237, 241))
        draw.text((x + 22, 164), key.upper(), font=_font(27), fill=MUTED)
        draw.text((x + 22, 204), str(counts[key]), font=_font(52), fill=INK)
    document.paste(views, (0, 300))
    draw.text((35, 1125), EXPERIMENTAL_NOTE, font=_font(27), fill=INK)
    if original is not None:
        detail = f"Original: left {original['left']}, right {original['right']}, total {original['total']}. Current revision: left {counts['left']}, right {counts['right']}, total {counts['total']}."
    else:
        detail = "Colored outline: cavity. Filled color: observed air. ? marks an uncertainty flag. Only included IDs are shown."
    draw.text((35, 1164), detail, font=_font(24), fill=MUTED)
    reference_text = f"Representative orthogonal planes through cell {reference}." if reference is not None else "No included cells to overlay."
    draw.text((35, 1200), reference_text + " Each view shows one slice; the PDF registry lists every cell.", font=_font(23), fill=MUTED)
    output = BytesIO()
    document.save(output, format="PNG", optimize=True)
    return output.getvalue()


def _pdf_document(views: Image.Image, case: str, cells: list[dict], counts: dict, original: dict | None, revision: int, reference: int | None) -> bytes:
    output = BytesIO()
    pdf = canvas.Canvas(output, pagesize=landscape(A4), invariant=1, pageCompression=0)
    pdf.setTitle("3D ethmoid cell count")
    pdf.setAuthor("Ethmoid prototype")
    pdf.setCreator("Ethmoid prototype")
    pdf.setSubject("Experimental count presentation; clinical accuracy unmeasured")
    width, height = landscape(A4)
    margin = 36

    def header(title: str, page: int):
        pdf.setFillColor(colors.HexColor("#182631"))
        pdf.setFont("Helvetica-Bold", 23)
        pdf.drawString(margin, height - 43, title)
        pdf.setFont("Helvetica", 10)
        pdf.setFillColor(colors.HexColor("#51636F"))
        pdf.drawString(margin, height - 62, f"{case}  |  Revision {revision}  |  " + ("Original automatic result" if revision == 0 else "Reviewed result"))
        pdf.setFont("Helvetica", 8)
        pdf.drawRightString(width - margin, 23, f"Page {page}")

    header("3D ethmoid cell count", 1)
    box_width = (width - margin * 2 - 24) / 3
    for index, key in enumerate(("left", "right", "total")):
        x = margin + index * (box_width + 12)
        pdf.setFillColor(colors.HexColor("#E6EDF1"))
        pdf.roundRect(x, height - 139, box_width, 61, 6, fill=1, stroke=0)
        pdf.setFillColor(colors.HexColor("#51636F"))
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(x + 13, height - 96, key.upper())
        pdf.setFillColor(colors.HexColor("#182631"))
        pdf.setFont("Helvetica-Bold", 26)
        pdf.drawString(x + 13, height - 126, str(counts[key]))
    image_width = width - margin * 2
    image_height = image_width * views.height / views.width
    pdf.drawImage(ImageReader(views), margin, height - 150 - image_height, image_width, image_height)
    pdf.setFillColor(colors.HexColor("#182631"))
    pdf.setFont("Helvetica-Bold", 9)
    pdf.drawString(margin, 115, EXPERIMENTAL_NOTE)
    pdf.setFont("Helvetica", 9)
    pdf.drawString(margin, 97, "Colored outlines show cavities; filled color shows observed air. ? marks an uncertainty flag.")
    pdf.drawString(margin, 81, (f"Planes are centered on cell {reference}. " if reference is not None else "No included cells are overlaid. ") + "Each image is one slice; all registered cells appear in the following table.")
    pdf.drawString(margin, 65, "Radiological orientation is labeled on each image. Physical aspect ratios are preserved; bone display WL 300 / WW 3000 HU.")
    if original is not None:
        pdf.setFont("Helvetica-Bold", 9)
        pdf.drawString(margin, 46, f"Original: L {original['left']} / R {original['right']} / Total {original['total']}     Current revision: L {counts['left']} / R {counts['right']} / Total {counts['total']}")
    pdf.showPage()
    if cells:
        rows_per_page = 19
        columns = [margin, 122, 225, 390, 550]
        headings = ["Stable cell ID", "Side", "Inclusion", "Uncertainty", "Origin"]
        for start in range(0, len(cells), rows_per_page):
            header("Cell registry", start // rows_per_page + 2)
            y = height - 94
            pdf.setFillColor(colors.HexColor("#E6EDF1"))
            pdf.rect(margin, y - 9, width - 2 * margin, 25, fill=1, stroke=0)
            pdf.setFillColor(colors.HexColor("#182631"))
            pdf.setFont("Helvetica-Bold", 10)
            for x, text in zip(columns, headings):
                pdf.drawString(x + 8, y, text)
            for row_number, cell in enumerate(cells[start:start + rows_per_page]):
                row_y = y - 28 - row_number * 22
                if row_number % 2 == 0:
                    pdf.setFillColor(colors.HexColor("#F4F7F8"))
                    pdf.rect(margin, row_y - 6, width - 2 * margin, 22, fill=1, stroke=0)
                values = [str(cell["id"]), cell["side"].capitalize(), "Retired" if cell["retired"] else ("Included" if cell["included"] else "Excluded"), "Flagged" if cell["uncertain"] else "Not flagged", cell["origin"].capitalize()]
                pdf.setFillColor(colors.HexColor("#182631"))
                pdf.setFont("Helvetica", 10)
                for x, text in zip(columns, values):
                    pdf.drawString(x + 8, row_y, text)
            pdf.setFont("Helvetica", 8)
            pdf.drawString(margin, 39, "Not flagged does not mean validated. Counts include only entries marked Included; manual changes are retained as separate revisions.")
            pdf.showPage()
    pdf.save()
    return output.getvalue()


def build_count_reports(source_path: Path, cavity_path: Path, air_path: Path, result: dict, case_label: str) -> dict[str, bytes]:
    """Return summary.pdf and representative.png without writing files.

    ``result`` is a completed revision containing counts/cells/revision. Optional
    original_counts and corrected_counts provide the comparison. Arbitrary free
    text and NIfTI header fields are intentionally omitted from both documents.
    """
    try:
        cells, counts, original, revision = _validate_result(result, case_label)
        hu, cavity, air, affine = _load_volumes(source_path, cavity_path, air_path, cells)
        center, bounds, reference = _physical_scene(hu.shape, affine, cavity, cells)
        views = Image.new("RGB", (2100, 820), PAPER)
        for index, plane in enumerate(PLANE_AXES):
            views.paste(_view(plane, hu, cavity, air, affine, cells, center, bounds), (index * 700, 0))
        return {"summary.pdf": _pdf_document(views, case_label, cells, counts, original, revision, reference), "representative.png": _png_document(views, case_label, counts, original, revision, reference)}
    except CountReportError:
        raise
    except (OSError, ValueError, TypeError, KeyError, np.linalg.LinAlgError):
        raise CountReportError("The local CT or count report inputs could not be read safely.") from None
