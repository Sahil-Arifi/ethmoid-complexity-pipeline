from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid


def write_ct_series(
    root: Path,
    *,
    positions: Iterable[float] | None = None,
    rows: int = 16,
    columns: int = 20,
    study_uid: str | None = None,
    series_uid: str | None = None,
    frame_uid: str | None = None,
    acquisition_numbers: Iterable[int] | None = None,
    irradiation_event_uids: Iterable[str] | None = None,
    image_type: list[str] | None = None,
    photometric: str = "MONOCHROME2",
    burned_in: str | None = "NO",
    instance_numbers: Iterable[int] | None = None,
) -> list[Path]:
    root.mkdir(parents=True, exist_ok=True)
    coordinate_values = list(positions if positions is not None else [index * 0.625 for index in range(6)])
    acquisitions = list(acquisition_numbers or [1] * len(coordinate_values))
    event_values = list(irradiation_event_uids or [generate_uid()] * len(coordinate_values))
    instances = list(instance_numbers or range(1, len(coordinate_values) + 1))
    if not (len(acquisitions) == len(event_values) == len(instances) == len(coordinate_values)):
        raise ValueError("synthetic per-slice values must have equal lengths")
    study = study_uid or generate_uid()
    series = series_uid or generate_uid()
    frame = frame_uid or generate_uid()
    paths: list[Path] = []
    for index, coordinate in enumerate(coordinate_values):
        sop = generate_uid()
        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = CTImageStorage
        meta.MediaStorageSOPInstanceUID = sop
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.ImplementationClassUID = generate_uid()
        path = root / f"image_{len(coordinate_values) - index:04d}.dcm"
        dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\x00" * 128)
        dataset.SOPClassUID = CTImageStorage
        dataset.SOPInstanceUID = sop
        dataset.StudyInstanceUID = study
        dataset.SeriesInstanceUID = series
        dataset.FrameOfReferenceUID = frame
        dataset.Modality = "CT"
        dataset.PatientName = "SYNTHETIC^ONLY"
        dataset.PatientID = "SYNTHETIC"
        dataset.StudyDate = "20000101"
        dataset.SeriesDescription = "SYNTHETIC BONE THIN"
        dataset.ImageType = image_type or ["ORIGINAL", "PRIMARY", "AXIAL"]
        dataset.ConvolutionKernel = "BONEPLUS"
        dataset.Rows = rows
        dataset.Columns = columns
        dataset.PixelSpacing = [0.5, 0.5]
        dataset.SliceThickness = 0.625
        dataset.SpacingBetweenSlices = 0.625
        dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        dataset.ImagePositionPatient = [0, 0, float(coordinate)]
        dataset.InstanceNumber = int(instances[index])
        dataset.AcquisitionNumber = int(acquisitions[index])
        dataset.IrradiationEventUID = event_values[index]
        dataset.SamplesPerPixel = 1
        dataset.PhotometricInterpretation = photometric
        dataset.BitsAllocated = 16
        dataset.BitsStored = 16
        dataset.HighBit = 15
        dataset.PixelRepresentation = 1
        dataset.RescaleSlope = 1
        dataset.RescaleIntercept = -1024
        dataset.RescaleType = "HU"
        if burned_in is not None:
            dataset.BurnedInAnnotation = burned_in
        pixels = (
            np.arange(rows * columns, dtype=np.int16).reshape(rows, columns)
            + np.int16(index * 100)
        )
        dataset.PixelData = pixels.tobytes()
        dataset.save_as(path, enforce_file_format=True)
        paths.append(path)
    return paths


def rewrite(path: Path, **values: object) -> None:
    from pydicom import dcmread

    dataset = dcmread(path)
    for keyword, value in values.items():
        if keyword == "TransferSyntaxUID":
            if "PixelData" in dataset:
                del dataset.PixelData
            dataset.file_meta.TransferSyntaxUID = value
        else:
            setattr(dataset, keyword, value)
    dataset.save_as(path, enforce_file_format=True)
