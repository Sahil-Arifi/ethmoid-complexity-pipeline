"""Read native CT frames without changing originals or guessing functional groups."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from uuid import NAMESPACE_OID, uuid5

import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import (CTImageStorage, EnhancedCTImageStorage, ExplicitVRLittleEndian,
    ImplicitVRLittleEndian, ExplicitVRBigEndian, DeflatedExplicitVRLittleEndian,
    RLELossless, JPEGLossless, JPEGLosslessSV1, JPEGLSLossless, JPEG2000Lossless)

UNCOMPRESSED = {ImplicitVRLittleEndian, ExplicitVRLittleEndian, ExplicitVRBigEndian, DeflatedExplicitVRLittleEndian}
LOSSLESS = UNCOMPRESSED | {RLELossless, JPEGLossless, JPEGLosslessSV1, JPEGLSLossless, JPEG2000Lossless}


def validate_encoding(ds: Dataset) -> None:
    if str(getattr(ds, "LossyImageCompression", "00")) == "01":
        raise ValueError("lossy image encoding is not supported for verified counting")
    if str(ds.file_meta.TransferSyntaxUID) not in LOSSLESS:
        raise ValueError("unsupported or lossy transfer syntax")


def _group(shared: Dataset, frame: Dataset, key: str, *, required=True) -> Dataset | None:
    # A macro cannot be specified in both scopes. Do not silently prefer one.
    a, b = getattr(shared, key, None), getattr(frame, key, None)
    if a is not None and b is not None:
        raise ValueError("Enhanced CT functional group occurs in both shared and per-frame scopes")
    sequence = b if b is not None else a
    if sequence is None:
        if required:
            raise ValueError("Enhanced CT required functional group is missing")
        return None
    if len(sequence) != 1:
        raise ValueError("Enhanced CT functional group must contain one item")
    return sequence[0]


def frame_headers(path: Path, *, enhanced: bool = True, only_index: int | None = None):
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    if str(ds.SOPClassUID) != str(EnhancedCTImageStorage) or not enhanced:
        yield None, ds
        return
    count = int(getattr(ds, "NumberOfFrames", 0))
    shared = getattr(ds, "SharedFunctionalGroupsSequence", [])
    frames = getattr(ds, "PerFrameFunctionalGroupsSequence", [])
    if count < 2 or count > 50000 or len(shared) != 1 or len(frames) != count:
        raise ValueError("Enhanced CT frame counts or functional groups are invalid")
    validate_encoding(ds)
    for index, frame in enumerate(frames):
        if only_index is not None and index != only_index:
            continue
        view = Dataset()
        view.file_meta = deepcopy(ds.file_meta)
        for element in ds:
            if element.keyword not in {"SharedFunctionalGroupsSequence", "PerFrameFunctionalGroupsSequence"}:
                view.add(deepcopy(element))
        measures = _group(shared[0], frame, "PixelMeasuresSequence")
        orientation = _group(shared[0], frame, "PlaneOrientationSequence")
        position = _group(shared[0], frame, "PlanePositionSequence")
        calibration = _group(shared[0], frame, "PixelValueTransformationSequence")
        for source, keywords in ((measures, ("PixelSpacing", "SliceThickness", "SpacingBetweenSlices")),
                                 (orientation, ("ImageOrientationPatient",)),
                                 (position, ("ImagePositionPatient",)),
                                 (calibration, ("RescaleSlope", "RescaleIntercept", "RescaleType"))):
            for key in keywords:
                if hasattr(view, key):
                    delattr(view, key)
                if hasattr(source, key):
                    setattr(view, key, deepcopy(getattr(source, key)))
        image_type = _group(shared[0], frame, "CTImageFrameTypeSequence", required=False)
        if image_type is not None:
            view.ImageType = deepcopy(image_type.FrameType)
        reconstruction = _group(shared[0], frame, "CTReconstructionSequence", required=False)
        if reconstruction is not None and hasattr(reconstruction, "ConvolutionKernel"):
            view.ConvolutionKernel = deepcopy(reconstruction.ConvolutionKernel)
        content = _group(shared[0], frame, "FrameContentSequence", required=False)
        if content is not None:
            if int(getattr(content, "TemporalPositionIndex", 1)) != 1:
                raise ValueError("Enhanced CT contains multiple temporal acquisitions")
            if hasattr(content, "FrameAcquisitionNumber"):
                view.AcquisitionNumber = content.FrameAcquisitionNumber
        view.NumberOfFrames = 1
        view.InstanceNumber = index + 1
        yield index, view


def frame_pixels(path: Path, frame_index: int | None = None) -> np.ndarray:
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    validate_encoding(ds)
    syntax = str(ds.file_meta.TransferSyntaxUID)
    # Select a tested decoder explicitly, never fall back to Pillow's potentially
    # irreversible image operations. Pydicom has a native lossless RLE decoder.
    plugin = "" if syntax in UNCOMPRESSED else "pydicom" if syntax == RLELossless else "pylibjpeg"
    return pydicom.pixels.pixel_array(path, index=frame_index, decoding_plugin=plugin, raw=True)


def write_classic_frame(record, output: Path, *, instance_number: int | None = None) -> None:
    """Private converter input; original identity remains in the source manifest."""
    pixels = frame_pixels(record.path, record.frame_index)
    source = next(ds for index, ds in frame_headers(record.path, only_index=record.frame_index) if index == record.frame_index)
    uid = record.sop_uid if record.frame_index is None else "2.25." + str(uuid5(NAMESPACE_OID, f"{record.sop_uid}:{record.frame_index}").int)
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID, meta.MediaStorageSOPInstanceUID = CTImageStorage, uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(str(output), {}, file_meta=meta, preamble=b"\0" * 128)
    for key in ("StudyInstanceUID", "SeriesInstanceUID", "FrameOfReferenceUID", "AcquisitionNumber", "IrradiationEventUID", "Modality", "Manufacturer", "PatientPosition", "ImageType", "ConvolutionKernel", "Rows", "Columns", "PixelSpacing", "SliceThickness", "SpacingBetweenSlices", "ImageOrientationPatient", "ImagePositionPatient", "InstanceNumber", "SamplesPerPixel", "PhotometricInterpretation", "BitsAllocated", "BitsStored", "HighBit", "PixelRepresentation", "RescaleSlope", "RescaleIntercept", "RescaleType"):
        if hasattr(source, key):
            setattr(ds, key, deepcopy(getattr(source, key)))
    if instance_number is not None:
        ds.InstanceNumber = instance_number
    ds.SOPClassUID, ds.SOPInstanceUID = CTImageStorage, uid
    ds.PatientName, ds.PatientID = "", ""
    ds.PixelData = pixels.astype(pixels.dtype.newbyteorder("<"), copy=False).tobytes()
    ds.save_as(output, enforce_file_format=True)
