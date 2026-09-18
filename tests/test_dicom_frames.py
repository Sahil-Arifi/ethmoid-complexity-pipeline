"""All compressed bytes are generated from known synthetic pixel arrays locally."""
from copy import deepcopy
import struct

import nibabel as nib
import numpy as np
import pydicom
from pydicom.dataset import Dataset
from pydicom.encaps import encapsulate
from pydicom.uid import EnhancedCTImageStorage, ExplicitVRLittleEndian, ImplicitVRLittleEndian, RLELossless, JPEGLSLossless, JPEG2000Lossless, JPEGLossless, JPEGLosslessSV1, generate_uid
import pytest

from ethmoid_pipeline.conversion import convert_series
from ethmoid_pipeline.dicom import PreflightFailure, preflight_series
from ethmoid_pipeline.dicom_frames import frame_headers, frame_pixels
from ethmoid_pipeline.util import default_dcm2niix, sha256_file
from ethmoid_pipeline.verification import verify_conversion
from tests.synthetic import write_ct_series


def item(**values):
    ds=Dataset()
    for name,value in values.items():
        setattr(ds,name,value)
    return ds


def enhanced_fixture(tmp_path, *, variable_calibration=True):
    original=write_ct_series(tmp_path/'classic',rows=32,columns=32)
    slices=[pydicom.dcmread(path) for path in original]
    ds=deepcopy(slices[0])
    ds.SOPClassUID=EnhancedCTImageStorage
    ds.SOPInstanceUID=generate_uid()
    ds.file_meta.MediaStorageSOPClassUID=ds.SOPClassUID
    ds.file_meta.MediaStorageSOPInstanceUID=ds.SOPInstanceUID
    ds.NumberOfFrames=len(slices)
    for key in ('ImagePositionPatient','ImageOrientationPatient','PixelSpacing','SliceThickness','SpacingBetweenSlices','RescaleSlope','RescaleIntercept','RescaleType'):
        delattr(ds,key)
    shared=item(PixelMeasuresSequence=[item(PixelSpacing=[0.5,0.5],SliceThickness=0.625,SpacingBetweenSlices=0.625)],PlaneOrientationSequence=[item(ImageOrientationPatient=[1,0,0,0,1,0])],CTImageFrameTypeSequence=[item(FrameType=['ORIGINAL','PRIMARY','VOLUME','NONE'])])
    ds.SharedFunctionalGroupsSequence=[shared]
    ds.PerFrameFunctionalGroupsSequence=[]
    expected=[]
    # Deliberately scramble frames; order must come from physical positions.
    order=[4,0,5,2,1,3]
    for index in order:
        slope=1+index*0.125 if variable_calibration else 1
        intercept=-1024+index if variable_calibration else -1024
        ds.PerFrameFunctionalGroupsSequence.append(item(PlanePositionSequence=[item(ImagePositionPatient=slices[index].ImagePositionPatient)],PixelValueTransformationSequence=[item(RescaleSlope=slope,RescaleIntercept=intercept,RescaleType='HU')],FrameContentSequence=[item(TemporalPositionIndex=1,FrameAcquisitionNumber=1)]))
        expected.append(slices[index].pixel_array.astype(np.float64)*slope+intercept)
    ds.PixelData=np.stack([slices[i].pixel_array for i in order]).tobytes()
    path=tmp_path/'enhanced.dcm'
    ds.save_as(path,enforce_file_format=True)
    return path,expected,order


def jpeg_lossless(array):
    """Small SOF3 predictor-1 encoder for a known 16-bit grayscale test matrix.

    Uses canonical five-bit DC codes for categories 0..16 and no restart marker.
    It is a fixture generator, not a production codec.
    """
    def marker(code,payload):
        return b'\xff'+bytes([code])+struct.pack('>H',len(payload)+2)+payload
    height,width=array.shape
    counts=bytes([0,0,0,0,17]+[0]*11)
    header=b'\xff\xd8'+marker(0xc4,b'\0'+counts+bytes(range(17)))
    header+=marker(0xc3,struct.pack('>BHHBBBB',16,height,width,1,1,0x11,0))
    header+=marker(0xda,bytes([1,1,0,1,0,0]))
    bits=[]
    for r in range(height):
        for c in range(width):
            prediction=int(array[r,c-1]) if c else int(array[r-1,0]) if r else 32768
            difference=int(array[r,c])-prediction
            category=abs(difference).bit_length()
            bits.append(f'{category:05b}')
            if category and category!=16:
                amplitude=difference if difference>0 else difference+(1<<category)-1
                bits.append(f'{amplitude:0{category}b}')
    packed=''.join(bits)
    packed+='1'*((-len(packed))%8)
    entropy=bytes(int(packed[i:i+8],2) for i in range(0,len(packed),8)).replace(b'\xff',b'\xff\0')
    return header+entropy+b'\xff\xd9'


def test_enhanced_per_frame_geometry_calibration_and_original_identity(tmp_path):
    path,expected,order=enhanced_fixture(tmp_path)
    result=preflight_series([path],profile='exploratory')
    assert len(result.records)==6
    assert [order[r.frame_index] for r in result.records]==list(range(6))
    assert [r.slope for r in result.records]==[1+i*0.125 for i in range(6)]
    for record in result.records:
        np.testing.assert_array_equal(frame_pixels(path,record.frame_index)*record.slope+record.intercept,expected[record.frame_index])
    with pytest.raises(PreflightFailure):
        preflight_series([path])


@pytest.mark.parametrize('problem',['missing','duplicate_scope','temporal','calibration','duplicate_plane'])
def test_enhanced_rejects_inconsistent_functional_groups(tmp_path,problem):
    path,_,_=enhanced_fixture(tmp_path)
    ds=pydicom.dcmread(path)
    frame=ds.PerFrameFunctionalGroupsSequence[0]
    if problem=='missing':
        del frame.PlanePositionSequence
    elif problem=='duplicate_scope':
        frame.PixelMeasuresSequence=deepcopy(ds.SharedFunctionalGroupsSequence[0].PixelMeasuresSequence)
    elif problem=='temporal':
        frame.FrameContentSequence[0].TemporalPositionIndex=2
    elif problem=='calibration':
        frame.PixelValueTransformationSequence[0].RescaleType='US'
    else:
        frame.PlanePositionSequence=deepcopy(ds.PerFrameFunctionalGroupsSequence[1].PlanePositionSequence)
    ds.save_as(path,enforce_file_format=True)
    with pytest.raises(PreflightFailure):
        preflight_series([path],profile='exploratory')


@pytest.mark.parametrize('syntax',[ExplicitVRLittleEndian,ImplicitVRLittleEndian,RLELossless,JPEGLSLossless,JPEG2000Lossless,JPEGLossless,JPEGLosslessSV1])
def test_lossless_decoders_match_known_16bit_synthetic_pixels(tmp_path,syntax):
    path=write_ct_series(tmp_path,rows=32,columns=32)[0]
    ds=pydicom.dcmread(path)
    expected=(ds.pixel_array.astype(np.uint16)+1)
    ds.PixelRepresentation=0
    ds.PixelData=expected.tobytes()
    if syntax in {ExplicitVRLittleEndian,ImplicitVRLittleEndian}:
        ds.file_meta.TransferSyntaxUID=syntax
    elif syntax in {JPEGLossless,JPEGLosslessSV1}:
        ds.PixelData=encapsulate([jpeg_lossless(expected)])
        ds['PixelData'].is_undefined_length=True
        ds.file_meta.TransferSyntaxUID=syntax
    else:
        ds.compress(syntax)
    ds.save_as(path,enforce_file_format=True)
    np.testing.assert_array_equal(frame_pixels(path),expected)


def test_lossy_history_and_lossy_transfer_syntax_rejected(tmp_path):
    paths=write_ct_series(tmp_path)
    ds=pydicom.dcmread(paths[0]); ds.LossyImageCompression='01'
    ds.save_as(paths[0],enforce_file_format=True)
    with pytest.raises(PreflightFailure,match='lossy'):
        preflight_series(paths,profile='exploratory')
    with pytest.raises(ValueError,match='lossy'):
        frame_pixels(paths[0])


@pytest.mark.integration
def test_enhanced_conversion_full_source_hu_proof(tmp_path):
    if not default_dcm2niix().is_file():
        pytest.skip('pinned converter unavailable')
    source=tmp_path/'source'; source.mkdir()
    path,_,_=enhanced_fixture(source)
    original=sha256_file(path)
    options={'pipeline_provenance':{'kind':'synthetic-enhanced','git_dirty':True},'preflight_profile':'exploratory'}
    result=convert_series(source,tmp_path/'work',[path],'Case001','Series001',default_dcm2niix(),**options)
    report=verify_conversion(source,tmp_path/'work',[path],'Case001','Series001',**options)
    assert report['status']=='AUTOMATED_PASS_PENDING_MANUAL'
    assert report['metrics']['max_absolute_hu_error']==0
    assert sha256_file(path)==original
    assert np.prod(nib.load(result['nifti']).shape)==32*32*6


@pytest.mark.integration
@pytest.mark.parametrize('syntax',[ImplicitVRLittleEndian,RLELossless,JPEGLSLossless,JPEG2000Lossless,JPEGLosslessSV1])
def test_lossless_conversion_and_all_voxel_hu_match_source(tmp_path,syntax):
    if not default_dcm2niix().is_file():
        pytest.skip('pinned converter unavailable')
    paths=write_ct_series(tmp_path/'source',rows=32,columns=32)
    for path in paths:
        ds=pydicom.dcmread(path)
        expected=ds.pixel_array.astype(np.uint16)+1
        ds.PixelRepresentation=0; ds.PixelData=expected.tobytes()
        if syntax==ImplicitVRLittleEndian:
            ds.file_meta.TransferSyntaxUID=syntax
        elif syntax==JPEGLosslessSV1:
            ds.PixelData=encapsulate([jpeg_lossless(expected)])
            ds['PixelData'].is_undefined_length=True
            ds.file_meta.TransferSyntaxUID=syntax
        else:
            ds.compress(syntax,generate_instance_uid=False)
        ds.save_as(path,enforce_file_format=True)
    hashes=[sha256_file(path) for path in paths]
    options={'pipeline_provenance':{'kind':'synthetic-lossless','git_dirty':True},'preflight_profile':'exploratory'}
    convert_series(tmp_path/'source',tmp_path/'work',paths,'Case001','Series001',default_dcm2niix(),**options)
    report=verify_conversion(tmp_path/'source',tmp_path/'work',paths,'Case001','Series001',**options)
    assert report['status']=='AUTOMATED_PASS_PENDING_MANUAL'
    assert report['metrics']['max_absolute_hu_error']==0
    assert [sha256_file(path) for path in paths]==hashes
