from dataclasses import replace
import json

import nibabel as nib
import numpy as np
import pydicom
import pytest

from ethmoid_pipeline.conversion import convert_series, reconcile_nifti_affine
from ethmoid_pipeline.dicom import PreflightFailure, preflight_series
from ethmoid_pipeline.geometry import prove_native_voxel_bijection
from ethmoid_pipeline.util import PipelineError, default_dcm2niix, sha256_file
from ethmoid_pipeline.verification import _full_volume_hu_metrics, verify_conversion
from tests.synthetic import rewrite, write_ct_series


def rounded_source(tmp_path, *, count=57, angle=23.4, reverse=False):
    paths = write_ct_series(tmp_path / "source", positions=np.arange(count)*.6, image_type=["DERIVED", "SECONDARY", "MPR"])
    theta = np.deg2rad(angle)
    row, column = np.array([1., 0., 0.]), np.array([0., np.cos(theta), np.sin(theta)])
    if reverse:
        column *= -1
    normal = np.cross(row, column)
    translation = np.array([31.123456789, -71.87654321, 321.123456789])
    for index, path in enumerate(paths):
        rewrite(path, ImageOrientationPatient=[*row, *column], ImagePositionPatient=(translation + np.round(normal*index*.6,3)).tolist(), SliceThickness=.6, SpacingBetweenSlices=.6)
    return paths


def biased_image(tmp_path, preflight, *, flip=False):
    iop = np.array(preflight.iop)
    normal = np.cross(iop[:3], iop[3:])
    data = np.stack([pydicom.dcmread(r.path).pixel_array for r in preflight.records], axis=2)
    affine = np.eye(4)
    lps = np.array([-1,-1,1])
    affine[:3,0] = lps*iop[3:]*preflight.pixel_spacing[0]
    affine[:3,1] = lps*iop[:3]*preflight.pixel_spacing[1]
    affine[:3,2] = lps*normal*preflight.grid_policy["original_median_interval_mm"]
    affine[:3,3] = lps*np.array(preflight.records[0].ipp)
    if flip:
        orientation = np.array([[1,-1],[2,1],[0,-1]])
        affine = affine @ nib.orientations.inv_ornt_aff(orientation,data.shape)
        data = nib.orientations.apply_orientation(data,orientation)
    image = nib.Nifti1Image(data,affine)
    image.set_qform(affine,1); image.set_sform(affine,1); image.header.set_slope_inter(1,-1024)
    path = tmp_path / "biased.nii"
    nib.save(image,path)
    return path


@pytest.mark.parametrize("flip,reverse", [(False,False),(True,False),(False,True),(True,True)])
def test_rounded_translated_mpr_reconstructs_header_without_changing_voxels(tmp_path, flip, reverse):
    paths=rounded_source(tmp_path,reverse=reverse)
    before=[sha256_file(p) for p in paths]
    preflight=preflight_series(paths,profile="exploratory")
    assert preflight.grid_policy["max_source_corner_residual_mm"] <= .001
    path=biased_image(tmp_path,preflight,flip=flip)
    image=nib.load(path);payload=path.read_bytes()[image.dataobj.offset:]
    report=reconcile_nifti_affine(path,preflight)
    reloaded=nib.load(path)
    assert path.read_bytes()[reloaded.dataobj.offset:]==payload
    assert report["method"]=="exploratory-mpr-quantized-grid-v1"
    assert max(report["reloaded_geometry"]["qform_max_world_residual_mm"],report["reloaded_geometry"]["sform_max_world_residual_mm"]) <= .001
    metrics=_full_volume_hu_metrics(reloaded,preflight.records,np.linalg.inv(reloaded.affine),physical_limit_mm=.001)
    assert metrics["max_absolute_error"]==0 and metrics["voxel_count"]==np.prod(reloaded.shape)
    assert [sha256_file(p) for p in paths]==before


def test_strict_and_native_exploratory_do_not_acquire_rounding_policy(tmp_path):
    paths=rounded_source(tmp_path,angle=7)
    with pytest.raises(PreflightFailure):
        preflight_series(paths)
    for path in paths:
        rewrite(path,ImageType=["ORIGINAL","PRIMARY","AXIAL"])
    assert preflight_series(paths).grid_policy is None
    assert preflight_series(paths,profile="exploratory").grid_policy is None


@pytest.mark.parametrize("failure", ["duplicate", "missing", "drift", "irregular", "corner"])
def test_quantized_policy_rejects_nonvolume_or_excess_corner_error(tmp_path,failure):
    paths=rounded_source(tmp_path,count=15,angle=0)
    if failure=="duplicate":
        rewrite(paths[7],ImagePositionPatient=pydicom.dcmread(paths[6]).ImagePositionPatient)
    elif failure=="missing":
        paths.pop(7)
    elif failure=="corner":
        theta=.00009
        rewrite(paths[-1],ImageOrientationPatient=[np.cos(theta),np.sin(theta),0,-np.sin(theta),np.cos(theta),0])
    else:
        for index,path in enumerate(paths):
            pos=np.array(pydicom.dcmread(path).ImagePositionPatient)
            pos[0 if failure=="drift" else 2] += index*.001 if failure=="drift" else (.003 if index==7 else 0)
            rewrite(path,ImagePositionPatient=pos.tolist())
    with pytest.raises(PreflightFailure):
        preflight_series(paths,profile="exploratory")


@pytest.mark.parametrize("amplitude,accepted", [(.001,True),(.00100002,False)])
def test_physical_hard_cap_has_no_epsilon_extension(tmp_path,amplitude,accepted):
    paths=write_ct_series(tmp_path/'source',positions=[0,2,4,6],image_type=["DERIVED","SECONDARY","MPR"])
    for path,x in zip(paths,[0,amplitude,-amplitude,0]):
        pos=list(pydicom.dcmread(path).ImagePositionPatient);pos[0]=x
        rewrite(path,ImagePositionPatient=pos)
    if accepted:
        assert preflight_series(paths,profile="exploratory").grid_policy["max_source_corner_residual_mm"]==.001
    else:
        with pytest.raises(PreflightFailure,match="physical corner residual"):
            preflight_series(paths,profile="exploratory")


def test_duplicate_planes_cannot_pass_bijection_even_with_identical_hu(tmp_path):
    paths=write_ct_series(tmp_path/'source',positions=[0,.625,1.25])
    records=preflight_series(paths).records
    affine=np.zeros((4,4));affine[0,1]=-.5;affine[1,0]=-.5;affine[2,2]=.625;affine[3,3]=1
    image=nib.Nifti1Image(np.zeros((16,20,3),dtype=np.int16),affine)
    duplicated=(records[0],replace(records[1],ipp=records[0].ipp),records[2])
    with pytest.raises(PipelineError,match="once to every consecutive"):
        _full_volume_hu_metrics(image,duplicated,np.linalg.inv(affine),physical_limit_mm=.001)


def test_bijection_rejects_in_plane_integer_translation(tmp_path):
    paths=write_ct_series(tmp_path/'source',positions=[0,.625,1.25])
    records=preflight_series(paths).records
    affine=np.zeros((4,4));affine[0,1]=-.5;affine[1,0]=-.5;affine[2,2]=.625;affine[3,3]=1;affine[0,3]=.5
    with pytest.raises(PipelineError):
        prove_native_voxel_bijection((16,20,3),affine,records)


def test_small_form_element_difference_cannot_hide_large_corner_error(tmp_path):
    paths=write_ct_series(tmp_path/'source',positions=[0,.625,1.25])
    records=tuple(replace(record,columns=512) for record in preflight_series(paths).records)
    affine=np.zeros((4,4));affine[0,1]=-.5;affine[1,0]=-.5;affine[2,2]=.625;affine[3,3]=1
    shifted=affine.copy();shifted[1,1]+=0.000004
    assert np.allclose(affine,shifted,atol=1e-4,rtol=0)
    with pytest.raises(PipelineError,match="physical limit") as failure:
        prove_native_voxel_bijection((16,512,3),shifted,records,physical_limit_mm=.001)
    assert failure.value.diagnostics["max_world_residual_mm"] > .001


@pytest.mark.integration
def test_real_converter_and_qc_use_explicit_quantization_policy(tmp_path):
    if not default_dcm2niix().is_file():
        pytest.skip("Pinned converter not installed")
    paths=rounded_source(tmp_path,count=67)
    provenance={"kind":"synthetic-quantized-MPR-test","git_dirty":True}
    result=convert_series(tmp_path/'source',tmp_path/'work',paths,"Case001","Series001",default_dcm2niix(),pipeline_provenance=provenance,preflight_profile="exploratory")
    report=verify_conversion(tmp_path/'source',tmp_path/'work',paths,"Case001","Series001",pipeline_provenance=provenance,preflight_profile="exploratory")
    assert report["status"]=="AUTOMATED_PASS_PENDING_MANUAL"
    assert report["metrics"]["max_world_residual_mm"] <= .001 and report["metrics"]["max_absolute_hu_error"]==0
    assert report["grid_policy"]["name"]=="exploratory-mpr-quantized-grid-v1"
    conversion=json.loads((result["output_dir"]/'conversion_report.json').read_text())
    assert conversion["affine_reconciliation"]["voxel_payload_unchanged"] is True
