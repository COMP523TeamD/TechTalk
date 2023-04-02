"""Helper functions for filters.ipynb."""

import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from rotations_helpers import disp_slice

STABLE_TIME_STEP: float = 0.0625

def get_center_axial_slice(img_path: str) -> sitk.Image:
    reader: sitk.ImageFileReader = sitk.ImageFileReader()
    reader.SetFileName(img_path)
    img_3d: sitk.Image = reader.Execute()
    orient_filter: sitk.DICOMOrientImageFilter = sitk.DICOMOrientImageFilter()
    orient_filter.SetDesiredCoordinateOrientation("LPS")
    img_3d_axial: sitk.Image = orient_filter.Execute(img_3d)
    z_center: int = (img_3d.GetSize()[2] - 1) // 2
    slice_axial: sitk.Image = img_3d_axial[:, :, z_center]
    return slice_axial

def apply_anisotropic_diffusion(img: sitk.Image, conductance_param: float) -> sitk.Image:
    anisotropic_diffusion_filter = sitk.GradientAnisotropicDiffusionImageFilter()
    anisotropic_diffusion_filter.SetConductanceParameter(conductance_param)
    anisotropic_diffusion_filter.SetTimeStep(STABLE_TIME_STEP)
    return anisotropic_diffusion_filter.Execute(img)

def apply_otsu_threshold(img: sitk.Image) -> sitk.Image:
    return sitk.OtsuThresholdImageFilter().Execute(img)

def apply_binary_threshold(img: sitk.Image, lower_threshold: int, upper_threshold: float) -> sitk.Image:
    binary_threshold_filter = sitk.BinaryThresholdImageFilter()
    binary_threshold_filter.SetLowerThreshold(lower_threshold)
    binary_threshold_filter.SetUpperThreshold(upper_threshold)
    return binary_threshold_filter.Execute(img)

def remove_holes(img: sitk.Image) -> sitk.Image:
    return sitk.BinaryGrindPeakImageFilter().Execute(img)

def invert_img(img: sitk.Image) -> sitk.Image:
    return sitk.NotImageFilter().Execute(img)

# Credit: https://discourse.itk.org/t/simpleitk-extract-largest-connected-component-from-binary-image/4958
def select_largest_component(img: sitk.Image) -> sitk.Image:
    component_image = sitk.ConnectedComponentImageFilter().Execute(img)
    sorted_component_image = sitk.RelabelComponent(
        component_image, sortByObjectSize=True
    )
    largest_component = sorted_component_image == 1
    return largest_component