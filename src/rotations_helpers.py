import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from typing import Union
from enum import Enum

class View(Enum):
    """X (sagittal), Y (coronal), or Z (axial) view.
    
    The letters are assigned indices 0 to 2 to be used for indexing operations."""
    X = 0
    Y = 1
    Z = 2

def disp_slice(img_2d: Union[sitk.Image, np.ndarray]) -> None:
    """Display 2D slice using matplotlib.pyplot.
    
    :param img_2d:
    :type img_2d: sitk.Image or np.ndarray"""
    if isinstance(img_2d, sitk.Image):
        img_2d = sitk.GetArrayFromImage(img_2d)
    plt.imshow(img_2d)
    plt.axis("off")
    plt.show()

def get_center_of_rotation(img: sitk.Image) -> tuple:
    """img.TransformContinuousIndexToPhysicalPoint([
    (dimension - 1) / 2.0 for dimension in img.GetSize()])

    :param: img
    :type img: sitk.Image
    :return: img.TransformContinuousIndexToPhysicalPoint([
    (dimension - 1) / 2.0 for dimension in img.GetSize()]
    :rtype: tuple"""
    return img.TransformContinuousIndexToPhysicalPoint(
        [(dimension - 1) / 2.0 for dimension in img.GetSize()]
    )

def resample_2d_rotation(img_2d: sitk.Image, euler_2d_transform: sitk.Euler2DTransform, theta: int) -> None:
    """Apply 2D rotation to 2D image and render.
    
    euler_2d_transform's center should be set to img_2d's center before passing into this function.
    
    :param img_2d:
    :type img_2d: sitk.Image
    :param euler_2d_transform:
    :type euler_2d_transform: sitk.Euler2DTransform
    :param theta: degrees
    :type theta: int
    :return: None
    :rtype: None"""
    euler_2d_transform.SetAngle(degrees_to_radians(theta))
    rotated_slice: sitk.Image = sitk.Resample(img_2d, euler_2d_transform)
    disp_slice(rotated_slice)

def resample_3d_rotation(img_3d: sitk.Image, euler_3d_transform: sitk.Euler3DTransform, theta_x: int, theta_y: int, theta_z: int, slice: int, view: View) -> None:
    """Apply 3D rotation to 3D image and render.
    
    euler_3d_transform's center should be set to img_3d's center before passing into this function.
    
    :param img_3d:
    :type img_3d: sitk.Image
    :param euler_3d_transform:
    :type euler_3d_transform: sitk.Euler3DTransform
    :param theta_x: degrees
    :type theta_x: int
    :param theta_y: degrees
    :type theta_y: int
    :param theta_z: degrees
    :type theta_z: int
    :param slice:
    :type slice: int
    :param view:
    :type view: View
    :return: None
    :rtype: None"""
    euler_3d_transform.SetRotation(degrees_to_radians(theta_x), degrees_to_radians(theta_y), degrees_to_radians(theta_z))
    rotated: sitk.Image = sitk.Resample(img_3d, euler_3d_transform)
    rotated_slice: sitk.Image
    if view == View.X:
        rotated_slice = rotated[slice, :, :]
    elif view == View.Y:
        rotated_slice = rotated[:, slice, :]
    else:
        rotated_slice = rotated[:, :, slice]
    disp_slice(rotated_slice)

def degrees_to_radians(angle: Union[int, float]) -> float:
    """It's quite simple.

    :param num: A degree measure
    :type num: int or float
    :return: Equivalent radian measure
    :rtype: float"""
    return angle * np.pi / 180
