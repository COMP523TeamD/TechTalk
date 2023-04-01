import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from typing import Union


def degrees_to_radians(angle: Union[int, float]) -> float:
    return angle * np.pi / 180

def disp_slice(slice: Union[sitk.Image, np.ndarray]) -> None:
    if isinstance(slice, sitk.Image):
        slice = sitk.GetArrayFromImage(slice)
    plt.imshow(slice)
    plt.axis("off")
    plt.show()

def resample_2d_rotation(img_2d: sitk.Image, euler_2d_transform: sitk.Euler2DTransform, theta) -> None:
    euler_2d_transform.SetAngle(degrees_to_radians(theta))
    rotated: sitk.Image = sitk.Resample(img_2d, euler_2d_transform)
    plt.imshow(sitk.GetArrayFromImage(rotated))
    plt.axis("off")
    plt.show()

def resample_3d_rotation(img_3d: sitk.Image, euler_3d_transform: sitk.Euler3DTransform, theta_x, theta_y, theta_z, slice) -> None:
    euler_3d_transform.SetRotation(degrees_to_radians(theta_x), degrees_to_radians(theta_y), degrees_to_radians(theta_z))
    rotated: sitk.Image = sitk.Resample(img_3d, euler_3d_transform)
    rotated_slice: sitk.Image = rotated[:, :, slice]
    plt.imshow(sitk.GetArrayFromImage(rotated_slice))
    plt.axis("off")
    plt.show()
