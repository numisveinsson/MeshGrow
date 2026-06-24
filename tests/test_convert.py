import SimpleITK as sitk
import numpy as np

from meshgrow.io.convert import images_share_geometry, resample_label_to_reference
from meshgrow.io.resample import resample, resample_image


def test_images_share_geometry_matches_identical_images():
    image = sitk.Image([8, 8, 8], sitk.sitkUInt8)
    image.SetSpacing((1.0, 1.0, 1.0))
    image.SetOrigin((0.0, 0.0, 0.0))
    assert images_share_geometry(image, image)


def test_images_share_geometry_differs_by_size():
    a = sitk.Image([8, 8, 8], sitk.sitkUInt8)
    b = sitk.Image([4, 8, 8], sitk.sitkUInt8)
    assert not images_share_geometry(a, b)


def test_resample_image_target_size_updates_spacing():
    label = sitk.Image([30, 36, 16], sitk.sitkUInt8)
    label.SetSpacing((0.3, 0.3, 1.0))
    resampled = resample_image(label, target_size=[10, 12, 8], order=0)
    assert np.allclose(resampled.GetSpacing(), (0.9, 0.9, 2.0))
    assert resampled.GetSize() == (11, 13, 8)


def test_resample_label_to_reference_matches_size():
    reference = sitk.Image([10, 12, 8], sitk.sitkUInt8)
    reference.SetSpacing((1.0, 1.0, 2.0))
    reference.SetOrigin((0.0, 0.0, 0.0))
    label = sitk.Image([30, 36, 24], sitk.sitkUInt8)
    label.SetSpacing((0.3333333, 0.3333333, 0.6666666))
    label.SetOrigin(reference.GetOrigin())
    label.SetDirection(reference.GetDirection())

    resampled = resample_label_to_reference(label, reference)
    assert resampled.GetSize() == reference.GetSize()
    assert resampled.GetSpacing() == reference.GetSpacing()