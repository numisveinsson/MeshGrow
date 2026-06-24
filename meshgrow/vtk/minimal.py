"""Minimal VTK helpers ported from vascular-segment-sampler."""

from __future__ import annotations

import os

import numpy as np
import vtk
from vtk.util.numpy_support import get_vtk_array_type
from vtk.util.numpy_support import numpy_to_vtk as n2v
from vtk.util.numpy_support import vtk_to_numpy as v2n


def read_geo(fname: str):
    _, ext = os.path.splitext(fname)
    if ext == ".vtp":
        reader = vtk.vtkXMLPolyDataReader()
    elif ext == ".vtu":
        reader = vtk.vtkXMLUnstructuredGridReader()
    else:
        raise ValueError(f"File extension {ext} unknown.")
    reader.SetFileName(fname)
    reader.Update()
    return reader


def write_geo(fname: str, input_data) -> None:
    _, ext = os.path.splitext(fname)
    if ext == ".vtp":
        writer = vtk.vtkXMLPolyDataWriter()
    elif ext == ".vtu":
        writer = vtk.vtkXMLUnstructuredGridWriter()
    else:
        raise ValueError(f"File extension {ext} unknown.")
    writer.SetFileName(fname)
    writer.SetInputData(input_data)
    writer.Write()


def read_img(fname: str):
    _, ext = os.path.splitext(fname)
    if ext == ".vti":
        reader = vtk.vtkXMLImageDataReader()
    else:
        raise ValueError(f"File extension {ext} unknown.")
    reader.SetFileName(fname)
    reader.Update()
    return reader


def write_img(fname: str, input_data) -> None:
    _, ext = os.path.splitext(fname)
    if ext == ".vti":
        writer = vtk.vtkXMLImageDataWriter()
    else:
        raise ValueError(f"File extension {ext} unknown.")
    writer.SetFileName(fname)
    writer.SetInputData(input_data)
    writer.Write()


def build_transform_matrix(image) -> np.ndarray:
    matrix = np.eye(4)
    matrix[:-1, :-1] = np.matmul(
        np.reshape(image.GetDirection(), (3, 3)), np.diag(image.GetSpacing())
    )
    matrix[:-1, -1] = np.array(image.GetOrigin())
    return matrix


def export_python2vtk(img: np.ndarray):
    vtk_array = n2v(
        num_array=img.flatten("F"),
        deep=True,
        array_type=get_vtk_array_type(img.dtype),
    )
    vtk_array.SetNumberOfComponents(1)
    return vtk_array


def export_sitk2vtk(sitk_im, spacing=None):
    """Convert a SimpleITK image to vtkImageData (vascular-segment-sampler)."""
    if not spacing:
        spacing = sitk_im.GetSpacing()
    import SimpleITK as sitk

    img = sitk.GetArrayFromImage(sitk_im).transpose(2, 1, 0)
    vtk_array = export_python2vtk(img)
    image_data = vtk.vtkImageData()
    image_data.SetDimensions(sitk_im.GetSize())
    image_data.GetPointData().SetScalars(vtk_array)
    image_data.SetOrigin([0.0, 0.0, 0.0])
    image_data.SetSpacing(spacing)
    matrix = build_transform_matrix(sitk_im)
    space_matrix = np.diag(list(spacing) + [1.0])
    matrix = np.matmul(matrix, np.linalg.inv(space_matrix))
    matrix = np.linalg.inv(matrix)
    vtk_matrix = vtk.vtkMatrix4x4()
    for i in range(4):
        for j in range(4):
            vtk_matrix.SetElement(i, j, matrix[i, j])
    reslice = vtk.vtkImageReslice()
    reslice.SetInputData(image_data)
    reslice.SetResliceAxes(vtk_matrix)
    reslice.SetInterpolationModeToNearestNeighbor()
    reslice.Update()
    return reslice.GetOutput(), vtk_matrix


def export_vtk2sitk(vtk_im):
    """Convert a vtkImageData reader output to SimpleITK (vascular-segment-sampler)."""
    import SimpleITK as sitk

    vtk_output = vtk_im.GetOutput()
    vtk_output.GetPointData().GetScalars().SetName("Scalars_")
    vtk_array = v2n(vtk_output.GetPointData().GetScalars())
    vtk_array = np.reshape(vtk_array, vtk_output.GetDimensions(), order="F")
    vtk_array = np.transpose(vtk_array, (2, 1, 0))
    sitk_im = sitk.GetImageFromArray(vtk_array)
    sitk_im.SetSpacing(vtk_output.GetSpacing())
    sitk_im.SetOrigin(vtk_output.GetOrigin())
    return sitk_im


def vtk_marching_cube_multi(vtk_label, bg_id):
    ids = np.unique(v2n(vtk_label.GetPointData().GetScalars()))
    ids = np.delete(ids, np.where(ids == bg_id))

    contour = vtk.vtkDiscreteMarchingCubes()
    contour.SetInputData(vtk_label)
    for index, i in enumerate(ids):
        contour.SetValue(index, i)
    contour.Update()
    return contour.GetOutput()


def smooth_polydata(
    poly,
    iteration=25,
    boundary=False,
    feature=False,
    smoothingFactor=0.0,
):
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(poly)
    smoother.SetPassBand(pow(10.0, -4.0 * smoothingFactor))
    smoother.SetBoundarySmoothing(boundary)
    smoother.SetFeatureEdgeSmoothing(feature)
    smoother.SetNumberOfIterations(iteration)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return smoother.GetOutput()


def convertPolyDataToImageData(poly, ref_im):
    ref_im.GetPointData().SetScalars(
        n2v(np.zeros(v2n(ref_im.GetPointData().GetScalars()).shape, dtype=np.int32))
    )
    ply2im = vtk.vtkPolyDataToImageStencil()
    ply2im.SetTolerance(0.05)
    ply2im.SetInputData(poly)
    ply2im.SetOutputSpacing(ref_im.GetSpacing())
    ply2im.SetInformationInput(ref_im)
    ply2im.Update()

    stencil = vtk.vtkImageStencil()
    stencil.SetInputData(ref_im)
    stencil.ReverseStencilOn()
    stencil.SetStencilData(ply2im.GetOutput())
    stencil.Update()
    output = stencil.GetOutput()

    output_array = v2n(output.GetPointData().GetScalars()).astype(np.int32)
    output.GetPointData().SetScalars(n2v(output_array))
    return output


def thresholdPolyData(poly, attr, threshold, mode):
    surface_thresh = vtk.vtkThreshold()
    surface_thresh.SetInputData(poly)
    lower, upper = threshold
    if hasattr(surface_thresh, "ThresholdBetween"):
        surface_thresh.ThresholdBetween(lower, upper)
    else:
        surface_thresh.SetLowerThreshold(lower)
        surface_thresh.SetUpperThreshold(upper)
    if mode == "cell":
        surface_thresh.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS, attr
        )
    else:
        surface_thresh.SetInputArrayToProcess(
            0, 0, 0, vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS, attr
        )
    surface_thresh.Update()
    surf_filter = vtk.vtkDataSetSurfaceFilter()
    surf_filter.SetInputData(surface_thresh.GetOutput())
    surf_filter.Update()
    return surf_filter.GetOutput()
