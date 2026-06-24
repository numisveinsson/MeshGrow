"""Step 5: combine cardiac and vascular models."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy

import SimpleITK as sitk

from meshgrow.config import PipelineConfig
from meshgrow.io import convert as ic
from meshgrow.io.staging import CasePaths, build_case_triplets
from meshgrow.vtk import minimal as vf

logger = logging.getLogger(__name__)


@dataclass
class CombineConfig:
    write_all: bool = False
    img_ext: str = ".vti"
    vascular_ext: str = ".vti"
    region_label: int = 6
    vascular_label: int = 1
    valve_label: int = 8
    keep_labels: list = field(default_factory=lambda: [1, 3, 8])
    blood_aorta_labels: list = field(default_factory=lambda: [3, 6])
    smooth_iterations: int = 50
    smooth_boundary: bool = False
    smooth_feature: bool = False
    smooth_factor: float = 0.5


def get_all_connected_polydata(poly):
    connectivity = vtk.vtkPolyDataConnectivityFilter()
    connectivity.SetInputData(poly)
    connectivity.SetExtractionModeToAllRegions()
    connectivity.ColorRegionsOn()
    connectivity.Update()
    return connectivity.GetOutput()


def multiclass_convert_polydata_to_imagedata(poly, ref_im):
    poly = get_all_connected_polydata(poly)
    out_im_py = np.zeros(vtk_to_numpy(ref_im.GetPointData().GetScalars()).shape)
    c = 0
    poly_i = vf.thresholdPolyData(poly, "RegionId", (c, c), "point")
    while poly_i.GetNumberOfPoints() > 0:
        poly_im = vf.convertPolyDataToImageData(poly_i, ref_im)
        poly_im_py = vtk_to_numpy(poly_im.GetPointData().GetScalars())
        mask = (
            (poly_im_py == 1) & (out_im_py == 0) if c == 6 else poly_im_py == 1
        )
        out_im_py[mask] = c + 1
        c += 1
        poly_i = vf.thresholdPolyData(poly, "RegionId", (c, c), "point")
    im = vtk.vtkImageData()
    im.DeepCopy(ref_im)
    im.GetPointData().SetScalars(numpy_to_vtk(out_im_py))
    return im


def get_bounding_box(seg_new, valve_label):
    indices = np.where(seg_new == valve_label)
    if indices[0].size == 0:
        return None
    return [
        np.min(indices[0]),
        np.max(indices[0]),
        np.min(indices[1]),
        np.max(indices[1]),
        np.min(indices[2]),
        np.max(indices[2]),
    ]


def combine_segs_aorta_area(
    segmentation,
    vascular,
    label=6,
    vascular_label=1,
    valve_label=8,
    keep_labels=None,
):
    keep_labels = [1, 3, 8] if keep_labels is None else keep_labels
    seg_new = vtk_to_numpy(segmentation.GetPointData().GetScalars())
    vas = vtk_to_numpy(vascular.GetPointData().GetScalars())

    dims = segmentation.GetDimensions()
    dims = (dims[2], dims[1], dims[0])
    seg_new = seg_new.reshape(dims)
    vas = vas.reshape(dims)

    bounds = get_bounding_box(seg_new, valve_label)
    if bounds is None:
        raise ValueError(f"Valve label {valve_label} not found in segmentation.")

    n_pad = 30
    bounds[0] = max(0, bounds[0] - n_pad // 2)
    bounds[1] = min(dims[0] - 1, bounds[1] + n_pad // 2)
    bounds[2] = max(0, bounds[2] - n_pad)
    bounds[3] = min(dims[1] - 1, bounds[3] + n_pad)
    bounds[4] = max(0, bounds[4] - n_pad)
    bounds[5] = min(dims[2] - 1, bounds[5] + n_pad)

    vas[bounds[0] : bounds[1], bounds[2] : bounds[3], bounds[4] : bounds[5]] = 2
    for label0 in keep_labels:
        vas[seg_new == label0] = 0
    seg_new[vas == vascular_label] = label
    seg_new[seg_new == valve_label] = label

    seg_new = seg_new.reshape(-1)
    vas = vas.reshape(-1)
    segmentation.GetPointData().SetScalars(numpy_to_vtk(seg_new))
    vascular.GetPointData().SetScalars(numpy_to_vtk(vas))
    return segmentation, vascular


def combine_blood_aorta(combined_seg, labels_keep=None):
    labels_keep = [3, 6] if labels_keep is None else labels_keep
    labels = np.unique(vtk_to_numpy(combined_seg.GetPointData().GetScalars()))
    labels = [label for label in labels if label in labels_keep]

    combined_seg_new = vtk.vtkImageData()
    combined_seg_new.DeepCopy(combined_seg)
    seg_new = vtk_to_numpy(combined_seg_new.GetPointData().GetScalars())
    seg_new[~np.isin(seg_new, labels)] = 0
    combined_seg_new.GetPointData().SetScalars(numpy_to_vtk(seg_new))
    poly = vf.vtk_marching_cube_multi(combined_seg_new, 0)
    return poly, combined_seg_new


def vtk_normals(poly):
    normals = vtk.vtkPolyDataNormals()
    normals.SetInputData(poly)
    normals.ComputePointNormalsOn()
    normals.ComputeCellNormalsOn()
    normals.FlipNormalsOn()
    normals.ConsistencyOn()
    normals.AutoOrientNormalsOn()
    normals.Update()
    return normals.GetOutput()


def remove_cells_with_region_3(polydata):
    filtered_polydata = vtk.vtkPolyData()
    filtered_cells = vtk.vtkCellArray()
    filtered_points = vtk.vtkPoints()
    filtered_scalars = vtk.vtkFloatArray()
    scalars = polydata.GetCellData().GetScalars()
    point_map = {}

    for cell_id in range(polydata.GetNumberOfCells()):
        label = scalars.GetTuple1(cell_id)
        if label != 3:
            cell = polydata.GetCell(cell_id)
            cell_points = cell.GetPoints()
            new_cell_point_ids = []
            for i in range(cell_points.GetNumberOfPoints()):
                point = cell_points.GetPoint(i)
                point_id = polydata.FindPoint(point)
                if point_id in point_map:
                    new_point_id = point_map[point_id]
                else:
                    new_point_id = filtered_points.InsertNextPoint(point)
                    point_map[point_id] = new_point_id
                new_cell_point_ids.append(new_point_id)
            filtered_cells.InsertNextCell(
                len(new_cell_point_ids), new_cell_point_ids
            )
            filtered_scalars.InsertNextValue(label)

    filtered_polydata.SetPoints(filtered_points)
    filtered_polydata.SetPolys(filtered_cells)
    filtered_polydata.GetCellData().SetScalars(filtered_scalars)
    return filtered_polydata


def convert_modelfaceid_to_int(polydata):
    labels = polydata.GetCellData().GetScalars()
    model_face_id = vtk.vtkIntArray()
    model_face_id.SetName("ModelFaceID")
    model_face_id.SetNumberOfComponents(1)
    model_face_id.SetNumberOfTuples(labels.GetNumberOfTuples())
    for i in range(labels.GetNumberOfTuples()):
        model_face_id.SetTuple1(i, int(labels.GetTuple1(i)))
    polydata.GetCellData().AddArray(model_face_id)
    return polydata


def add_cap_id(polydata):
    labels = polydata.GetCellData().GetScalars()
    cap_id = vtk.vtkIntArray()
    cap_id.SetName("CapID")
    cap_id.SetNumberOfComponents(1)
    cap_id.SetNumberOfTuples(labels.GetNumberOfTuples())

    def label_id(label):
        if label == 3:
            return 1
        if label == 6:
            return 1
        if label == 9:
            return 3
        if label == 10:
            return 2
        return 0

    for i in range(labels.GetNumberOfTuples()):
        cap_id.SetTuple1(i, label_id(labels.GetTuple1(i)))
    polydata.GetCellData().AddArray(cap_id)
    return polydata


def update_labels_based_on_polydata2(polydata1, polydata2, combine_cfg: CombineConfig):
    polydata2 = remove_cells_with_region_3(polydata2)
    labels_polydata1 = polydata1.GetCellData().GetScalars()
    labels_polydata2 = polydata2.GetCellData().GetScalars()
    point_locator = vtk.vtkPointLocator()
    point_locator.SetDataSet(polydata2)
    point_locator.BuildLocator()

    for cell_id in range(polydata1.GetNumberOfCells()):
        label_polydata1 = labels_polydata1.GetTuple1(cell_id)
        if label_polydata1 == 3:
            cell = polydata1.GetCell(cell_id)
            points = cell.GetPoints()
            match_found = False
            for point_id in range(points.GetNumberOfPoints()):
                point = points.GetPoint(point_id)
                closest_point_id = point_locator.FindClosestPoint(point)
                if closest_point_id >= 0:
                    cell_ids = vtk.vtkIdList()
                    polydata2.GetPointCells(closest_point_id, cell_ids)
                    for i in range(cell_ids.GetNumberOfIds()):
                        corresponding_cell_id = cell_ids.GetId(i)
                        label_polydata2 = labels_polydata2.GetTuple1(
                            corresponding_cell_id
                        )
                        if label_polydata2 in [2, 6]:
                            if label_polydata2 == 2:
                                labels_polydata1.SetTuple1(cell_id, 9)
                            elif label_polydata2 == 6:
                                labels_polydata1.SetTuple1(cell_id, 10)
                            match_found = True
                            break
                    if match_found:
                        break

    polydata1.Modified()
    polydata1 = vf.smooth_polydata(
        polydata1,
        iteration=combine_cfg.smooth_iterations,
        boundary=combine_cfg.smooth_boundary,
        feature=combine_cfg.smooth_feature,
        smoothingFactor=combine_cfg.smooth_factor,
    )
    polydata1.GetCellData().GetScalars().SetName("ModelFaceID")
    polydata1 = convert_modelfaceid_to_int(polydata1)
    polydata1 = add_cap_id(polydata1)
    return polydata1


def combine_config_from_pipeline(cfg: PipelineConfig) -> CombineConfig:
    return CombineConfig(
        write_all=cfg.combine_write_all,
        img_ext=cfg.combine_img_ext,
        vascular_ext=cfg.combine_vascular_ext,
        region_label=cfg.combine_region_label,
        vascular_label=cfg.combine_vascular_label,
        valve_label=cfg.combine_valve_label,
        keep_labels=cfg.combine_keep_labels,
        blood_aorta_labels=cfg.combine_blood_aorta_labels,
        smooth_iterations=cfg.combine_smooth_iterations,
        smooth_boundary=cfg.combine_smooth_boundary,
        smooth_feature=cfg.combine_smooth_feature,
        smooth_factor=cfg.combine_smooth_factor,
    )


def _out_name(case_name: str, img_ext: str, suffix: str) -> str:
    return case_name.replace(img_ext, suffix)


def process_case(
    mesh_path: Path,
    img_path: Path,
    vascular_path: Path,
    out_dir: Path,
    combine_cfg: CombineConfig,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    case_name = img_path.name
    ext = combine_cfg.img_ext

    img = vf.read_img(str(img_path)).GetOutput()
    logger.info("Processing image %s", img_path)
    poly_mesh = vf.read_geo(str(mesh_path)).GetOutput()
    logger.info("Processing mesh %s", mesh_path)

    poly_mesh_region = vf.thresholdPolyData(
        poly_mesh,
        "RegionId",
        (combine_cfg.region_label, combine_cfg.region_label),
        "point",
    )
    if combine_cfg.write_all:
        vf.write_geo(
            str(out_dir / _out_name(case_name, ext, "_region6.vtp")),
            poly_mesh_region,
        )

    segmentation = multiclass_convert_polydata_to_imagedata(poly_mesh, img)
    if combine_cfg.write_all:
        vf.write_img(
            str(out_dir / _out_name(case_name, ext, "_seg.vti")),
            segmentation,
        )

    poly = vf.vtk_marching_cube_multi(segmentation, 0)
    if combine_cfg.write_all:
        vf.write_geo(
            str(out_dir / _out_name(case_name, ext, "_seg.vtp")),
            poly,
        )

    vascular = vf.read_img(str(vascular_path)).GetOutput()
    logger.info("Processing vascular segmentation %s", vascular_path)

    combined_seg, new_vasc = combine_segs_aorta_area(
        segmentation,
        vascular,
        label=combine_cfg.region_label,
        vascular_label=combine_cfg.vascular_label,
        valve_label=combine_cfg.valve_label,
        keep_labels=combine_cfg.keep_labels,
    )

    if combine_cfg.write_all:
        vf.write_img(
            str(out_dir / _out_name(case_name, ext, "_seg_combined_area.vti")),
            combined_seg,
        )
        vf.write_img(
            str(out_dir / _out_name(case_name, ext, "_vasc_combined_area.vti")),
            new_vasc,
        )

    poly = vf.vtk_marching_cube_multi(combined_seg, 0)
    if combine_cfg.write_all:
        vf.write_geo(
            str(
                out_dir
                / _out_name(case_name, ext, "_combined_model_unsmoothed.vtp")
            ),
            poly,
        )

    poly = vf.smooth_polydata(
        poly,
        iteration=combine_cfg.smooth_iterations,
        boundary=combine_cfg.smooth_boundary,
        feature=combine_cfg.smooth_feature,
        smoothingFactor=combine_cfg.smooth_factor,
    )
    vf.write_geo(
        str(out_dir / _out_name(case_name, ext, "_combined_model.vtp")),
        poly,
    )

    combined_blood_aorta_vtp, _ = combine_blood_aorta(
        combined_seg, labels_keep=combine_cfg.blood_aorta_labels
    )
    if combine_cfg.write_all:
        vf.write_geo(
            str(out_dir / _out_name(case_name, ext, "_blood_aorta.vtp")),
            combined_blood_aorta_vtp,
        )

    blood_aorta_valve = update_labels_based_on_polydata2(
        combined_blood_aorta_vtp, poly, combine_cfg
    )
    out_path = out_dir / _out_name(case_name, ext, "_LV_aorta.vtp")
    vf.write_geo(str(out_path), blood_aorta_valve)
    return out_path


def run_combine_step(
    paths: CasePaths,
    input_image: Path,
    cfg: PipelineConfig,
    *,
    dry_run: bool = False,
) -> Path:
    final_path = paths.final_mesh
    if paths.is_step_complete("combine"):
        logger.info("Combine step already complete for %s", paths.case_id)
        return final_path

    combine_cfg = combine_config_from_pipeline(cfg)
    meshes_dir = paths.linflonet_dir / "meshes"
    paths.images_vti_dir.mkdir(parents=True, exist_ok=True)
    paths.vascular_vti_dir.mkdir(parents=True, exist_ok=True)

    ref_vti = paths.images_vti_dir / f"{paths.case_id}{combine_cfg.img_ext}"
    vasc_vti = paths.vascular_vti_dir / f"{paths.case_id}{combine_cfg.vascular_ext}"

    try:
        reference_image = paths.combine_reference_image(combine_cfg.img_ext)
    except FileNotFoundError:
        logger.warning(
            "SeqSeg staged image missing for %s; using pipeline input image",
            paths.case_id,
        )
        reference_image = input_image

    if dry_run:
        logger.info(
            "[dry-run] Would combine mesh=%s reference_image=%s ref_vti=%s vasc_vti=%s",
            paths.cardiac_mesh,
            reference_image,
            ref_vti,
            vasc_vti,
        )
        return final_path

    if reference_image.resolve() != input_image.resolve():
        logger.info(
            "Using SeqSeg staged image as combine reference: %s",
            reference_image,
        )

    reference = sitk.ReadImage(str(reference_image))
    ic.write_sitk_as_vti(reference, ref_vti)
    vascular_mha = paths.find_vascular_seg_mha()
    if vascular_mha is None:
        raise FileNotFoundError("Vascular segmentation from SeqSeg not found")
    ic.write_label_vti_to_reference(vascular_mha, reference, vasc_vti)

    triplets = build_case_triplets(
        meshes_dir,
        paths.images_vti_dir,
        paths.vascular_vti_dir,
        img_ext=combine_cfg.img_ext,
        vascular_ext=combine_cfg.vascular_ext,
    )
    match = [t for t in triplets if t[0].stem == paths.case_id]
    if not match:
        raise ValueError(f"No combine triplet for case {paths.case_id}")

    mesh_path, img_path, vascular_path = match[0]
    return process_case(
        mesh_path,
        img_path,
        vascular_path,
        paths.combined_output_dir,
        combine_cfg,
    )
