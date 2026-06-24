"""Case discovery and per-case staging paths."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

IMAGE_EXTENSIONS = (".nii.gz", ".nii", ".mha", ".nrrd")

PIPELINE_STEPS = ("nnunet", "crop", "linflonet", "seqseg", "combine")

STEP_DIRS = {
    "nnunet": "01_nnunet",
    "crop": "02_crop",
    "linflonet": "03_linflonet",
    "seqseg": "04_seqseg",
    "combine": "05_combined",
}


def strip_image_extension(name: str) -> str:
    for ext in IMAGE_EXTENSIONS:
        if name.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def discover_cases(input_dir: Path) -> list[str]:
    """Discover case IDs from image files in a directory."""
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    cases: list[str] = []
    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue
        name = path.name
        if any(name.endswith(ext) for ext in IMAGE_EXTENSIONS):
            cases.append(strip_image_extension(name))
    if not cases:
        raise ValueError(f"No image files found in {input_dir}")
    return cases


def find_image_file(input_dir: Path, case_id: str) -> Path:
    for ext in IMAGE_EXTENSIONS:
        candidate = input_dir / f"{case_id}{ext}"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No image for case {case_id!r} in {input_dir}")


class CasePaths:
    """Staging paths for one case."""

    def __init__(self, output_root: Path, case_id: str):
        self.output_root = output_root
        self.case_id = case_id
        self.case_root = output_root / "cases" / case_id

    def step_dir(self, step: str) -> Path:
        return self.case_root / STEP_DIRS[step]

    @property
    def nnunet_input(self) -> Path:
        return self.step_dir("nnunet") / "input"

    @property
    def nnunet_output(self) -> Path:
        return self.step_dir("nnunet") / "output"

    @property
    def binary_seg(self) -> Path:
        return self.step_dir("nnunet") / "binary_seg.nii.gz"

    @property
    def subvolume(self) -> Path:
        return self.step_dir("crop") / "subvolume.nii.gz"

    @property
    def linflonet_dir(self) -> Path:
        return self.step_dir("linflonet")

    @property
    def cardiac_mesh(self) -> Path:
        return self.linflonet_dir / "meshes" / f"{self.case_id}.vtp"

    @property
    def seqseg_data_dir(self) -> Path:
        return self.step_dir("seqseg") / "data_dir"

    @property
    def seqseg_results(self) -> Path:
        return self.step_dir("seqseg") / "results"

    @property
    def combined_output_dir(self) -> Path:
        return self.step_dir("combine") / "output"

    @property
    def images_vti_dir(self) -> Path:
        return self.step_dir("combine") / "images_vti"

    @property
    def vascular_vti_dir(self) -> Path:
        return self.step_dir("combine") / "vascular_segs_vti"

    def combine_reference_image(self, img_ext: str = ".nii.gz") -> Path:
        """Image grid used by SeqSeg; preferred reference for combine."""
        images_dir = self.seqseg_data_dir / "images"
        for ext in (img_ext, ".nii.gz", ".nii", ".mha", ".nrrd"):
            candidate = images_dir / f"{self.case_id}{ext}"
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(f"SeqSeg input image not found under {images_dir}")

    @property
    def final_mesh(self) -> Path:
        return (
            self.combined_output_dir
            / f"{self.case_id}_LV_aorta.vtp"
        )

    def step_marker(self, step: str) -> Path:
        return self.step_dir(step) / ".complete"

    def is_step_complete(self, step: str) -> bool:
        marker = self.step_marker(step)
        if marker.is_file():
            return True
        return self._step_output_exists(step)

    def _step_output_exists(self, step: str) -> bool:
        if step == "nnunet":
            return self.binary_seg.is_file()
        if step == "crop":
            return self.subvolume.is_file()
        if step == "linflonet":
            return self.cardiac_mesh.is_file()
        if step == "seqseg":
            return self.find_vascular_seg_mha() is not None
        if step == "combine":
            return self.final_mesh.is_file()
        return False

    def mark_step_complete(self, step: str) -> None:
        self.step_marker(step).parent.mkdir(parents=True, exist_ok=True)
        self.step_marker(step).write_text("ok\n", encoding="utf-8")

    def find_vascular_seg_mha(self) -> Optional[Path]:
        results = self.seqseg_results
        if not results.is_dir():
            return None
        pattern = re.compile(
            rf"^{re.escape(self.case_id)}_segmentation_.*_steps\.mha$"
        )
        matches = [p for p in results.glob("*.mha") if pattern.match(p.name)]
        if not matches:
            matches = list(results.glob(f"{self.case_id}_segmentation_*.mha"))
        if not matches:
            return None
        return max(matches, key=lambda p: p.stat().st_mtime)


def normalize_vascular_stem(stem: str) -> str:
    return stem.replace("_seg_rem_3d_fullres_0", "")


def build_case_triplets(
    meshes_dir: Path,
    images_dir: Path,
    vascular_dir: Path,
    img_ext: str = ".vti",
    vascular_ext: str = ".vti",
) -> list[tuple[Path, Path, Path]]:
    meshes_by_stem = {p.stem: p for p in meshes_dir.glob("*.vtp")}
    imgs_by_stem = {p.stem: p for p in images_dir.glob(f"*{img_ext}")}
    vascular_by_stem: dict[str, Path] = {}
    for p in vascular_dir.glob(f"*{vascular_ext}"):
        vascular_by_stem[normalize_vascular_stem(p.stem)] = p
    common = sorted(
        set(meshes_by_stem) & set(imgs_by_stem) & set(vascular_by_stem)
    )
    return [
        (meshes_by_stem[s], imgs_by_stem[s], vascular_by_stem[s]) for s in common
    ]


def steps_from(from_step: Optional[str]) -> Iterable[str]:
    if from_step is None:
        return PIPELINE_STEPS
    if from_step not in PIPELINE_STEPS:
        raise ValueError(f"Unknown step {from_step!r}")
    idx = PIPELINE_STEPS.index(from_step)
    return PIPELINE_STEPS[idx:]


def steps_until(to_step: Optional[str]) -> Iterable[str]:
    if to_step is None:
        return PIPELINE_STEPS
    if to_step not in PIPELINE_STEPS:
        raise ValueError(f"Unknown step {to_step!r}")
    idx = PIPELINE_STEPS.index(to_step) + 1
    return PIPELINE_STEPS[:idx]
