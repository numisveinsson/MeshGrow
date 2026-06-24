"""Configuration loading and path resolution."""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")


def repo_root() -> Path:
    """Return MeshGrow repository / install root."""
    env = os.environ.get("MESHGROW_ROOT")
    if env:
        return Path(env).resolve()
    return Path(__file__).resolve().parents[1]


def bundled_example_config() -> Path:
    candidates = [
        repo_root() / "config" / "pipeline.example.yaml",
        Path(__file__).resolve().parent / "data" / "pipeline.example.yaml",
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("pipeline.example.yaml not found in package or repo")


def _expand_vars(value: Any, context: dict[str, str]) -> Any:
    if isinstance(value, str):
        prev = None
        out = value
        while prev != out:
            prev = out
            out = _VAR_PATTERN.sub(
                lambda m: context.get(m.group(1), m.group(0)), out
            )
        return out
    if isinstance(value, list):
        return [_expand_vars(v, context) for v in value]
    if isinstance(value, dict):
        return {k: _expand_vars(v, context) for k, v in value.items()}
    return value


@dataclass
class PipelineConfig:
    modality: str = "ct"
    weights_dir: Path = field(default_factory=lambda: repo_root() / "models")
    models_ct_nnunet_cardiac: Path = field(default_factory=Path)
    models_mr_nnunet_cardiac: Path = field(default_factory=Path)
    linflonet: Path = field(default_factory=Path)
    seqseg_nnunet_root: Path = field(default_factory=Path)
    zenodo_cardiac_record: Optional[str] = "20804513"
    zenodo_seqseg_record: str = "15020477"
    zenodo_linflonet_record: str = "20802633"
    nnunet_configuration: str = "3d_lowres"
    nnunet_fold: str = "all"
    nnunet_checkpoint: str = "checkpoint_best.pth"
    crop_padding_voxels: int = 20
    seqseg_train_dataset_ct: str = "Dataset005_SEQAORTANDFEMOMR"
    seqseg_train_dataset_mr: str = "Dataset006_SEQAORTANDFEMOCT"
    seqseg_config_name_ct: str = "global_aorta"
    seqseg_config_name_mr: str = "global_aorta"
    seqseg_img_ext: str = ".nii.gz"
    seqseg_unit: str = "mm"
    seqseg_scale: float = 0.1
    seqseg_max_n_steps: int = 200
    combine_region_label: int = 6
    combine_vascular_label: int = 1
    combine_valve_label: int = 8
    combine_keep_labels: list[int] = field(default_factory=lambda: [1, 3, 8])
    combine_blood_aorta_labels: list[int] = field(default_factory=lambda: [3, 6])
    combine_write_all: bool = False
    combine_img_ext: str = ".vti"
    combine_vascular_ext: str = ".vti"
    combine_smooth_iterations: int = 50
    combine_smooth_boundary: bool = False
    combine_smooth_feature: bool = False
    combine_smooth_factor: float = 0.5
    runtime_continue_on_error: bool = False
    runtime_device: str = "auto"
    cardiac_weights_override: Optional[Path] = None

    def nnunet_cardiac_model(self, modality: Optional[str] = None) -> Path:
        mod = (modality or self.modality).lower()
        if mod == "mr":
            return self.models_mr_nnunet_cardiac
        return self.models_ct_nnunet_cardiac

    def seqseg_train_dataset(self, modality: Optional[str] = None) -> str:
        mod = (modality or self.modality).lower()
        if mod == "mr":
            return self.seqseg_train_dataset_mr
        return self.seqseg_train_dataset_ct

    def seqseg_config_name(self, modality: Optional[str] = None) -> str:
        mod = (modality or self.modality).lower()
        if mod == "mr":
            return self.seqseg_config_name_mr
        return self.seqseg_config_name_ct


def default_config_dict() -> dict[str, Any]:
    with bundled_example_config().open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def load_config(
    config_path: Optional[Path] = None,
    overrides: Optional[Mapping[str, Any]] = None,
) -> PipelineConfig:
    data = default_config_dict()
    if config_path is not None:
        with Path(config_path).open("r", encoding="utf-8") as fh:
            file_data = yaml.safe_load(fh) or {}
        data = _deep_merge(data, file_data)
    if overrides:
        data = _deep_merge(data, dict(overrides))

    root = repo_root()
    weights_dir = Path(
        os.environ.get("MESHGROW_WEIGHTS_DIR", data.get("weights_dir", root / "models"))
    )
    if isinstance(weights_dir, str):
        weights_dir = Path(weights_dir)

    context = {
        "MESHGROW_ROOT": str(root),
        "weights_dir": str(weights_dir),
    }
    data = _expand_vars(data, context)

    models = data.get("models", {})
    ct = models.get("ct", {})
    mr = models.get("mr", {})
    crop = data.get("crop", {})
    seqseg = data.get("seqseg", {})
    combine = data.get("combine", {})
    zenodo = data.get("zenodo", {})
    nnunet = data.get("nnunet", {})
    runtime = data.get("runtime", {})

    subvol_pad = crop.get("padding_voxels", 20)
    cfg = PipelineConfig(
        modality=data.get("modality", "ct"),
        weights_dir=Path(data.get("weights_dir", weights_dir)),
        models_ct_nnunet_cardiac=Path(
            ct.get(
                "nnunet_cardiac",
                weights_dir
                / "cardiac/nnUNet_cardiac_weights/Dataset015_HEARTTOTALSEGCT/nnUNetTrainer__nnUNetPlans__3d_lowres",
            )
        ),
        models_mr_nnunet_cardiac=Path(
            mr.get(
                "nnunet_cardiac",
                weights_dir
                / "cardiac/nnUNet_cardiac_weights/Dataset016_HEARTMMWHSMR/nnUNetTrainer__nnUNetPlans__3d_lowres",
            )
        ),
        linflonet=Path(
            models.get("linflonet", weights_dir / "linflonet/best_model.pth")
        ),
        seqseg_nnunet_root=Path(
            models.get(
                "seqseg_nnunet_root", weights_dir / "seqseg/nnUNet_results"
            )
        ),
        zenodo_cardiac_record=zenodo.get("cardiac_record"),
        zenodo_seqseg_record=str(zenodo.get("seqseg_record", "15020477")),
        zenodo_linflonet_record=str(zenodo.get("linflonet_record", "20802633")),
        nnunet_configuration=nnunet.get("configuration", "3d_lowres"),
        nnunet_fold=str(nnunet.get("fold", "all")),
        nnunet_checkpoint=nnunet.get("checkpoint", "checkpoint_best.pth"),
        crop_padding_voxels=int(subvol_pad),
        seqseg_train_dataset_ct=seqseg.get("train_dataset", {}).get(
            "ct", "Dataset005_SEQAORTANDFEMOMR"
        ),
        seqseg_train_dataset_mr=seqseg.get("train_dataset", {}).get(
            "mr", "Dataset006_SEQAORTANDFEMOCT"
        ),
        seqseg_config_name_ct=seqseg.get("config_name", {}).get(
            "ct", "global_aorta"
        ),
        seqseg_config_name_mr=seqseg.get("config_name", {}).get(
            "mr", "global_aorta"
        ),
        seqseg_img_ext=seqseg.get("img_ext", ".nii.gz"),
        seqseg_unit=seqseg.get("unit", "mm"),
        seqseg_scale=float(seqseg.get("scale", 0.1)),
        seqseg_max_n_steps=int(seqseg.get("max_n_steps", 200)),
        combine_region_label=int(combine.get("region_label", 6)),
        combine_vascular_label=int(combine.get("vascular_label", 1)),
        combine_valve_label=int(combine.get("valve_label", 8)),
        combine_keep_labels=list(combine.get("keep_labels", [1, 3, 8])),
        combine_blood_aorta_labels=list(
            combine.get("blood_aorta_labels", [3, 6])
        ),
        combine_write_all=bool(combine.get("write_all", False)),
        combine_img_ext=combine.get("img_ext", ".vti"),
        combine_vascular_ext=combine.get("vascular_ext", ".vti"),
        combine_smooth_iterations=int(combine.get("smooth_iterations", 50)),
        combine_smooth_boundary=bool(combine.get("smooth_boundary", False)),
        combine_smooth_feature=bool(combine.get("smooth_feature", False)),
        combine_smooth_factor=float(combine.get("smooth_factor", 0.5)),
        runtime_continue_on_error=bool(runtime.get("continue_on_error", False)),
        runtime_device=str(runtime.get("device", "auto")),
    )
    if overrides and overrides.get("cardiac_weights_override"):
        cfg.cardiac_weights_override = Path(overrides["cardiac_weights_override"])
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def resolve_inference_device(device: str = "auto") -> str:
    """Pick nnU-Net / PyTorch inference device (cuda, cpu, or mps)."""
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    # nnU-Net multiprocessing + MPS pin_memory is unreliable on Apple Silicon.
    return "cpu"
