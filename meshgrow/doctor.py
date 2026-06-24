"""Environment and dependency checks."""

from __future__ import annotations

import importlib
import shutil
from typing import Optional

from meshgrow.config import PipelineConfig, load_config
from meshgrow.weights import verify_weights


def check_import(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, f"OK: {module_name}"
    except ImportError as exc:
        return False, f"MISSING: {module_name} ({exc})"


def check_cli(name: str) -> tuple[bool, str]:
    path = shutil.which(name)
    if path:
        return True, f"OK: {name} ({path})"
    return False, f"MISSING: {name} not on PATH"


def run_doctor(cfg: Optional[PipelineConfig] = None) -> int:
    cfg = cfg or load_config()
    ok = True
    lines: list[str] = []

    for mod in ("yaml", "SimpleITK", "vtk", "numpy", "scipy"):
        passed, msg = check_import(mod)
        lines.append(msg)
        ok = ok and passed

    for mod in ("nnunetv2", "seqseg", "linflonet"):
        passed, msg = check_import(mod)
        lines.append(msg)
        ok = ok and passed

    passed, msg = check_import("torch")
    lines.append(msg)

    for cli in ("nnUNetv2_predict_from_modelfolder", "nnUNetv2_predict", "seqseg", "linflonet"):
        passed, msg = check_cli(cli)
        lines.append(msg)
        if cli in ("nnUNetv2_predict_from_modelfolder", "nnUNetv2_predict"):
            if passed:
                ok = ok and passed
        elif cli in ("seqseg", "linflonet"):
            ok = ok and passed

    missing_weights = verify_weights(cfg)
    if missing_weights:
        ok = False
        lines.append("Missing model weights:")
        lines.extend(f"  - {m}" for m in missing_weights)
    else:
        lines.append("OK: all configured model paths present")

    print("\n".join(lines))
    return 0 if ok else 1
