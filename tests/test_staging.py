"""Staging and case discovery tests."""

from pathlib import Path

import pytest

from meshgrow.io.staging import (
    build_case_triplets,
    discover_cases,
    normalize_vascular_stem,
    strip_image_extension,
)


def test_strip_image_extension():
    assert strip_image_extension("case_001.nii.gz") == "case_001"
    assert strip_image_extension("case.mha") == "case"


def test_discover_cases(tmp_path: Path):
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.nii.gz").write_bytes(b"x")
    (images / "b.nii.gz").write_bytes(b"x")
    cases = discover_cases(images)
    assert cases == ["a", "b"]


def test_discover_cases_empty(tmp_path: Path):
    images = tmp_path / "empty"
    images.mkdir()
    with pytest.raises(ValueError):
        discover_cases(images)


def test_normalize_vascular_stem():
    assert normalize_vascular_stem("case_seg_rem_3d_fullres_0") == "case"


def test_build_case_triplets(tmp_path: Path):
    meshes = tmp_path / "meshes"
    images = tmp_path / "images_vti"
    vascular = tmp_path / "vascular"
    meshes.mkdir()
    images.mkdir()
    vascular.mkdir()
    (meshes / "case1.vtp").write_text("x")
    (images / "case1.vti").write_text("x")
    (vascular / "case1_seg_rem_3d_fullres_0.vti").write_text("x")
    triplets = build_case_triplets(meshes, images, vascular)
    assert len(triplets) == 1
    assert triplets[0][0].stem == "case1"
