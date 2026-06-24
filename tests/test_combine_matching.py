"""Combine matching tests."""

from meshgrow.io.staging import build_case_triplets, normalize_vascular_stem


def test_vascular_stem_normalization_matches_mesh():
    vascular_stem = "0176_0000_seg_rem_3d_fullres_0"
    assert normalize_vascular_stem(vascular_stem) == "0176_0000"


def test_build_case_triplets_partial_overlap(tmp_path):
    meshes = tmp_path / "m"
    images = tmp_path / "i"
    vascular = tmp_path / "v"
    for d in (meshes, images, vascular):
        d.mkdir()
    (meshes / "a.vtp").write_text("")
    (meshes / "b.vtp").write_text("")
    (images / "a.vti").write_text("")
    (vascular / "a.vti").write_text("")
    triplets = build_case_triplets(meshes, images, vascular)
    assert len(triplets) == 1
    assert triplets[0][0].stem == "a"
