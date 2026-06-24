"""Config loading tests."""

from pathlib import Path

from meshgrow.config import load_config, repo_root


def test_default_config_loads():
    cfg = load_config()
    assert cfg.modality == "ct"
    assert cfg.seqseg_scale == 0.1
    assert cfg.seqseg_max_n_steps == 200
    assert "Dataset005" in cfg.seqseg_train_dataset("ct")
    assert "Dataset006" in cfg.seqseg_train_dataset("mr")


def test_modality_override():
    cfg = load_config(overrides={"modality": "mr"})
    assert cfg.modality == "mr"
    assert cfg.nnunet_cardiac_model("mr") == cfg.models_mr_nnunet_cardiac


def test_weights_dir_expansion(tmp_path, monkeypatch):
    monkeypatch.setenv("MESHGROW_ROOT", str(tmp_path))
    cfg = load_config(overrides={"weights_dir": str(tmp_path / "models")})
    assert cfg.weights_dir == tmp_path / "models"
    assert str(tmp_path / "models") in str(cfg.linflonet)


def test_example_config_file_exists():
    assert (repo_root() / "config" / "pipeline.example.yaml").is_file()
