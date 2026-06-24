# Installation

MeshGrow requires **Python 3.10+**. Use a virtual environment so pipeline dependencies stay isolated:

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

## Core package

```bash
pip install -e .
```

## Pipeline dependencies

MeshGrow orchestrates external tools. Install them in this order:

```bash
pip install seqseg linflonet nnunetv2

# PyTorch — required by nnU-Net, SeqSeg, and LinFlo-Net
# Linux + NVIDIA GPU: install the CUDA wheel from https://pytorch.org/get-started/locally/
pip install torch

# pytorch3d — required by LinFlo-Net (builds from source; torch must already be installed)
pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

`--no-build-isolation` is needed because pytorch3d’s build imports torch. There is no universal PyPI wheel; building from the `@stable` tag usually works on macOS and Linux.

On **macOS**, use the default CPU torch build. MeshGrow sets nnU-Net to CPU when no CUDA device is available (`runtime.device: auto` in the default config). Expect long runtimes on large CT volumes.

See the [LinFlo-Net quick start](https://github.com/ArjunNarayanan/LinFlo-Net/blob/main/docs/quick_start.md) for additional platform notes.

## Model weights

```bash
meshgrow download-weights --dest models/
meshgrow doctor
```

Weights are stored under `models/` (gitignored). Set `MESHGROW_ROOT` or `MESHGROW_WEIGHTS_DIR` to override the default location.

## Verify setup

```bash
meshgrow doctor
python -c "import torch; import pytorch3d; print('torch', torch.__version__, 'pytorch3d', pytorch3d.__version__)"
```

`meshgrow doctor` checks Python imports for the pipeline packages, CLI tools on `PATH`, and that expected checkpoint files exist under `models/`. It does not import pytorch3d — use the one-liner above to confirm LinFlo-Net will run.
