# MeshGrow

Unified pipeline for patient-specific **cardiac + vascular** simulation mesh construction:

1. **nnU-Net** — binary cardiac localization  
2. **Crop** — subvolume around the heart  
3. **LinFlo-Net** — whole-heart mesh and segmentation  
4. **SeqSeg** — aortic/vascular tracing (seeded from cardiac mesh)  
5. **Combine** — merged simulation-ready model (`{case_id}_LV_aorta.vtp`)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e .

# Pipeline tools (nnU-Net, SeqSeg, LinFlo-Net)
pip install seqseg linflonet nnunetv2

# PyTorch — install before pytorch3d; use a CUDA wheel on Linux if you have a GPU
pip install torch

# pytorch3d (required by LinFlo-Net; not on PyPI for all platforms)
pip install --no-build-isolation \
  "git+https://github.com/facebookresearch/pytorch3d.git@stable"

meshgrow download-weights --dest models/
meshgrow doctor

meshgrow run \
  --input /path/to/images \
  --output /path/to/results \
  --modality ct
```

Install **torch before pytorch3d**. On macOS, use the default CPU torch build; MeshGrow runs nnU-Net on CPU automatically (`runtime.device: auto`). GPU builds are strongly recommended on Linux for nnU-Net and SeqSeg.

`meshgrow doctor` checks nnU-Net, SeqSeg, and LinFlo-Net CLIs but not pytorch3d — verify with:

```bash
python -c "import torch; import pytorch3d; print(torch.__version__, pytorch3d.__version__)"
```

No config file is required — built-in defaults are used after weights are downloaded.

Optional project scaffold:

```bash
meshgrow init --dest ./my_project
meshgrow run --config ./my_project/pipeline.yaml \
  --input ./my_project/images \
  --output ./results \
  --modality mr
```

## Model weights

| Step | Zenodo | Path after download |
|------|--------|---------------------|
| Cardiac binary seg | [10.5281/zenodo.20804513](https://doi.org/10.5281/zenodo.20804513) | `models/cardiac/nnUNet_cardiac_weights/` |
| LinFlo-Net | [10.5281/zenodo.20802633](https://doi.org/10.5281/zenodo.20802633) | `models/linflonet/best_model.pth` |
| SeqSeg aorta | [10.5281/zenodo.15020477](https://doi.org/10.5281/zenodo.15020477) | `models/seqseg/nnUNet_results/` |

Optional: use `--cardiac-path ./nnUNet_cardiac_weights` instead of downloading from Zenodo.

## CLI

| Command | Description |
|---------|-------------|
| `meshgrow download-weights` | Fetch weights from Zenodo |
| `meshgrow doctor` | Check dependencies and model paths |
| `meshgrow init --dest DIR` | Create `pipeline.yaml` + `images/` |
| `meshgrow run` | Run full or partial pipeline |

Resume from a step:

```bash
meshgrow run --output results/ --input images/ --modality ct \
  --case case_001 --from-step seqseg
```

## Documentation

- [Installation](docs/installation.md)
- [Pipeline](docs/pipeline.md)

## Citation

When using this workflow, please cite SeqSeg, LinFlo-Net, and the underlying nnU-Net models. See the respective project pages for BibTeX entries.
