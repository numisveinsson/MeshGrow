# Pipeline

MeshGrow runs five resumable steps per case under `output/cases/{case_id}/`.

## Staging layout

```
output/
├── pipeline.log
└── cases/{case_id}/
    ├── 01_nnunet/binary_seg.nii.gz
    ├── 02_crop/subvolume.nii.gz
    ├── 03_linflonet/meshes/{case_id}.vtp
    ├── 04_seqseg/results/
    └── 05_combined/output/{case_id}_LV_aorta.vtp
```

## Steps

### 1. nnU-Net cardiac localization

Binary heart mask using modality-specific **3d_lowres** weights (`Dataset015` CT, `Dataset016` MR), `fold_all`, `checkpoint_best`.

### 2. Crop

Tight crop to the binary cardiac mask bounding box, plus voxel padding (default **20** voxels per side). Configure with `crop.padding_voxels` in `pipeline.yaml`.

### 3. LinFlo-Net

Whole-heart mesh (`.vtp`) on the cropped subvolume. Same checkpoint for CT and MR; `--modality` selects preprocessing.

### 4. SeqSeg

Vascular tracing on the **original full-volume** CT/MR (not the cardiac crop). Seeds come from the LinFlo-Net mesh (`cardiac_mesh: true`; aortic valve Region 8, LV Region 3). Mesh vertices stay in physical/world coordinates from the crop step, so seeds align with the global image.

SeqSeg nnU-Net weights are trained in **cm**. For typical mm NIfTI inputs, MeshGrow passes `-unit mm -scale 0.1`. Tracing is capped at **200** steps by default (`seqseg.max_n_steps` or `--seqseg-max-steps`).

| Modality | SeqSeg dataset |
|----------|----------------|
| CT | `Dataset005_SEQAORTANDFEMOMR` |
| MR | `Dataset006_SEQAORTANDFEMOCT` |

### 5. Combine

Merges cardiac mesh labels with SeqSeg vascular segmentation into a bounded simulation mesh. The final deliverable is `{case_id}_LV_aorta.vtp` under `05_combined/output/`. Set `combine.write_all: true` to also write intermediate debug VTK/VTI files.

## Configuration

Optional YAML (`config/pipeline.example.yaml`). Resolution order:

1. CLI flags (`--modality`, etc.)
2. `--config` file
3. Built-in defaults

## Troubleshooting

| Issue | Check |
|-------|--------|
| Missing weights | `meshgrow download-weights` or `--cardiac-path` |
| SeqSeg seed failure | LinFlo-Net mesh has `RegionId` labels 3 and 8 |
| Combine grid mismatch | SeqSeg may upsample output (`ASSEMBLY_SPACING_FACTOR`); combine aligns vascular labels to the SeqSeg input image grid |
| nnU-Net not found | `nnUNetv2_predict` on `PATH` after `pip install nnunetv2` |

Use `--dry-run` to print planned steps without executing GPU work.
