# Priv-ViT: Privacy-Preserving Egocentric Activity Recognition

Lightweight Vision Transformer for on-device action recognition that maintains accuracy under aggressive visual privacy filtering.

## Results

| Privacy Level | Input Size | Top-1 | Top-5 | GFLOPs | FPS (A100) |
|---|---|---|---|---|---|
| Level 0 (baseline) | 224×224 | 27.09% | 59.05% | 1.07 | 71.7 |
| Level 1 (face blur) | 112×112 | 26.44% | 61.21% | 1.07 | 71.7 |
| Level 2 (aggressive) | 56×56 | **28.06%** | **63.81%** | 1.07 | 71.7 |

**103.6% Level-2 recovery** — aggressive privacy filtering *improved* generalisation via a regularisation effect.

## Method

![Priv-ViT architecture](assets/architecture.png)

Three-level privacy pipeline → factorised ViViT-S student ← cross-fidelity KL+CE distillation from frozen SlowFast-R50 teacher.

## Key Findings

- Privacy filtering at Level 2 outperforms the clean baseline (103.6% recovery)
- 66× compute reduction vs ViViT-B (1.07 vs 71.2 GFLOPs)
- On-device capable: 8.3 FPS on CPU, 71.7 FPS on A100

## Repository Layout

```text
priv-vit/
├── README.md
├── priv_vit_train.ipynb      # Full training & evaluation pipeline
├── process_dataset_local.py  # Local EPIC-KITCHENS clip extraction
├── requirements.txt
└── assets/
    └── architecture.png
```

## Reproduce

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Dataset download

Download RGB frame `.tar` files for participants **P01, P06, P07, P09, P11, P12** from [Academic Torrents](https://academictorrents.com/) (EPIC-KITCHENS-100) and place them under `dataset/`:

```text
dataset/
├── P01/rgb_frames/P01_101.tar
├── P06/rgb_frames/...
└── ...
```

### 3. Local dataset processing

```bash
python process_dataset_local.py
```

This reads frames directly from `.tar` archives, samples 16 frames per action segment, resizes to 224×224, and writes compressed `.npz` clips to `cv_project/processed_clips/`. Annotation CSVs are downloaded automatically.

Output structure (upload `cv_project/` to Google Drive for Colab training):

```text
cv_project/
├── annotations/
├── processed_clips/
│   ├── train/
│   └── val/
├── checkpoints/
└── logs/
```

### 4. Train on Google Colab

1. Upload the entire `cv_project/` folder to Google Drive (e.g. `My Drive/cv_project/`).
2. Open [`priv_vit_train.ipynb`](priv_vit_train.ipynb) in Colab.
3. In the **Configuration** cell, set `DRIVE_ROOT` to your Drive path (e.g. `/content/drive/MyDrive/cv_project`).
4. Select **GPU** runtime (`Runtime → Change runtime type`).
5. Run all cells top-to-bottom.

The notebook runs: privacy transforms → Priv-ViT training with SlowFast-R50 distillation → multi-level evaluation and benchmark plots.

### 5. Train locally (optional)

Set `DRIVE_ROOT = './cv_project'` in the notebook configuration cell and run with a CUDA-capable GPU. Local training skips the Colab disk cache step.

## Citation

If you use this implementation, please cite the original Priv-ViT paper and the EPIC-KITCHENS-100 dataset.
