# Multimodal Skin Lesion Classification on HAM10000: A Fusion-Strategy Comparison

A multimodal deep learning pipeline for 7-class skin lesion classification that fuses **dermoscopic images** with **patient metadata** (age, sex, localization), evaluated **honestly on the natural, imbalanced HAM10000 distribution** — no artificial rebalancing.

This repo systematically compares three fusion strategies — **Hadamard product**, **self/cross-attention**, and **concatenation** — across two CNN backbones (ResNet-50, EfficientNet-B4), combined with a dedicated preprocessing pipeline (hair removal + lesion segmentation), weighted focal loss, MixUp/CutMix, and 8-transform test-time augmentation (TTA).

> **Key finding:** Hadamard fusion (ResNet-50) outperforms attention-based and concatenation-based fusion on this natural-imbalance dataset — **87.57% accuracy, 59.98% balanced accuracy** — while all configurations struggle on extreme minority classes (0% recall on dermatofibroma across every configuration tested).

---

## Table of Contents

- [Motivation](#motivation)
- [Datasets](#datasets)
- [Pipeline Overview](#pipeline-overview)
- [Preprocessing](#preprocessing)
- [Metadata Encoding](#metadata-encoding)
- [Architecture](#architecture)
- [Fusion Strategies](#fusion-strategies)
- [Data Augmentation](#data-augmentation)
- [Loss Function](#loss-function)
- [Training Setup](#training-setup)
- [Test-Time Augmentation](#test-time-augmentation)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [Citation](#citation)
- [License](#license)

---

## Motivation

Most published skin lesion classifiers on HAM10000 report headline accuracies of 90–99% — but many achieve this by **artificially rebalancing** the dataset (replicating the rarest class, `df`, by 50×+) or by omitting balanced accuracy / per-class metrics entirely, which can mask near-total failure on minority classes.

This project instead:
1. Preserves HAM10000's **natural class imbalance** (66.9% `nv` vs. 1.1% `df`) in training and evaluation.
2. Reports **full per-class precision/recall/F1/AP/specificity**, not just aggregate accuracy.
3. Systematically ablates **fusion strategy** (Hadamard vs. attention vs. concatenation) under identical preprocessing/training conditions — an axis most prior multimodal work does not control for.
4. Uses a **dedicated preprocessing pipeline** (hair removal + lesion-segmentation cropping) that most reviewed literature skips entirely.

---

## Datasets

| Dataset | Description | Link |
|---|---|---|
| **HAM10000** | 10,015 dermoscopic images, 7 diagnostic classes (`nv`, `mel`, `bkl`, `bcc`, `akiec`, `vasc`, `df`), plus metadata (age, sex, localization) | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) / [Kaggle](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000) |
| **HAM10000 Lesion Segmentation Masks** | Pixel-level binary lesion masks corresponding to HAM10000 images, used to isolate lesions during preprocessing | [HAM10000 Segmentations (Kaggle)](https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations) |
| **Custom Processed Dataset** | Custom dataset created by applying the HAM10000 lesion segmentation masks to the original HAM10000 images, producing segmentation-guided lesion crops used as input to the classification pipeline | [Skin Lesion Classification - HAM10000 (Kaggle)](https://www.kaggle.com/datasets/krishu22/skin-lesion-classification-ham10000) |

### Class distribution (natural, unbalanced)

| Class | Count | % |
|---|---|---|
| `nv` (melanocytic nevi) | 6,705 | 66.9% |
| `mel` (melanoma) | 1,113 | 11.1% |
| `bkl` (benign keratosis) | 1,099 | 11.0% |
| `bcc` (basal cell carcinoma) | 514 | 5.1% |
| `akiec` (actinic keratoses) | 327 | 3.3% |
| `vasc` (vascular lesions) | 142 | 1.4% |
| `df` (dermatofibroma) | 115 | 1.1% |

No oversampling, undersampling, or synthetic replication is applied to this distribution at any stage — imbalance is handled entirely at the **loss level** (weighted focal loss) and **augmentation level** (MixUp/CutMix + TTA).

---

## Pipeline Overview

```
Raw HAM10000 image + metadata
        │
        ▼
┌───────────────────────┐      ┌────────────────────────┐
│  Image Preprocessing   │      │  Metadata Preprocessing │
│  1. DullRazor hair     │      │  Age → impute+scale     │
│     removal            │      │  Sex → one-hot          │
│  2. Segmentation-guided│      │  Localization → one-hot │
│     crop + pad → 224²  │      │  → 19-dim vector         │
└───────────┬────────────┘      └────────────┬────────────┘
            │                                 │
            ▼                                 ▼
   CNN Backbone (ResNet-50 /            Metadata MLP
   EfficientNet-B4)                     (19 → 128 → 256)
   → 2048-d / 1792-d features           → 256-d features
            │                                 │
            └───────────────┬─────────────────┘
                             ▼
                 Fusion Module (one of three)
          Concatenation | Hadamard | Self+Cross-Attention
                             │
                             ▼
                    Classification Head
                             │
                             ▼
                  Weighted Focal Loss (training)
                             │
                             ▼
              8-Transform TTA (inference) → softmax avg
```

See `diagrams/architecture.png`, `diagrams/image_preprocessing.png`, `diagrams/metadata_preprocessing.png`, and `diagrams/feature_extraction_fusion.png` for visual references.

---

## Preprocessing

### 1. DullRazor Hair Removal
Removes hair artifacts that can act as spurious classification cues:
- Grayscale conversion
- 9×9 cross-shaped structuring element (`cv2.getStructuringElement`)
- Morphological **blackhat** transform to isolate hair-like structures
- Gaussian blur (3×3)
- Binary threshold (value = 10, max = 255)
- Inpainting via `cv2.INPAINT_TELEA` (radius = 6)
- The traditional DullRazor border-cropping step (`image[30:410, 30:560]`) is explicitly **skipped** to avoid information loss.

### 2. Lesion Segmentation-Guided Cropping
Uses the HAM10000 segmentation masks (linked above) to localize and isolate the lesion:
- Mask resized to match image dimensions if needed
- Lesion coordinates found via `np.where(mask > 0)`
- Bounding box padded by 20px on all sides (clipped to image bounds)
- Fallback to the unmodified image if no mask/lesion is found
- Aspect-ratio-preserving resize to **224×224** with zero-padding (Lanczos4 interpolation)

---

## Metadata Encoding

| Field | Encoding | Dim |
|---|---|---|
| Age | Median-imputed, MinMax-scaled + missingness flag | 2 |
| Sex | One-hot (`sex_male`, `sex_female`, no drop) | 2 |
| Localization | One-hot over 15 anatomical regions | 15 |
| **Total metadata vector** | | **19** |

Labels are integer-mapped: `{'nv':0, 'mel':1, 'bkl':2, 'bcc':3, 'akiec':4, 'vasc':5, 'df':6}`.

Metadata files:
- `data/metadata/original_metadata.csv` — raw HAM10000 metadata
- `data/metadata/combined_metadata.csv` — merged/cleaned metadata used for the pipeline
- `data/metadata/training_metadata.csv` / `val_metadata.csv` — stratified 80/20 split (by `dx`, seed 42)

---

## Architecture

### Image Encoder
- **ResNet-50** (ImageNet-pretrained, FC layer removed) → **2048-d** features
- **EfficientNet-B4** (ImageNet-pretrained, `.features` output) → **1792-d** features
- Global average pooling; **no projection layer** applied post-pooling (full-dimensionality features passed directly into fusion)

### Metadata Encoder
- `Linear(19 → 128)` → `BatchNorm1d` → `ReLU` → `Dropout(0.3)`
- `Linear(128 → 256)` → `BatchNorm1d` → `ReLU` → `Dropout(0.3)`
- Output: **256-d** metadata embedding

### Classification Head
- `Linear(fused_dim → 512)` → `BatchNorm1d` → `ReLU` → `Dropout(0.5)`
- `Linear(512 → 256)` → `BatchNorm1d` → `ReLU` → `Dropout(0.4)`
- `Linear(256 → 7)` — logits (softmax/cross-entropy applied downstream)

---

## Fusion Strategies

Three fusion mechanisms are compared under identical training conditions:

| Strategy | Mechanism | Output dim |
|---|---|---|
| **Concatenation** | `torch.cat([image_features, metadata_features])` | 2304 (ResNet) / 2048 (EfficientNet) |
| **Hadamard product** | Image features linearly projected to 256-d, then element-wise multiplied with metadata features | 256 |
| **Self + Cross-Attention** | Intra-modal self-attention per modality → bidirectional cross-attention (image-as-query/metadata-as-query) → concatenated | 2304 |

---

## Data Augmentation

**Training-time transforms** (torchvision, applied in order):
1. `Resize(256, 256)`
2. `RandomResizedCrop(224, scale=(0.8, 1.0))`
3. `RandomHorizontalFlip(p=0.4)`
4. `RandomRotation(15°)`
5. `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)`
6. `RandomGrayscale(p=0.1)`
7. `ToTensor()` + ImageNet normalization

**Validation transform:** `Resize(224,224)` → `ToTensor()` → ImageNet normalization (no augmentation).

**MixUp / CutMix:**
- MixUp: `λ ~ Beta(0.2, 0.2)`, linearly mixes both images and metadata
- CutMix: `λ ~ Beta(1.0, 1.0)`, random rectangular patch swap; metadata mixed proportionally to actual patch area
- Applied per-batch: ~20% of batches get MixUp, ~20% get CutMix, 60% unaugmented
- Loss is a convex combination: `λ·L(preds, y_a) + (1-λ)·L(preds, y_b)`

---

## Loss Function

**Weighted Focal Loss** (γ = 2.0):

```
pt = softmax(logits)[target]
loss = -class_weight[target] · (1 - pt)^γ · log(pt)
```

Validation loss uses standard `CrossEntropyLoss(weight=class_weights)`.

---

## Training Setup

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW, lr = 1e-4 |
| Scheduler | `CosineAnnealingWarmRestarts(T_0=10, T_mult=1, eta_min=0.0)` |
| Batch size | 32 |
| Max epochs | 100 |
| Early stopping | Patience = 10, monitored on **validation balanced accuracy** |
| Checkpointing | Best model saved whenever validation BAcc improves |
| Data split | Stratified 80/20 by `dx`, seed 42 (not lesion-ID grouped — see [Limitations](#limitations)) |

---

## Test-Time Augmentation

8 deterministic transforms applied at inference, each following `Resize(256)` → `CenterCrop(224)` → transform → normalize:

1. Plain center crop
2. Horizontal flip
3. Vertical flip
4. Rotation ±2°
5. Rotation ±3°
6. Color jitter (brightness/contrast 0.1)
7. Horizontal flip + color jitter
8. Vertical flip + color jitter

Softmax probabilities from all 8 views are averaged before the final argmax prediction.

---

## Results

Four ablation configurations were trained under **identical** preprocessing, metadata encoding, augmentation, loss, optimizer/scheduler, TTA, and data split — varying only **backbone** and **fusion strategy**.

| Configuration | Accuracy | Balanced Accuracy | AUC |
|---|---|---|---|
| EfficientNet-B4 + Attention | 84.22% | 54.09% | 94.82 |
| ResNet-50 + Attention | 84.92% | 56.38% | 93.94 |
| ResNet-50 + Concatenation | 82.78% | 55.26% | 94.35 |
| **ResNet-50 + Hadamard** | **87.57%** | **59.98%** | 93.87 |

**🏆 Best overall: ResNet-50 + Hadamard fusion.**

### Per-class recall (df / akiec — the two hardest classes)

| Configuration | `df` recall | `akiec` recall |
|---|---|---|
| EfficientNet + Attention | 0.000 | 0.431 |
| ResNet + Attention | 0.000 | 0.138 |
| ResNet + Concatenation | 0.000 | 0.123 |
| ResNet + Hadamard | 0.000 | 0.154 |

> `df` (dermatofibroma, n=115) collapses to **0% recall in every single configuration**, regardless of backbone or fusion strategy — this is the most consistent and clinically important finding in this study, and is treated as a structural limitation of the current pipeline.

Full results, including confusion matrices, ROC curves, and per-class precision/recall/F1/AP/specificity plots, are available in `results/<config_name>/`:
- `confusion_matrix.png`
- `metrics.png` (accuracy/BAcc/AUC summary)
- `per-class_plots.png` (precision, recall, F1, AP, specificity per class)
- `roc_curve.png`
- `training_metrics.json` (full epoch-by-epoch training log)

### Key takeaways
- **Fusion effect** (backbone fixed at ResNet-50): Hadamard > Attention > Concatenation, on both accuracy and balanced accuracy.
- **Backbone effect** (fusion fixed at Attention): ResNet-50 slightly beats EfficientNet-B4 on accuracy/BAcc, but EfficientNet-B4 has higher AUC and much better `akiec` recall — the two backbones make different class-level trade-offs rather than one uniformly dominating.
- **AUC is not well correlated with balanced accuracy** across configurations — EfficientNet+Attention has the *highest* AUC but the *lowest* BAcc, reinforcing that AUC alone is an insufficient metric under severe imbalance.

---

## Repository Structure

```
sl-code/
├── README.md
├── configs/                          # (training/model configs)
├── data/
│   └── metadata/
│       ├── original_metadata.csv     # raw HAM10000 metadata
│       ├── combined_metadata.csv     # cleaned/merged metadata
│       ├── training_metadata.csv     # 80% stratified split
│       └── val_metadata.csv          # 20% stratified split
├── diagrams/
│   ├── architecture.png
│   ├── feature_extraction_fusion.png
│   ├── image_preprocessing.png
│   └── metadata_preprocessing.png
├── notebooks/
│   ├── 01_preprocessing.ipynb        # DullRazor + segmentation + metadata pipeline
│   ├── 02_resnet_hadamard.ipynb      # ResNet-50 + Hadamard fusion (best config)
│   ├── 03_resnet_attention.ipynb     # ResNet-50 + self/cross-attention fusion
│   ├── 04_resnet_concatenation.ipynb # ResNet-50 + concatenation fusion
│   └── 05_efficientnet_attention.ipynb # EfficientNet-B4 + attention fusion
└── results/
    ├── resnet_hadamard/
    ├── resnet_attention/
    ├── resnet_concat/
    └── efficientnet_attention/
        ├── confusion_matrix.png
        ├── metrics.png
        ├── per-class_plots.png
        ├── roc_curve.png
        └── training_metrics.json
```

---

## Limitations

- **0% recall on `df`** (dermatofibroma) across all four configurations — the model rarely assigns this class meaningfully separable probability mass (AUC 0.17–0.20).
- **Weak `akiec`/`mel` recall** throughout, despite weighted focal loss.
- Only 3 fusion strategies and 2 backbones tested (vs. up to 8 fusion methods in some prior work).
- **Single train/val split** — no cross-validation or multi-seed variance/statistical-significance testing.
- **Image-level split, not lesion-level split** HAM10000 contains multiple images per lesion (same lesion photographed at different angles/magnifications), but our stratified 80/20 split was performed at the **image level** (by `dx`, seed 42), not grouped by `lesion_id`. As a result, images of the *same lesion* can appear in both the training and validation sets, allowing the model to potentially memorize lesion-specific visual cues rather than learning generalizable diagnostic features. This inflates reported validation performance to an unknown degree and is a genuine methodological limitation of the current results, not merely a theoretical concern.
- **No external validation** (e.g., ISIC 2019, PAD-UFES-20, BCN20000) — all results are on the HAM10000 held-out split only.
- **TTA's independent contribution is not isolated** — an ablation with/without TTA was not run.
- No calibration (ECE) analysis.

---

## Future Work

- External validation on ISIC 2019 / PAD-UFES-20 / BCN20000.
- Explicit TTA ablation (with/without) to isolate its contribution.
- Expand fusion comparison to bilinear, gated, and tensor fusion.
- Add calibration metrics (Expected Calibration Error).
- Multi-seed / cross-validation reporting with statistical significance testing (McNemar, bootstrap CIs).
- Explore CN-SMOTE-style synthetic oversampling or focal-Dice losses as alternatives for `df`/`akiec` recall.
- Add Grad-CAM / SHAP explainability.
- **Re-run all experiments with a lesion-level (patient/lesion-ID-grouped) train/val split** instead of the current image-level split, to eliminate cross-set leakage and obtain a trustworthy estimate of true generalization performance.
- **Replace CNN backbones (ResNet-50, EfficientNet-B4) with more powerful vision transformer backbones** (e.g., ViT, Swin Transformer, or hybrid CNN-Transformer architectures) to evaluate whether stronger global-context modeling improves minority-class recall and overall balanced accuracy, consistent with trends observed in recent transformer-based literature on this dataset.

---

## Citation

If you use this pipeline, implementation, or results, please cite this repository.

### Datasets

This work uses the following datasets and data resources:

- **Custom Combined Metadata:** Garg, K. *Skin Lesion Classification - HAM10000.* Kaggle.  
  [Dataset](https://www.kaggle.com/datasets/krishu22/skin-lesion-classification-ham10000)

- **HAM10000:** Tschandl, P., Rosendahl, C., & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Scientific Data, 5, 180161 (2018).  
  [Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)

- **HAM10000 Lesion Segmentation Masks:** Tschandl, P. et al. *HAM10000 Lesion Segmentations.* Kaggle.  
  [Dataset](https://www.kaggle.com/datasets/tschandl/ham10000-lesion-segmentations)

---

## License

The code and original materials in this repository are released under the MIT License.

The original HAM10000 images and lesion segmentation masks are provided by their respective sources and are not covered by this repository's MIT License.

The custom processed dataset was created by applying the HAM10000 lesion segmentation masks to the corresponding original images and is provided as part of this research work.
