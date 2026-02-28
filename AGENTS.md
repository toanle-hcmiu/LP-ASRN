# LP-ASRN Training Agents

This document describes the progressive training agents used in LP-ASRN v3.1 with RRDB-EA generator and PARSeq OCR.

## Overview

LP-ASRN uses a **five-stage** progressive training approach:

| Stage | Agent | Purpose | OCR |
|-------|-------|---------|-----|
| 0 | Pretrain | Fine-tune PARSeq on HR images | Training |
| 1 | Warm-up | Stabilize RRDB-EA generator | Frozen |
| 2 | LCOFL | Character-driven optimization + PSNR guardrail | Frozen |
| 3 | Fine-tune | Extended optimization with multi-loss | Frozen |
| 4 | Hard Mining | Focus on difficult samples | Frozen |

---

## Stage 0: PARSeq Pretrain Agent

**Purpose**: Fine-tune PARSeq OCR on high-resolution license plate images.

```yaml
stage0:
  name: "pretrain"
  epochs: 50
  lr: 0.0005
  loss_components: ["ocr"]
  freeze_ocr: false
```

**Behavior**:
- Loads pretrained PARSeq from HuggingFace (`baudm/parseq-base`)
- Fine-tunes with Permutation Language Modeling (PLM)
- Teacher forcing with multiple permutation orderings
- Trains on both Scenario-A (PNG) and Scenario-B (JPG) HR images
- OCR is frozen after this stage for Stages 1-2

**Duration**: 50 epochs

**Output**: `checkpoints/ocr/best.pth`

---

## Stage 1: Warm-up Agent

**Purpose**: Initialize RRDB-EA generator with stable features.

```yaml
stage1:
  name: "warmup"
  epochs: 80
  lr: 0.0001
  loss_components: ["l1"]
  freeze_ocr: true
```

**Behavior**:
- L1 reconstruction loss only
- Stabilizes RRDB-EA generator before complex losses
- OCR frozen, used for monitoring only

**Duration**: 80 epochs

---

## Stage 2: LCOFL Agent

**Purpose**: Character-driven optimization with frozen PARSeq and quality guardrails.

```yaml
stage2:
  name: "lcofl"
  epochs: 200
  lr: 0.0001
  loss_components: ["l1", "lcofl", "ssim"]
  freeze_ocr: true
  update_confusion: true
  psnr_floor: 12.5
```

**Behavior**:
- LCOFL loss with classification + layout penalty
- **Standalone SSIM loss** (`lambda_ssim: 0.2`) prevents visual quality collapse
- **PSNR guardrail** (`psnr_floor: 12.5`): dynamically scales down LCOFL weight if PSNR drops below floor
- **Balanced checkpoint** (`best_balanced.pth`): saves model with best `word_acc * min(psnr/13.0, 1.0)` score
- Confusion matrix updated each epoch; character weights adapt to error patterns
- Layout penalty enforces digit/letter position constraints

**Duration**: 200 epochs

---

## Stage 3: Fine-tuning Agent

**Purpose**: Extended optimization with multi-loss supervision (OCR stays frozen).

```yaml
stage3:
  name: "finetune"
  epochs: 200
  lr: 0.00001
  loss_components: ["l1", "lcofl", "ssim", "gradient", "frequency", "edge"]
  freeze_ocr: true
```

**Behavior**:
- OCR remains **frozen** for stability (changed from earlier unfrozen plan)
- Lower LR prevents destabilization
- Six loss components: L1 + LCOFL + SSIM + gradient + frequency + edge
- Aspect ratio range narrowed to `[0.25, 0.45]` matching actual test distribution

**Duration**: 200 epochs

---

## Stage 4: Hard Mining Agent

**Purpose**: Focus training on samples OCR struggles with.

```yaml
stage4:
  name: "hard_mining"
  epochs: 50
  lr: 0.000005
  loss_components: ["l1", "lcofl"]
  freeze_ocr: true
  hard_mining:
    difficulty_alpha: 2.0
    reweight_interval: 5
```

**Behavior**:
- **HardExampleMiner**: Tracks per-sample accuracy
- **Weighted Sampling**: Prioritizes difficult samples
- **Curriculum**: Gradually shifts from easy to hard examples
- OCR frozen for stability

**Key Components**:
- `HardExampleMiner`: Per-sample difficulty tracking
- `CharacterConfusionTracker`: Character-level error patterns
- `CurriculumSampler`: Progressive difficulty curriculum

**Duration**: 50 epochs

---

## RRDB-EA Generator Components

**Architecture**: Residual-in-Residual Dense Block with Enhanced Attention

```yaml
model:
  num_features: 64              # Feature channels
  num_blocks: 12                # Number of RRDB-EA blocks
  num_layers_per_block: 3       # Dense layers per block
  use_enhanced_attention: true  # Enhanced Attention Module per block
  use_deformable: true          # Deformable convolutions
  upscale_factor: 2             # 2x super-resolution
  use_character_attention: false # MultiScale Character Attention (optional)
```

**Components**:
- **Shallow Feature Extractor**: PixelUnshuffle autoencoder with Conv layers
- **Deep Feature Extractor**: 12× RRDB-EA blocks with dense connections + Enhanced Attention Module + optional deformable convolutions, plus global residual connection
- **Upscaling Module**: PixelShuffle (2x)
- **Reconstruction Layer**: Conv3×3 → skip connection (weight=0.2, soft clamp via conditional tanh)

**Parameters**: ~3.99M total

> **Note**: A SwinIR Transformer variant exists as a backup (`src/models/generator_swinir_backup.py`). The inference script auto-detects architecture from checkpoint keys and supports both RRDB-EA and SwinIR models.

---

## PARSeq OCR Model

**Architecture**: Pretrained attention-based OCR from HuggingFace

```yaml
ocr:
  model_type: "parseq"
  pretrained_path: "baudm/parseq-base"
  freeze_ocr: true
  max_length: 7
  vocab: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

**Components**:
- **ViT Encoder**: Vision Transformer for image encoding
- **Autoregressive Decoder**: Cross-attention + self-attention
- **Permutation Language Modeling**: 6 permutations (forward + mirrored + random)
- **Character Prediction Head**: 36-character LP vocabulary mapped from native 97-char PARSeq vocab
- **PlateFormatValidator**: Post-processing correction for Brazilian/Mercosur formats
- **Preprocessing**: Resize to 32×128 + ImageNet normalization

**Parameters**: ~51M (pretrained, frozen during most training)

---

## Quality Guardrails (v3.1)

### PSNR Guardrail

Prevents LCOFL from collapsing visual quality (observed: PSNR 13.88→12.0 with high lambda_lcofl):

```
if val_psnr < psnr_floor:
    lcofl_scale = max(0.1, val_psnr / psnr_floor)
    effective_lcofl_weight = lambda_lcofl * lcofl_scale
```

Configured via `psnr_floor: 12.5` in Stage 2.

### Balanced Checkpoint

Saves the best model that combines both recognition accuracy AND visual quality:

```
balanced_score = word_acc * min(val_psnr / 13.0, 1.0)
```

Output: `best_balanced.pth` (use for submission instead of `best.pth` if PSNR collapsed)

### Standalone SSIM Loss

Separate from SSIM inside LCOFL. Added to Stage 2/3 `loss_components` as `"ssim"`:

```yaml
loss:
  lambda_ssim: 0.2  # Standalone SSIM weight
```

---

## Data Loading

### Dual-Format Support

Dataset loads both PNG and JPG images (Scenario-A uses PNG, Scenario-B uses JPG):
- Tries `lr-{i:03d}.png` first, falls back to `lr-{i:03d}.jpg`
- Same logic for HR images
- 5 image pairs per track × 20,000 tracks = ~100,000 samples

### Augmentation Pipeline

```yaml
data:
  jpeg_augment: true           # JPEG compression artifacts (bridges PNG→JPG gap)
  jpeg_quality_range: [60, 95]
  test_resolution_augment: true # Downsample to match test-public resolution
  test_resolution_prob: 0.7
  no_crop_prob: 0.3            # Skip corner cropping (test images lack corners)
  aspect_ratio_augment: true   # Vary aspect ratios
  test_aspect_range: [0.25, 0.45]
```

---

## Hyperparameter Summary

| Parameter | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|-----------|---------|---------|---------|---------|---------|
| LR | 5e-4 | 1e-4 | 1e-4 | 1e-5 | 5e-6 |
| Epochs | 50 | 80 | 200 | 200 | 50 |
| Loss | PARSeq | L1 | L1+LCOFL+SSIM | L1+LCOFL+SSIM+Grad+Freq+Edge | L1+LCOFL |
| OCR | Training | Frozen | Frozen | Frozen | Frozen |
| Primary Metric | Loss | Loss | Recog Rate | Word Acc | Word Acc |
| Special | — | — | PSNR guardrail | Narrow aspect range | Weighted sampling |

---

## Running Agents

```bash
# All stages
python scripts/train_progressive.py --config configs/lp_asrn.yaml

# Individual stages
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 0  # PARSeq Pretrain
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 1  # Warm-up
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 2  # LCOFL
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 3  # Fine-tune
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 4  # Hard Mining
```

---

## Inference

```bash
# Standard inference
python scripts/inference.py --checkpoint outputs/run_XXXXX/best.pth --data-root data/test-public

# With enhancements
python scripts/inference.py --checkpoint outputs/run_XXXXX/best_balanced.pth \
    --multi-scale --tta --jpeg-deblock --preserve-aspect

# Diagnostic mode (detailed per-track analysis)
python scripts/inference.py --checkpoint outputs/run_XXXXX/best.pth --diagnose

# OCR-only (skip super-resolution)
python scripts/inference.py --checkpoint outputs/run_XXXXX/best.pth --ocr-only
```

**Key CLI arguments**: `--multi-scale`, `--tta`, `--jpeg-deblock`, `--preserve-aspect`, `--diagnose`, `--diagnose-val`, `--ocr-only`, `--beam-width`

---

## TensorBoard Metrics

Access at `http://localhost:6007`

| Stage | Key Metrics |
|-------|-------------|
| Stage 0 | `stage0_pretrain/train_loss`, `stage0_pretrain/val_char_acc` |
| Stage 1 | `stage1_warmup/train_l1`, `stage1_warmup/val_psnr` |
| Stage 2 | `stage2_lcofl/train_lcofl`, `stage2_lcofl/val_word_acc`, `stage2_lcofl/lcofl_scale` |
| Stage 3 | `stage3_finetune/train_loss`, `stage3_finetune/val_word_acc` |
| Stage 4 | `stage4_hard_mining/train_loss`, `stage4_hard_mining/hard_mining_stats` |

---

## Troubleshooting

### Low Word Accuracy
1. Run Stage 4 (hard mining)
2. Check balanced checkpoint (`best_balanced.pth`) vs best checkpoint
3. Reduce LCOFL weight if PSNR is collapsing: `lambda_lcofl: 0.3`
4. Enable test-resolution augmentation: `test_resolution_augment: true`

### PSNR Collapse During LCOFL
1. Reduce `lambda_lcofl` (current: 0.5, was 1.5 before fix)
2. Add standalone SSIM: include `"ssim"` in `loss_components`
3. Set `psnr_floor: 12.5` to auto-scale LCOFL weight
4. Use `best_balanced.pth` checkpoint for submission

### Slow Training
1. Reduce RRDB blocks:
   ```yaml
   model:
     num_features: 48
     num_blocks: 8
     use_deformable: false
   ```

### Memory Issues
1. Reduce batch size: `batch_size: 32`
2. Reduce feature channels: `num_features: 48`
3. Disable deformable convolutions: `use_deformable: false`

---

## Configuration Reference

### Current Configuration (Active)

```yaml
model:
  num_features: 64
  num_blocks: 12
  num_layers_per_block: 3
  use_enhanced_attention: true
  use_deformable: true
  upscale_factor: 2
  use_character_attention: false

loss:
  lambda_lcofl: 0.5
  lambda_layout: 0.5
  lambda_ssim: 0.2
  lambda_gradient: 0.05
  lambda_frequency: 0.05
  lambda_edge: 0.05
```

### Lightweight Configuration (Faster Training)

```yaml
model:
  num_features: 48
  num_blocks: 8
  num_layers_per_block: 2
  use_enhanced_attention: true
  use_deformable: false
  upscale_factor: 2
```
