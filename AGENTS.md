# LP-ASRN Training Agents

This document describes the progressive training agents used in LP-ASRN v3.0 with SwinIR and PARSeq.

## Overview

LP-ASRN uses a **five-stage** progressive training approach:

| Stage | Agent | Purpose | OCR |
|-------|-------|---------|-----|
| 0 | Pretrain | Fine-tune PARSeq on HR images | Training |
| 1 | Warm-up | Stabilize SwinIR generator | Frozen |
| 2 | LCOFL | Character-driven optimization | Frozen |
| 3 | Fine-tune | Joint optimization | Unfrozen |
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
- OCR is frozen after this stage for Stages 1-2

**Duration**: 50 epochs

**Output**: `checkpoints/ocr/best.pth`

---

## Stage 1: Warm-up Agent

**Purpose**: Initialize SwinIR generator with stable features.

```yaml
stage1:
  name: "warmup"
  epochs: 30
  lr: 0.0001
  loss_components: ["l1"]
  freeze_ocr: true
```

**Behavior**:
- L1 reconstruction loss only
- Stabilizes SwinIR transformer before complex losses
- OCR frozen, used for monitoring only

**Duration**: 30 epochs

---

## Stage 2: LCOFL Agent

**Purpose**: Character-driven optimization with frozen PARSeq.

```yaml
stage2:
  name: "lcofl"
  epochs: 200
  lr: 0.0002
  loss_components: ["l1", "lcofl"]
  freeze_ocr: true
  update_confusion: true
```

**Behavior**:
- LCOFL loss with classification + layout penalty + SSIM
- Confusion matrix updated each epoch
- Character weights adapt to error patterns
- Layout penalty enforces digit/letter position constraints

**Duration**: 200 epochs

---

## Stage 3: Fine-tuning Agent

**Purpose**: Joint optimization of SwinIR generator and PARSeq OCR.

```yaml
stage3:
  name: "finetune"
  epochs: 100
  lr: 0.00001
  loss_components: ["l1", "lcofl"]
  freeze_ocr: false
```

**Behavior**:
- Both generator and OCR trainable
- Lower LR prevents destabilization
- Co-adaptation for final accuracy boost

**Duration**: 100 epochs

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
- OCR refrozen for stability

**Key Components**:
- `HardExampleMiner`: Per-sample difficulty tracking
- `CharacterConfusionTracker`: Character-level error patterns
- `CurriculumSampler`: Progressive difficulty curriculum

**Duration**: 50 epochs

---

## SwinIR Generator Components

**Architecture**: Transformer-based generator with shifted window attention

```yaml
model:
  swinir_embed_dim: 144         # Embedding dimension
  swinir_num_rstb: 8            # Number of Residual Swin Transformer Blocks
  swinir_num_heads: 8           # Number of attention heads
  swinir_window_size: 6         # Window size for attention
  swinir_num_blocks_per_rstb: 3 # Swin blocks per RSTB
  swinir_mlp_ratio: 6.0         # MLP expansion ratio
  use_pyramid_attention: true   # Character Pyramid Attention
```

**Components**:
- **Shallow Feature Extractor**: Initial convolution features
- **SwinIR Deep Features**: 8 RSTB with shifted window attention
- **Character Pyramid Attention**: Layout-aware multi-scale character focus
- **Upscaling Module**: PixelShuffle with progressive refinement
- **Reconstruction Layer**: Final output with skip connection

**Parameters**: 12.8M total

---

## PARSeq OCR Model

**Architecture**: Pretrained attention-based OCR from HuggingFace

```yaml
ocr:
  model_type: "parseq"
  pretrained_path: "baudm/parseq-base"
  freeze_ocr: true
```

**Components**:
- **ViT Encoder**: Vision Transformer for image encoding
- **Autoregressive Decoder**: Cross-attention + self-attention
- **Permutation Language Modeling**: Multiple permutation orderings
- **Character Prediction Head**: 36-character vocabulary (0-9, A-Z)

**Parameters**: 51M (pretrained, frozen during most training)

---

## New v3.0 Features

### SwinIR Transformer Architecture

- **Shifted Window Attention (W-MSA/SW-MSA)**: Efficient self-attention
- **Residual Swin Transformer Blocks (RSTB)**: Hierarchical features
- **Linear Complexity**: O(n) vs O(n²) for global attention

### Character Pyramid Attention

Layout-aware multi-scale character attention:

```yaml
model:
  use_pyramid_attention: true
  pyramid_layout: "brazilian"   # or "mercocur"
```

- Stroke detection (H/V/Diagonal)
- Gap detection between characters
- Layout-aware positional encoding
- Multi-scale processing (1.0x, 0.5x, 0.25x)

### PARSeq OCR

Pretrained attention-based OCR:

```yaml
ocr:
  model_type: "parseq"
  pretrained_path: "baudm/parseq-base"
```

- Pretrained on millions of text images
- Autoregressive decoding with language modeling
- State-of-the-art text recognition

---

## Hyperparameter Summary

| Parameter | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|-----------|---------|---------|---------|---------|---------|
| LR | 5e-4 | 1e-4 | 2e-4 | 1e-5 | 5e-6 |
| Loss | PARSeq | L1 | L1+LCOFL | L1+LCOFL | L1+LCOFL |
| OCR | Training | Frozen | Frozen | Unfrozen | Frozen |
| Primary Metric | Loss | Loss | Recog Rate | Word Acc | Word Acc |

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

## TensorBoard Metrics

Access at `http://localhost:6007`

| Stage | Key Metrics |
|-------|-------------|
| Stage 0 | `stage0_pretrain/train_loss`, `stage0_pretrain/val_char_acc` |
| Stage 1 | `stage1_warmup/train_l1`, `stage1_warmup/val_psnr` |
| Stage 2 | `stage2_lcofl/train_lcofl`, `stage2_lcofl/val_word_acc` |
| Stage 3 | `stage3_finetune/train_loss`, `stage3_finetune/val_word_acc` |
| Stage 4 | `stage4_hard_mining/train_loss`, `stage4_hard_mining/hard_mining_stats` |

---

## Troubleshooting

### Low Word Accuracy
1. Run Stage 4 (hard mining)
2. Enable pyramid attention: `use_pyramid_attention: true`
3. Increase LCOFL weight: `lambda_lcofl: 2.0`

### Slow Training
1. Use lightweight config:
   ```yaml
   model:
     swinir_embed_dim: 96
     swinir_num_rstb: 4
     use_pyramid_attention: false
   ```

### Memory Issues
1. Reduce batch size: `batch_size: 32`
2. Disable pyramid attention: `use_pyramid_attention: false`
3. Reduce window size: `swinir_window_size: 8`

---

## Configuration Reference

### Maximum Configuration (Best Accuracy)

```yaml
model:
  swinir_embed_dim: 144
  swinir_num_rstb: 8
  swinir_num_heads: 8
  swinir_window_size: 6
  swinir_num_blocks_per_rstb: 3
  swinir_mlp_ratio: 6.0
  use_pyramid_attention: true

loss:
  lambda_lcofl: 1.0
  lambda_layout: 0.5
  lambda_ssim: 0.2
```

### Lightweight Configuration (Faster Training)

```yaml
model:
  swinir_embed_dim: 96
  swinir_num_rstb: 4
  swinir_num_heads: 6
  swinir_window_size: 8
  swinir_num_blocks_per_rstb: 2
  swinir_mlp_ratio: 4.0
  use_pyramid_attention: false
```
