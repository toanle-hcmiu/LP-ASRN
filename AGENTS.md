# LP-ASRN Training Agents

This document describes the progressive training agents used in LP-ASRN v2.0.

## Overview

LP-ASRN uses a **five-stage** progressive training approach:

| Stage | Agent | Purpose | OCR |
|-------|-------|---------|-----|
| 0 | Pretrain | Train OCR on HR images | Training |
| 1 | Warm-up | Stabilize generator | Frozen |
| 2 | LCOFL | Character-driven optimization | Frozen |
| 3 | Fine-tune | Joint optimization | Unfrozen |
| 4 | **Hard Mining** | Focus on difficult samples | Frozen |

---

## Stage 0: OCR Pretrain Agent

**Purpose**: Train OCR model on high-resolution images before SR training.

```yaml
stage0:
  name: "pretrain"
  epochs: 50
  lr: 0.001
  loss_components: ["ocr"]
  freeze_ocr: false
```

**Behavior**:
- Trains SimpleCRNN OCR on HR license plate images
- CTC loss for sequence-to-sequence learning
- OCR is frozen after this stage for Stages 1-2

**Duration**: 50 epochs

---

## Stage 1: Warm-up Agent

**Purpose**: Initialize generator with stable features.

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
- Stabilizes network before complex losses

**Duration**: 30 epochs

---

## Stage 2: LCOFL Agent

**Purpose**: Character-driven optimization with frozen OCR.

```yaml
stage2:
  name: "lcofl"
  epochs: 300
  lr: 0.0001
  loss_components: ["l1", "lcofl"]
  freeze_ocr: true
  update_confusion: true
```

**Behavior**:
- LCOFL loss with classification + layout penalty
- Confusion matrix updated each epoch
- Character weights adapt to error patterns

**Duration**: 300 epochs

---

## Stage 3: Fine-tuning Agent

**Purpose**: Joint optimization of generator and OCR.

```yaml
stage3:
  name: "finetune"
  epochs: 150
  lr: 0.00001
  loss_components: ["l1", "lcofl"]
  freeze_ocr: false
```

**Behavior**:
- Both generator and OCR trainable
- Lower LR prevents destabilization
- Co-adaptation for final accuracy boost

**Duration**: 150 epochs

---

## Stage 4: Hard Mining Agent (NEW)

**Purpose**: Focus training on samples OCR struggles with.

```yaml
stage4:
  name: "hard_mining"
  epochs: 50
  lr: 0.000005
  loss_components: ["l1", "lcofl", "embedding"]
  freeze_ocr: true
  hard_mining:
    difficulty_alpha: 2.0
    reweight_interval: 5
```

**Behavior**:
- **HardExampleMiner**: Tracks per-sample accuracy
- **Weighted Sampling**: Prioritizes difficult samples
- **Embedding Loss**: Added for perceptual consistency
- **Curriculum**: Gradually shifts from easy to hard examples

**Key Components**:
- `HardExampleMiner`: Per-sample difficulty tracking
- `CharacterConfusionTracker`: Character-level error patterns
- `CurriculumSampler`: Progressive difficulty curriculum

**Duration**: 50 epochs

---

## New v2.0 Features

### Embedding Consistency Loss (LCOFL-EC)

Contrastive loss using Siamese network:

```yaml
loss:
  lambda_embed: 0.3        # Warmed up from 0
  embedding_dim: 128
  embed_margin: 2.0
```

### DCNv4 Support

3x faster deformable convolutions:

```yaml
model:
  use_dcnv4: true          # Falls back to DCNv3 if unavailable
```

### Multi-Scale Character Attention (MSCA)

Character-aware attention at multiple scales:

```yaml
model:
  use_character_attention: true
  msca_scales: [1.0, 0.5, 0.25]
```

---

## Hyperparameter Summary

| Parameter | Stage 0 | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
|-----------|---------|---------|---------|---------|---------|
| LR | 1e-3 | 1e-4 | 1e-4 | 1e-5 | 5e-6 |
| Loss | CTC | L1 | L1+LCOFL | L1+LCOFL | L1+LCOFL+Embed |
| OCR | Training | Frozen | Frozen | Unfrozen | Frozen |
| Primary Metric | Loss | Loss | Recog Rate | Word Acc | Word Acc |

---

## Running Agents

```bash
# All stages
python scripts/train_progressive.py --stage all

# Individual stages
python scripts/train_progressive.py --stage 0  # Pretrain
python scripts/train_progressive.py --stage 1  # Warm-up
python scripts/train_progressive.py --stage 2  # LCOFL
python scripts/train_progressive.py --stage 3  # Fine-tune
python scripts/train_progressive.py --stage 4  # Hard Mining
```

---

## TensorBoard Metrics

| Stage | Key Metrics |
|-------|-------------|
| Stage 0 | `stage0_pretrain/train_loss`, `stage0_pretrain/val_char_acc` |
| Stage 1 | `stage1_warmup/train_l1`, `stage1_warmup/val_psnr` |
| Stage 2 | `stage2_lcofl/train_lcofl`, `stage2_lcofl/val_word_acc` |
| Stage 3 | `stage3_finetune/train_loss`, `stage3_finetune/val_word_acc` |
| Stage 4 | `stage4_hard_mining/train_loss`, `stage4_hard_mining/embedding_loss` |

---

## Troubleshooting

### Low Word Accuracy
1. Run Stage 4 (hard mining)
2. Enable character attention: `use_character_attention: true`
3. Increase embedding loss: `lambda_embed: 0.5`

### Slow Training
1. Install DCNv4: `pip install dcnv4`
2. Reduce MSCA scales: `msca_scales: [1.0, 0.5]`

### Memory Issues
1. Use lightweight embedder: `use_lightweight_embedder: true`
2. Disable MSCA: `use_character_attention: false`
