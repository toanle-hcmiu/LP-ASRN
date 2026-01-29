# LACD-SRN Training Agents

This document describes the progressive training agents used in LACD-SRN and their configuration.

## Overview

LACD-SRN uses a three-stage progressive training approach to optimize both image quality and character recognition accuracy. Each stage focuses on specific objectives, with the final stage achieving the best recognition performance.

---

## Training Stages

### Stage 1: Warm-up Agent

**Purpose**: Initialize the network with stable features before introducing complex losses.

**Configuration**:
```yaml
progressive_training:
  stage1:
    name: "warmup"
    epochs: 10
    lr: 0.0001
    loss_components: ["l1"]      # L1 reconstruction loss only
    freeze_ocr: true           # OCR remains frozen
    update_confusion: false
```

**Behavior**:
- Trains generator with L1 (Mean Absolute Error) loss only
- OCR model is frozen (no gradients)
- Lower learning rate for stable convergence
- Stabilizes the network before adding character supervision

**Expected Outcomes**:
- Generator learns basic upsampling and reconstruction
- Stable training without oscillation
- Foundation for LCOFL training

**Duration**: 5-10 epochs

---

### Stage 2: LCOFL Agent

**Purpose**: Optimize for character recognition using the Layout and Character Oriented Focal Loss.

**Configuration**:
```yaml
progressive_training:
  stage2:
    name: "lcofl"
    epochs: 50
    lr: 0.0001
    loss_components: ["l1", "lcofl"]  # Add LCOFL loss
    freeze_ocr: true                         # OCR still frozen
    update_confusion: true                     # Update weights based on confusion
```

**Behavior**:
- Adds LCOFL loss with frozen OCR discriminator
- Confusion matrix is computed after each epoch
- Character weights are updated based on confusion frequency
- StepLR reduces learning rate by 0.9 every 5 epochs if no recognition improvement
- Monitors **recognition rate** (not loss) for scheduling

**Loss Components**:
- **L1 Loss**: Pixel-level reconstruction constraint
- **Classification Loss**: Weighted cross-entropy with adaptive weights
- **Layout Penalty**: Penalizes digit/letter position mismatches
- **SSIM Loss**: Structural similarity constraint

**Weight Update Formula**:
```
w_k = w_k + α * confusion_count(k)
```
where α = 0.1

**Expected Outcomes**:
- Character-focused reconstruction
- Improved OCR accuracy
- Adaptive handling of confused character pairs

**Duration**: 50+ epochs

---

### Stage 3: Fine-tuning Agent

**Purpose**: Joint optimization of generator and OCR for final refinement.

**Configuration**:
```yaml
progressive_training:
  stage3:
    name: "finetune"
    epochs: 20
    lr: 0.00001  # Lower learning rate
    loss_components: ["l1", "lcofl"]
    freeze_ocr: false                        # Unfreeze OCR
    update_confusion: true
```

**Behavior**:
- Both generator and OCR are trainable
- Lower learning rate to prevent destabilization
- Fine-tunes OCR to the specific generator output distribution
- Focuses on maximizing final word accuracy

**Expected Outcomes**:
- Co-adaptation of generator and OCR
- Final boost in recognition accuracy
- Optimized for the specific use case

**Duration**: 20+ epochs

---

## Agent Configuration

### Stage Selection

Training can run:
- **All stages sequentially**: `--stage all`
- **Single stage**: `--stage 1`, `--stage 2`, `--stage 3`
- **By name**: `--stage warmup`, `--stage lcofl`, `--stage finetune`

### Hyperparameters

| Parameter | Stage 1 | Stage 2 | Stage 3 |
|-----------|---------|---------|---------|
| Learning Rate | 1e-4 | 1e-4 | 1e-5 |
| Loss | L1 | L1 + LCOFL | L1 + LCOFL |
| OCR Frozen | Yes | Yes | No |
| Confusion Update | No | Yes | Yes |
| Primary Metric | Loss | Recognition Rate | Word Accuracy |

### Early Stopping

Each stage uses early stopping based on:
- **Stage 1**: Loss plateau (20 epochs without improvement)
- **Stage 2**: Recognition rate plateau
- **Stage 3**: Word accuracy plateau

Patience can be configured via `training.early_stop_patience`.

---

## LCOFL Loss Components

### Classification Loss

```
L_C = -(1/K) * Σ w_k * log(p(y_GT_k | x_SR))
```

Where:
- `K` is the maximum sequence length (7 for license plates)
- `w_k` are adaptive weights based on character confusion
- `p(y_GT_k | x_SR)` is the OCR predicted probability

### Layout Penalty

```
L_P = Σ [D(pred_i) * A(GT_i) + A(pred_i) * D(GT_i)]
```

Where:
- `D(c) = β` if character c is a digit
- `A(c) = β` if character c is a letter
- `β = 1.0` (configurable)

### Total Loss

```
L_LCOFL = L_C + λ_layout * L_P + λ_ssim * L_S
```

## Monitoring

### TensorBoard Metrics

Scalars logged:
- `train/loss`, `train/l1`, `train/lcofl`
- `val/psnr`, `val/ssim`, `val/char_acc`, `val/word_acc`
- `learning_rate`, `gradients/total_norm`

Images logged:
- Comparison grids: [LR | SR | HR] with text labels
- Sample batches for visual inspection

Histograms logged:
- Weight distributions per layer
- Gradient norms

Confusion matrices:
- Character-to-character confusion heatmaps
- Updated after each validation epoch in Stage 2 and 3

---

## Training Procedure

### 1. Preparation

```bash
# Fine-tune Parseq on HR images
python scripts/finetune_parseq.py --epochs 10
```

### 2. Progressive Training

```bash
# Run all stages (TensorBoard starts automatically on port 6007)
python scripts/train_progressive.py --stage all

# Or train individual stages
python scripts/train_progressive.py --stage 1
python scripts/train_progressive.py --stage 2 --resume checkpoints/stage1.pth
python scripts/train_progressive.py --stage 3 --resume checkpoints/stage2.pth
```

### 3. Evaluation

```bash
python scripts/evaluate.py --checkpoint checkpoints/lacd_srnn/best.pth
```

---

## Troubleshooting

### Training Instability

If you encounter training instability:

1. **Gradient explosion**: Enable gradient clipping (default: max_norm=1.0)
2. **Loss oscillation**: Reduce learning rate or increase warm-up epochs
3. **Poor character accuracy**: Check OCR fine-tuning, increase λ_layout weight
4. **Mode collapse**: Ensure Stage 1 warm-up is long enough

### Memory Issues

If running out of memory:

1. Reduce batch size
2. Reduce number of RRDB blocks
3. Use gradient checkpointing

### Low Recognition Accuracy

If word accuracy is low:

1. Verify Parseq is fine-tuned on your dataset
2. Check that plate text extraction is correct
3. Increase λ_layout weight to enforce layout constraints
4. Train for more epochs in Stage 2

---

## Advanced Configuration

### Custom Stage Configurations

You can modify stage configurations in `configs/lacd_srnn.yaml`:

```yaml
progressive_training:
  enabled: true
  stage1:
    epochs: 5          # Shorter warm-up for already-trained models
    lr: 5e-5          # Lower starting LR
  stage2:
    epochs: 100         # Longer training for difficult cases
    lr: 5e-5          # Lower LR for stability
  stage3:
    epochs: 30         # Extended fine-tuning
    lr: 1e-6          # Very low LR for joint training
```

### Loss Weight Tuning

Adjust LCOFL loss weights in the config:

```yaml
loss:
  lambda_layout: 0.5    # Layout penalty weight (try 0.2-1.0)
  lambda_ssim: 0.2      # SSIM loss weight (try 0.1-0.5)
  alpha: 0.1           # Confusion weight increment (try 0.05-0.2)
  beta: 1.0            # Layout penalty value (try 0.5-2.0)
```
