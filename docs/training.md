# LP-ASRN Training Guide

Complete guide for training the Layout-Aware and Character-Driven Super-Resolution Network with SwinIR and PARSeq.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Progressive Training](#progressive-training)
4. [New Features (v3.0)](#new-features-v30)
5. [Monitoring with TensorBoard](#monitoring-with-tensorboard)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with 12GB+ VRAM recommended (for 12.8M parameter model)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ for datasets and checkpoints

### Software Requirements

```bash
# Core dependencies
python >= 3.8
torch >= 2.0
torchvision
cuda >= 11.7

# OCR dependencies
torchtext
str == 0.9.5  # or newer

# Optional
tensorboard  # For visualization
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Data Preparation

### Dataset Structure

```
data/train/
├── Scenario-A/          # Light degradation
│   ├── Brazilian/       # LLLNNNN layout
│   │   ├── images/
│   │   └── labels.txt
│   └── Mercosur/        # LLLNLNN layout
│       ├── images/
│       └── labels.txt
└── Scenario-B/          # Heavy degradation
    ├── Brazilian/
    └── Mercosur/
```

### Label Format

Each `labels.txt` file contains one annotation per line:
```
image_name.jpg ABC1234
```

---

## Progressive Training

LP-ASRN uses a **five-stage** progressive training approach optimized for SwinIR + PARSeq.

### Training Stages Overview

| Stage | Name | Epochs | Loss | OCR | Purpose |
|-------|------|--------|------|-----|---------|
| 0 | Pretrain | 50 | PARSeq | Training | Fine-tune PARSeq on LP data |
| 1 | Warm-up | 30 | L1 | Frozen | Stabilize generator |
| 2 | LCOFL | 200 | L1 + LCOFL | Frozen | Character-driven training |
| 3 | Fine-tune | 100 | L1 + LCOFL | Unfrozen | Joint optimization |
| 4 | Hard Mining | 50 | L1 + LCOFL | Frozen | Focus on hard examples |

### Stage 0: PARSeq Pretraining

Fine-tune PARSeq on high-resolution license plate images:
```bash
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 0
```

**What happens:**
- Loads pretrained PARSeq from HuggingFace (`baudm/parseq-base`)
- Fine-tunes with Permutation Language Modeling (PLM)
- Saves to `checkpoints/ocr/best.pth`

### Stage 1: Warm-up

Stabilize SwinIR generator with simple L1 loss:
```bash
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 1
```

**What happens:**
- Creates SwinIR generator (12.8M parameters)
- Trains with L1 loss only
- OCR is frozen and used for monitoring only

### Stage 2: LCOFL Training

Character-driven optimization with frozen PARSeq:
```bash
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 2
```

**What happens:**
- L1 + LCOFL loss
- Character confusion weights updated
- Layout penalty enforced
- OCR remains frozen

### Stage 3: Fine-tuning

Joint optimization with unfrozen OCR:
```bash
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 3
```

**What happens:**
- Lower learning rate
- Both generator and OCR trainable
- Co-adaptation for better accuracy

### Stage 4: Hard Example Mining

Focus on samples OCR struggles with:
```bash
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 4
```

**What happens:**
- Tracks per-sample OCR accuracy
- Weighted sampling (harder samples = higher weight)
- OCR refrozen for stability

### Full Training

Run all stages sequentially:
```bash
python scripts/train_progressive.py --config configs/lp_asrn.yaml
```

---

## New Features (v3.0)

### 1. SwinIR Generator

Transformer-based architecture for license plate SR:

```yaml
model:
  # SwinIR Architecture (MAXIMUM for best accuracy)
  swinir_embed_dim: 144         # Embedding dimension
  swinir_num_rstb: 8            # Number of Residual Swin Transformer Blocks
  swinir_num_heads: 8           # Number of attention heads
  swinir_window_size: 6         # Window size for attention
  swinir_num_blocks_per_rstb: 3 # Swin blocks per RSTB
  swinir_mlp_ratio: 6.0         # MLP expansion ratio
```

**Benefits:**
- Better long-range modeling than CNN
- More stable training
- Higher recognition accuracy

### 2. PARSeq OCR

Pretrained attention-based OCR from HuggingFace:

```yaml
ocr:
  model_type: "parseq"          # PARSeq pretrained OCR
  pretrained_path: "baudm/parseq-base"
  freeze_ocr: true              # Keep OCR frozen during SR training
```

**Benefits:**
- Pretrained on millions of text images
- Attention-based architecture
- Autoregressive decoding with language modeling

### 3. Character Pyramid Attention

Layout-aware multi-scale character attention:

```yaml
model:
  use_pyramid_attention: true   # Layout-aware character attention
  pyramid_layout: "brazilian"   # "brazilian" or "mercocur"
```

**Benefits:**
- Stroke detection (H/V/Diagonal)
- Gap detection between characters
- Layout-aware positional encoding
- Multi-scale processing

### 4. Hard Example Mining (Stage 4)

Focus training on difficult samples:

```yaml
progressive_training:
  stage4:
    name: "hard_mining"
    epochs: 50
    lr: 0.000005
    hard_mining:
      difficulty_alpha: 2.0      # Weight exponent
      reweight_interval: 5       # Re-weight every N epochs
```

---

## Monitoring with TensorBoard

### Starting TensorBoard

Starts automatically with training. Access at:
```
http://localhost:6007
```

Or start manually:
```bash
tensorboard --logdir outputs/lp_asrn/logs --port 6007
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `train/loss` | Total training loss |
| `train/l1_loss` | L1 reconstruction loss |
| `train/lcofl_loss` | LCOFL character-driven loss |
| `val/word_acc` | Word-level accuracy (primary metric) |
| `val/char_acc` | Character-level accuracy |
| `val/psnr` | Peak Signal-to-Noise Ratio |
| `val/ssim` | Structural Similarity Index |

### Image Visualization

TensorBoard displays:
- LR images (input)
- SR images (output)
- HR images (ground truth)
- Updated every N epochs

---

## Configuration

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
  pyramid_layout: "brazilian"

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

---

## Troubleshooting

### Low Recognition Accuracy

1. **Enable Character Pyramid Attention**:
   ```yaml
   model:
     use_pyramid_attention: true
   ```

2. **Run Stage 4** (Hard Example Mining):
   ```bash
   python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 4
   ```

3. **Increase LCOFL Loss Weight**:
   ```yaml
   loss:
     lambda_lcofl: 2.0
   ```

### Slow Training

1. **Use Lightweight Configuration**:
   ```yaml
   model:
     swinir_embed_dim: 96
     swinir_num_rstb: 4
     use_pyramid_attention: false
   ```

2. **Reduce Batch Size** (if memory limited):
   ```yaml
   data:
     batch_size: 32  # Default: 64
   ```

### Memory Issues

1. **Disable Character Pyramid Attention**:
   ```yaml
   model:
     use_pyramid_attention: false
   ```

2. **Use Gradient Accumulation**:
   ```python
   # In training script
   accumulation_steps = 2
   ```

3. **Reduce Window Size** (for SwinIR):
   ```yaml
   model:
     swinir_window_size: 8  # Larger = fewer windows
   ```

---

## Best Practices

### Recommended Configuration

```yaml
# Optimal settings for word accuracy
model:
  swinir_embed_dim: 144
  swinir_num_rstb: 8
  swinir_num_heads: 8
  swinir_window_size: 6
  swinir_num_blocks_per_rstb: 3
  swinir_mlp_ratio: 6.0
  use_pyramid_attention: true
  pyramid_layout: "brazilian"

loss:
  lambda_lcofl: 1.0
  lambda_layout: 0.5
  lambda_ssim: 0.2
```

### Training Workflow

```bash
# 1. Full progressive training
python scripts/train_progressive.py --config configs/lp_asrn.yaml

# 2. Monitor in TensorBoard
# http://localhost:6007

# 3. Evaluate final model
python scripts/evaluate.py --checkpoint outputs/lp_asrn/best.pth --data-root data/test
```

### Tips for Best Results

1. **Always start with Stage 0** to fine-tune PARSeq on your data
2. **Use Character Pyramid Attention** for layout-aware processing
3. **Run Stage 4** for hard example mining
4. **Monitor TensorBoard** for loss curves and sample visualizations
5. **Validate regularly** to track word accuracy progression

---

## References

- [SwinIR Paper (2022)](https://arxiv.org/abs/2109.15272): Image Restoration Using Swin Transformer
- [PARSeq Paper (2022)](https://arxiv.org/abs/2207.06966): Pre-training Autoregressive Objectively
- [Paper 1 (2023)](https://arxiv.org/abs/2305.17313): Attention-based LP Super-Resolution
- [Paper 2 (2024)](https://arxiv.org/abs/2408.15103): Layout-Aware LP Super-Resolution
