# LP-ASRN Training Guide

Complete guide for training the Layout-Aware and Character-Driven Super-Resolution Network.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Progressive Training](#progressive-training)
4. [New Features (v2.0)](#new-features-v20)
5. [Monitoring with TensorBoard](#monitoring-with-tensorboard)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM recommended
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ for datasets and checkpoints

### Software Requirements

```bash
# Core dependencies
python >= 3.8
torch >= 2.0
torchvision
cuda >= 11.7

# Optional (recommended for 3x faster training)
pip install dcnv4  # DCNv4 support
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
│   └── Mercosur/        # LLLNLNN layout
└── Scenario-B/          # Heavy degradation
    ├── Brazilian/
    └── Mercosur/
```

---

## Progressive Training

LP-ASRN uses a **five-stage** progressive training approach.

### Training Stages Overview

| Stage | Name | Epochs | Loss | OCR | Purpose |
|-------|------|--------|------|-----|---------|
| 0 | Pretrain | 50 | CTC | Training | Train OCR on HR images |
| 1 | Warm-up | 30 | L1 | Frozen | Stabilize generator |
| 2 | LCOFL | 300 | L1 + LCOFL | Frozen | Character-driven training |
| 3 | Fine-tune | 150 | L1 + LCOFL | Unfrozen | Joint optimization |
| 4 | Hard Mining | 50 | L1 + LCOFL + Embed | Frozen | Focus on hard examples |

### Stage 0: OCR Pretraining

Train OCR model on high-resolution images:
```bash
python scripts/train_progressive.py --stage 0
```

### Stage 1: Warm-up

Stabilize with simple L1 loss:
```bash
python scripts/train_progressive.py --stage 1
```

### Stage 2: LCOFL Training

Character-driven optimization with frozen OCR:
```bash
python scripts/train_progressive.py --stage 2
```

### Stage 3: Fine-tuning

Joint optimization with unfrozen OCR:
```bash
python scripts/train_progressive.py --stage 3
```

### Stage 4: Hard Example Mining (NEW)

Focus on samples OCR struggles with:
```bash
python scripts/train_progressive.py --stage 4
```

**Features:**
- **HardExampleMiner**: Tracks per-sample accuracy
- **Weighted Sampling**: Prioritizes difficult samples
- **Embedding Loss**: Added for perceptual consistency

### Full Training

Run all stages sequentially:
```bash
python scripts/train_progressive.py --stage all
```

---

## New Features (v2.0)

### 1. Embedding Consistency Loss (LCOFL-EC)

Contrastive loss using Siamese network embeddings:

```yaml
# In lp_asrn.yaml
loss:
  lambda_embed: 0.3        # Target weight (warmed up from 0)
  embedding_dim: 128       # Embedding dimension
  embed_margin: 2.0        # Contrastive margin
```

**Adaptive Warm-up**: `λ_embed` increases from 0 → 0.3 over 50 epochs.

### 2. DCNv4 Integration

3x faster deformable convolutions:

```yaml
model:
  use_dcnv4: true          # Prefer DCNv4 if available
```

Install DCNv4 (optional):
```bash
pip install dcnv4
```

### 3. Multi-Scale Character Attention (MSCA)

Character-aware attention at multiple scales:

```yaml
model:
  use_character_attention: true
  msca_scales: [1.0, 0.5, 0.25]
  msca_num_prototypes: 36    # One per character class
```

### 4. Hard Example Mining (Stage 4)

Focus training on difficult samples:

```yaml
progressive_training:
  stage4:
    name: "hard_mining"
    epochs: 50
    lr: 0.000005
    loss_components: ["l1", "lcofl", "embedding"]
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

### Key Metrics

| Metric | Description |
|--------|-------------|
| `train/loss` | Total training loss |
| `train/embedding_loss` | Embedding consistency loss |
| `val/word_acc` | Word-level accuracy (primary metric) |
| `val/char_acc` | Character-level accuracy |
| `stage4_hard_mining/*` | Stage 4 specific metrics |

---

## Troubleshooting

### Low Recognition Accuracy

1. **Enable Character Attention**:
   ```yaml
   model:
     use_character_attention: true
   ```

2. **Run Stage 4**:
   ```bash
   python scripts/train_progressive.py --stage 4
   ```

3. **Increase Embedding Loss**:
   ```yaml
   loss:
     lambda_embed: 0.5
   ```

### Slow Training

1. **Install DCNv4**:
   ```bash
   pip install dcnv4
   ```

2. **Reduce MSCA scales**:
   ```yaml
   model:
     msca_scales: [1.0, 0.5]  # Remove 0.25x
   ```

### Memory Issues

1. **Use Lightweight Embedder**:
   ```yaml
   loss:
     use_lightweight_embedder: true
   ```

2. **Disable MSCA**:
   ```yaml
   model:
     use_character_attention: false
   ```

---

## Best Practices

### Recommended Configuration

```yaml
# Optimal settings for word accuracy
model:
  num_rrdb_blocks: 12
  use_dcnv4: true
  use_character_attention: true

loss:
  lambda_embed: 0.3
  lambda_layout: 0.5

progressive_training:
  stage4:
    epochs: 50
```

### Training Workflow

```bash
# 1. Full progressive training
python scripts/train_progressive.py --stage all

# 2. Monitor in TensorBoard
# http://localhost:6007

# 3. Evaluate final model
python scripts/evaluate.py --checkpoint checkpoints/lp_asrn/best.pth
```

---

## References

- [Paper 1 (2023)](https://arxiv.org/abs/2305.17313): Attention-based LP Super-Resolution
- [Paper 2 (2024)](https://arxiv.org/abs/2408.15103): Layout-Aware LP Super-Resolution
- [Sendjasni et al. (2025)](https://arxiv.org/): Embedding Consistency for SR
