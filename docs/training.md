# LP-ASRN Training Guide

Complete guide for training the Layout-Aware and Character-Driven Super-Resolution Network.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [OCR Fine-tuning](#ocr-fine-tuning)
4. [Progressive Training](#progressive-training)
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

# Training utilities
tensorboard >= 2.13
tqdm
pyyaml

# OCR
strq  # Parseq implementation
```

Install all dependencies:
```bash
pip install -r requirements.txt
```

---

## Data Preparation

### Dataset Structure

Organize your data as follows:

```
data/train/
├── Scenario-A/          # Light degradation
│   ├── Brazilian/       # LLLNNNN layout
│   │   ├── track_001/
│   │   │   ├── lr-001.png, lr-002.png, ...
│   │   │   ├── hr-001.png, hr-002.png, ...
│   │   │   └── annotations.json
│   │   └── ...
│   └── Mercosur/        # LLLNLNN layout
│       └── ...
└── Scenario-B/          # Heavy degradation
    ├── Brazilian/
    └── Mercosur/
```

### Annotation Format

Each `annotations.json` should contain:

```json
{
  "track_id": "track_001",
  "layout": "Brazilian",
  "plate_text": "ABC1234",
  "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]],
  "images": [
    {"lr": "lr-001.png", "hr": "hr-001.png"},
    {"lr": "lr-002.png", "hr": "hr-002.png"}
  ]
}
```

---

## OCR Fine-tuning

Before training the super-resolution model, fine-tune Parseq on your high-resolution license plate images.

### Why Fine-tune OCR?

Parseq is pre-trained on general scene text. Fine-tuning on license plates:
- Adapts to license plate fonts and styles
- Learns the specific character vocabulary
- Improves accuracy on the target domain

### Fine-tuning Command

```bash
python scripts/finetune_parseq.py \
    --data-root data/train \
    --epochs 10 \
    --batch-size 32 \
    --lr 1e-4 \
    --save-dir checkpoints/parseq
```

### Expected Results

After fine-tuning, you should see:
- Character accuracy > 95% on HR validation set
- Word accuracy > 90% on HR validation set

---

## Progressive Training

LP-ASRN uses a three-stage progressive training approach for stability and performance.

### Stage 1: Warm-up

**Purpose**: Stabilize the network with simple reconstruction loss before introducing character supervision.

**Configuration**:
- Loss: L1 (pixel-level reconstruction)
- OCR: Frozen
- Learning Rate: 1e-4
- Duration: 5-10 epochs

**What happens**:
- Generator learns basic upsampling
- Features stabilize without complex loss landscape
- Foundation for LCOFL training

**Run standalone**:
```bash
python scripts/train_progressive.py \
    --stage 1 \
    --epochs 10 \
    --config configs/lp_asrn.yaml
```

### Stage 2: LCOFL Training

**Purpose**: Optimize for character recognition using the Layout and Character Oriented Focal Loss.

**Configuration**:
- Loss: L1 + LCOFL
- OCR: Frozen (provides stable gradients)
- Learning Rate: 1e-4
- Duration: 50+ epochs

**What happens**:
- Character-focused reconstruction
- Confusion matrix updated after each epoch
- Adaptive weights for confused character pairs
- StepLR reduces LR if recognition plateaus

**Run standalone**:
```bash
python scripts/train_progressive.py \
    --stage 2 \
    --resume checkpoints/stage1.pth \
    --config configs/lp_asrn.yaml
```

### Stage 3: Fine-tuning

**Purpose**: Joint optimization of generator and OCR for final refinement.

**Configuration**:
- Loss: L1 + LCOFL
- OCR: Unfrozen (joint training)
- Learning Rate: 1e-5 (lower for stability)
- Duration: 20+ epochs

**What happens**:
- Co-adaptation of generator and OCR
- Final boost in recognition accuracy
- Optimized for specific use case

**Run standalone**:
```bash
python scripts/train_progressive.py \
    --stage 3 \
    --resume checkpoints/stage2.pth \
    --config configs/lp_asrn.yaml
```

### Full Progressive Training

Run all stages sequentially:

```bash
python scripts/train_progressive.py \
    --stage all \
    --config configs/lp_asrn.yaml \
    --data-root data/train
```

TensorBoard will automatically start on port 6007.

---

## Monitoring with TensorBoard

### Starting TensorBoard

TensorBoard starts automatically when training begins. Access at:
```
http://localhost:6007
```

Or start manually:
```bash
tensorboard --logdir logs/tensorboard --port 6007
```

### Metrics Dashboard

#### Scalars

| Metric | Description | Location |
|--------|-------------|----------|
| `train/loss` | Total training loss | Train |
| `train/l1` | L1 reconstruction loss | Train |
| `train/lcofl` | LCOFL character loss | Train |
| `val/psnr` | Peak Signal-to-Noise Ratio | Validation |
| `val/ssim` | Structural Similarity Index | Validation |
| `val/char_acc` | Character-level accuracy | Validation |
| `val/word_acc` | Word-level (full plate) accuracy | Validation |
| `learning_rate` | Current learning rate | Training |
| `gradients/total_norm` | Gradient norm (clipping indicator) | Training |

#### Images

- `comparison/epoch_N`: Side-by-side LR | SR | HR comparisons
- Updated every 5 epochs
- Shows reconstruction quality with text labels

#### Histograms

- Weight distributions per layer (every 10 epochs)
- Gradient norms (for detecting issues)

#### Confusion Matrix

- Character-to-character confusion heatmap
- Updated every 5 epochs in Stage 2 and 3
- Helps identify problematic character pairs

### Interpreting Metrics

**Healthy Training**:
- `val/word_acc` steadily increases
- `train/loss` decreases without oscillation
- `gradients/total_norm` < 1.0 (gradient clipping working)

**Warning Signs**:
- `val/word_acc` plateaus → Consider reducing LR or moving to next stage
- `train/loss` oscillates → Enable gradient clipping
- `gradients/total_norm` spikes → Reduce learning rate

---

## Troubleshooting

### Training Instability

**Symptoms**: Loss oscillation, NaN values, poor convergence

**Solutions**:
1. **Gradient Clipping** (enabled by default):
   ```yaml
   training:
     gradient_clip: 1.0
   ```

2. **Lower Learning Rate**:
   ```bash
   python scripts/train_progressive.py --stage 2 --lr 5e-5
   ```

3. **Extend Warm-up**:
   ```yaml
   progressive_training:
     stage1:
       epochs: 20  # Increase from 10
   ```

### Low Recognition Accuracy

**Symptoms**: High PSNR/SSIM but low word accuracy

**Solutions**:
1. **Verify OCR Fine-tuning**:
   ```bash
   python scripts/evaluate.py --checkpoint checkpoints/parseq/best.pth
   ```

2. **Increase Layout Penalty**:
   ```yaml
   loss:
     lambda_layout: 1.0  # Increase from 0.5
   ```

3. **Train Longer in Stage 2**:
   ```yaml
   progressive_training:
     stage2:
       epochs: 100  # Increase from 50
   ```

### Memory Issues

**Symptoms**: CUDA out of memory errors

**Solutions**:
1. **Reduce Batch Size**:
   ```bash
   python scripts/train_progressive.py --batch-size 8
   ```

2. **Reduce Model Size**:
   ```yaml
   model:
     num_rrdb_blocks: 8  # Reduce from 16
     num_filters: 32     # Reduce from 64
   ```

3. **Enable Gradient Checkpointing**:
   ```python
   # In generator.py, add:
   from torch.utils.checkpoint import checkpoint
   ```

### Poor Character Recognition

**Symptoms**: Specific characters consistently misrecognized

**Solutions**:
1. **Check Confusion Matrix** in TensorBoard
2. **Increase Alpha** (confusion weight increment):
   ```yaml
   loss:
     alpha: 0.2  # Increase from 0.1
   ```
3. **Verify Vocabulary** includes all characters in your dataset

---

## Best Practices

### Hyperparameter Tuning

**Learning Rate**:
- Start with 1e-4 (default)
- Reduce by 0.9 every 5 epochs if no improvement (StepLR)
- Use 1e-5 for Stage 3 (joint training)

**Loss Weights**:
```yaml
loss:
  lambda_layout: 0.5    # Layout penalty (0.2-1.0)
  lambda_ssim: 0.2      # SSIM loss (0.1-0.5)
  alpha: 0.1            # Confusion increment (0.05-0.2)
  beta: 1.0             # Layout penalty value (0.5-2.0)
```

**Batch Size**:
- Larger = more stable gradients (if VRAM allows)
- Default: 16
- Reduce if out of memory

### Checkpointing

Checkpoints are saved to `checkpoints/lp_asrn/`:
- `best.pth`: Best model (highest word accuracy)
- `stage_{name}_epoch_{N}.pth`: Per-epoch checkpoints

**Resume Training**:
```bash
python scripts/train_progressive.py \
    --stage 2 \
    --resume checkpoints/lp_asrn/best.pth
```

### Evaluation

Evaluate on test set:
```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/lp_asrn/best.pth \
    --save-dir results/evaluation
```

Metrics reported:
- PSNR (dB)
- SSIM (0-1)
- Character Accuracy (%)
- Word Accuracy (%)

### Common Mistakes

1. **Skipping OCR Fine-tuning**: Results in poor LCOFL gradients
2. **Skipping Warm-up Stage**: Can cause early instability
3. **Unfreezing OCR Too Early**: Wait until Stage 3
4. **Monitoring Loss Instead of Recognition**: LCOFL optimizes for recognition, not pixel metrics
5. **Using 4x Upscaling**: Paper 2 shows 2x achieves better recognition (49.8% vs 39.0%)

---

## Training Workflow Summary

```bash
# 1. Fine-tune OCR on HR images
python scripts/finetune_parseq.py --epochs 10

# 2. Run progressive training (all stages)
python scripts/train_progressive.py --stage all

# 3. Monitor in TensorBoard
# Open http://localhost:6007

# 4. Evaluate final model
python scripts/evaluate.py --checkpoint checkpoints/lp_asrn/best.pth
```

---

## References

- [Paper 1 (2023)](https://arxiv.org/abs/2305.17313): Super-Resolution of License Plate Images Using Attention Modules
- [Paper 2 (2024)](https://arxiv.org/abs/2408.15103): Enhancing License Plate Super-Resolution
- [TensorBoard PyTorch Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
