# LP-ASRN Training Guide

Complete guide for training the Layout-Aware and Character-Driven Super-Resolution Network with RRDB-EA and PARSeq.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Data Preparation](#data-preparation)
3. [Progressive Training](#progressive-training)
4. [Quality Guardrails (v3.1)](#quality-guardrails-v31)
5. [Inference](#inference)
6. [Monitoring with TensorBoard](#monitoring-with-tensorboard)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Prerequisites

### Hardware Requirements

- **GPU**: NVIDIA GPU with 8GB+ VRAM (12GB+ recommended for batch_size=64)
- **RAM**: 16GB+ system memory
- **Storage**: 10GB+ for datasets and checkpoints
- **Recommended**: 2x NVIDIA A100-SXM4-80GB for full training

### Software Requirements

```bash
# Core dependencies
python >= 3.8
torch >= 2.0
torchvision >= 0.15
cuda >= 11.7

# OCR
timm >= 0.9.0           # For PARSeq model loading

# Image processing
Pillow >= 10.0.0
opencv-python >= 4.8.0

# Utilities
PyYAML >= 6.0
tqdm >= 4.65.0
tensorboard >= 2.13.0
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
+-- Scenario-A/               # Light degradation (PNG images)
|   +-- Brazilian/             # LLLNNNN layout
|   |   +-- track_00001/
|   |   |   +-- annotations.json
|   |   |   +-- lr-001.png ... lr-005.png
|   |   |   +-- hr-001.png ... hr-005.png
|   |   +-- track_00002/
|   |   +-- ...
|   +-- Mercosur/              # LLLNLNN layout
|       +-- track_10001/
|       +-- ...
+-- Scenario-B/                # Heavy degradation (JPG images)
    +-- Brazilian/
    |   +-- track_20001/
    |   |   +-- annotations.json
    |   |   +-- lr-001.jpg ... lr-005.jpg
    |   |   +-- hr-001.jpg ... hr-005.jpg
    |   +-- ...
    +-- Mercosur/
        +-- ...
```

### Dual-Format Support (v3.1)

The dataset loader automatically handles both PNG and JPG images:
- Tries `lr-{i:03d}.png` first, falls back to `lr-{i:03d}.jpg`
- Same logic for HR images
- 5 image pairs per track x 20,000 tracks = ~100,000 total samples

### Annotation Format

Each track folder contains `annotations.json`:
```json
{
  "plate_text": "ABC1234",
  "plate_layout": "Brazilian",
  "corners": [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
}
```

---

## Progressive Training

LP-ASRN uses a **five-stage** progressive training approach optimized for RRDB-EA + PARSeq.

### Training Stages Overview

| Stage | Name | Epochs | LR | Loss | OCR | Special |
|-------|------|--------|-----|------|-----|---------|
| 0 | Pretrain | 50 | 5e-4 | PARSeq PLM | Training | Fine-tune on LP data |
| 1 | Warm-up | 80 | 1e-4 | L1 | Frozen | Stabilize generator |
| 2 | LCOFL | 200 | 1e-4 | L1+LCOFL+SSIM | Frozen | PSNR guardrail |
| 3 | Fine-tune | 200 | 1e-5 | L1+LCOFL+SSIM+Grad+Freq+Edge | Frozen | Narrow aspect range |
| 4 | Hard Mining | 50 | 5e-6 | L1+LCOFL | Frozen | Weighted sampling |

### Stage 0: PARSeq Pretraining

Fine-tune PARSeq on high-resolution license plate images:
```bash
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 0
```

**What happens:**
- Loads pretrained PARSeq from HuggingFace (`baudm/parseq-base`)
- Fine-tunes with Permutation Language Modeling (PLM)
- Trains on both Scenario-A (PNG) and Scenario-B (JPG) HR images
- Saves to `checkpoints/ocr/best.pth`

### Stage 1: Warm-up

Stabilize RRDB-EA generator with simple L1 loss:
```bash
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 1
```

**What happens:**
- Creates RRDB-EA generator (~3.99M parameters)
- Trains with L1 reconstruction loss only
- OCR is frozen and used for monitoring only

### Stage 2: LCOFL Training

Character-driven optimization with frozen PARSeq and quality guardrails:
```bash
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 2
```

**What happens:**
- L1 + LCOFL + standalone SSIM loss
- Character confusion weights updated each epoch
- Layout penalty enforced (digit/letter positions)
- **PSNR guardrail**: auto-scales LCOFL weight if PSNR drops below 12.5
- **Balanced checkpoint**: saves `best_balanced.pth` with best `word_acc * min(psnr/13, 1)` score
- OCR remains frozen

### Stage 3: Fine-tuning

Extended optimization with multi-loss supervision:
```bash
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 3
```

**What happens:**
- Six loss components: L1 + LCOFL + SSIM + gradient + frequency + edge
- Lower learning rate (1e-5) prevents destabilization
- OCR remains **frozen** for stability
- Aspect ratio range narrowed to [0.25, 0.45] matching test distribution

### Stage 4: Hard Example Mining

Focus on samples OCR struggles with:
```bash
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 4
```

**What happens:**
- Tracks per-sample OCR accuracy
- Weighted sampling (harder samples = higher weight)
- OCR frozen for stability
- `difficulty_alpha: 2.0` controls weighting strength

### Full Training

Run all stages sequentially:
```bash
python scripts/train_progressive.py --config configs/lp_asrn.yaml
```

---

## Quality Guardrails (v3.1)

### Problem: LCOFL Quality Collapse

With aggressive LCOFL training (lambda_lcofl=1.5), PSNR collapsed from 13.88 to 12.0. The generator learned to produce artifacts that fooled OCR on training data but generalized poorly.

### Solution 1: PSNR Guardrail

Dynamically scales LCOFL weight when PSNR drops below a floor:

```python
if val_psnr < psnr_floor:      # psnr_floor = 12.5
    lcofl_scale = max(0.1, val_psnr / psnr_floor)
    effective_weight = lambda_lcofl * lcofl_scale
```

Configured in Stage 2:
```yaml
progressive_training:
  stage2:
    psnr_floor: 12.5
```

### Solution 2: Balanced Checkpoint

Saves best model combining accuracy AND visual quality:

```python
balanced_score = word_acc * min(val_psnr / 13.0, 1.0)
```

Output: `best_balanced.pth` — use for submission instead of `best.pth` if PSNR collapsed.

### Solution 3: Standalone SSIM Loss

Separate SSIM loss component (independent from SSIM inside LCOFL):

```yaml
progressive_training:
  stage2:
    loss_components: ["l1", "lcofl", "ssim"]  # "ssim" = standalone
loss:
  lambda_ssim: 0.2
```

### Solution 4: Reduced LCOFL Weight

Reduced `lambda_lcofl` from 1.5 to 0.5:

```yaml
loss:
  lambda_lcofl: 0.5  # Was 1.5, caused PSNR collapse
```

---

## Inference

### Standard Inference

```bash
python scripts/inference.py --checkpoint outputs/run_XXXXX/best_balanced.pth --data-root data/test-public
```

### Enhanced Inference

```bash
python scripts/inference.py --checkpoint outputs/run_XXXXX/best_balanced.pth \
    --multi-scale --tta --jpeg-deblock --preserve-aspect
```

### Diagnostic Mode

```bash
# Per-track diagnostic analysis
python scripts/inference.py --checkpoint outputs/run_XXXXX/best.pth --diagnose

# Validation diagnostic (compare with training labels)
python scripts/inference.py --checkpoint outputs/run_XXXXX/best.pth \
    --diagnose-val data/train
```

### OCR-Only Mode

```bash
# Skip super-resolution, run OCR directly on LR images
python scripts/inference.py --checkpoint outputs/run_XXXXX/best.pth --ocr-only
```

### CLI Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--checkpoint` | str | required | Path to model checkpoint (.pth) |
| `--config` | str | `configs/lp_asrn.yaml` | Path to config YAML |
| `--data-root` | str | `data/test-public` | Root directory of test data |
| `--output-dir` | str | `submission` | Output directory |
| `--batch-size` | int | 16 | Batch size for inference |
| `--beam-width` | int | 5 | Beam width for OCR decoding |
| `--multi-scale` | flag | — | Multi-scale inference (0.8x, 1.0x, 1.2x) |
| `--tta` | flag | — | Test-time augmentation |
| `--jpeg-deblock` | flag | — | Gaussian deblocking for JPEG images |
| `--preserve-aspect` | flag | — | Pad images to preserve aspect ratio |
| `--ocr-only` / `--no-sr` | flag | — | Skip SR, run OCR on LR images |
| `--diagnose` | flag | — | Detailed per-track diagnostic output |
| `--diagnose-val` | str | — | Training data root for validation comparison |

### Output

- `submission/predictions.txt`: `track_id plate_text;confidence`
- `submission/submission.zip`: Ready for CodaBench upload
- Auto-detects RRDB-EA vs SwinIR architecture from checkpoint keys

---

## Monitoring with TensorBoard

### Starting TensorBoard

Starts automatically with training. Access at:
```
http://localhost:6007
```

Or start manually:
```bash
tensorboard --logdir outputs/run_XXXXX/logs --port 6007
```

### Key Metrics

| Metric | Description |
|--------|-------------|
| `train/loss` | Total training loss |
| `train/l1_loss` | L1 reconstruction loss |
| `train/lcofl_loss` | LCOFL character-driven loss |
| `train/ssim_loss` | Standalone SSIM loss |
| `val/word_acc` | Word-level accuracy (primary metric) |
| `val/char_acc` | Character-level accuracy |
| `val/psnr` | Peak Signal-to-Noise Ratio |
| `val/ssim` | Structural Similarity Index |
| `stage2/lcofl_scale` | PSNR guardrail scaling factor |
| `stage2/balanced_score` | Balanced checkpoint score |

### Image Visualization

TensorBoard displays:
- LR images (input)
- SR images (super-resolved output)
- HR images (ground truth)
- Updated every N epochs (configurable via `log_images_every`)

---

## Configuration

### Active Configuration (RRDB-EA)

```yaml
model:
  num_features: 64              # Feature channels
  num_blocks: 12                # Number of RRDB-EA blocks
  num_layers_per_block: 3       # Dense layers per block
  use_enhanced_attention: true  # Enhanced Attention Module
  use_deformable: true          # Deformable convolutions
  upscale_factor: 2             # 2x super-resolution
  use_character_attention: false

loss:
  lambda_lcofl: 0.5            # LCOFL weight
  lambda_layout: 0.5           # Layout penalty
  lambda_ssim: 0.2             # Standalone SSIM
  lambda_gradient: 0.05        # Gradient loss (Stage 3)
  lambda_frequency: 0.05       # Frequency loss (Stage 3)
  lambda_edge: 0.05            # Edge loss (Stage 3)

data:
  scenarios: ["Scenario-A", "Scenario-B"]
  layouts: ["Brazilian", "Mercosur"]
  batch_size: 64
  val_split: 0.10
  jpeg_augment: true
  jpeg_quality_range: [60, 95]
  test_resolution_augment: true
  test_resolution_prob: 0.7
  no_crop_prob: 0.3
  aspect_ratio_augment: true
  test_aspect_range: [0.25, 0.45]
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

---

## Troubleshooting

### Low Recognition Accuracy

1. **Check balanced checkpoint** (`best_balanced.pth`) vs `best.pth`
2. **Run Stage 4** (Hard Example Mining):
   ```bash
   python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 4
   ```
3. **Reduce LCOFL weight** if PSNR is collapsing:
   ```yaml
   loss:
     lambda_lcofl: 0.3
   ```
4. **Enable test-resolution augmentation**:
   ```yaml
   data:
     test_resolution_augment: true
     test_resolution_prob: 0.7
   ```

### PSNR Collapse During LCOFL (Stage 2)

1. Reduce `lambda_lcofl` (current: 0.5, was 1.5 before fix)
2. Add standalone SSIM: include `"ssim"` in `loss_components`
3. Set `psnr_floor: 12.5` to auto-scale LCOFL weight
4. Use `best_balanced.pth` checkpoint for submission

### Slow Training

1. **Reduce model complexity**:
   ```yaml
   model:
     num_features: 48
     num_blocks: 8
     use_deformable: false
   ```
2. **Reduce batch size** (if CPU-bound on data loading):
   ```yaml
   data:
     batch_size: 32
     num_workers: 8
   ```

### Memory Issues

1. Reduce batch size: `batch_size: 32`
2. Reduce feature channels: `num_features: 48`
3. Disable deformable convolutions: `use_deformable: false`

### Train-Test Domain Gap

1. Enable JPEG augmentation: `jpeg_augment: true`
2. Enable test-resolution augmentation: `test_resolution_augment: true`
3. Set no-crop probability: `no_crop_prob: 0.3`
4. Narrow aspect ratio range to match test: `test_aspect_range: [0.25, 0.45]`

---

## Best Practices

### Recommended Configuration

```yaml
# Optimal settings for word accuracy (current active config)
model:
  num_features: 64
  num_blocks: 12
  num_layers_per_block: 3
  use_enhanced_attention: true
  use_deformable: true
  upscale_factor: 2

loss:
  lambda_lcofl: 0.5
  lambda_layout: 0.5
  lambda_ssim: 0.2

progressive_training:
  stage2:
    psnr_floor: 12.5
```

### Training Workflow

```bash
# 1. Full progressive training
python scripts/train_progressive.py --config configs/lp_asrn.yaml

# 2. Monitor in TensorBoard
# http://localhost:6007

# 3. Inference with best model
python scripts/inference.py --checkpoint outputs/run_XXXXX/best_balanced.pth \
    --data-root data/test-public --multi-scale --tta
```

### Tips for Best Results

1. **Always start with Stage 0** to fine-tune PARSeq on your LP data
2. **Use balanced checkpoint** (`best_balanced.pth`) for submission
3. **Monitor PSNR during Stage 2** — if dropping below 12.5, guardrail activates
4. **Run Stage 4** for hard example mining after Stages 2-3
5. **Enable all augmentations** to bridge train-test domain gap
6. **Check diagnostic mode** (`--diagnose`) to analyze per-track failures
7. **Use multi-scale + TTA** for best inference accuracy

---

## References

- [Paper 1 (2023)](https://arxiv.org/abs/2305.17313): Attention-based LP Super-Resolution
- [Paper 2 (2024)](https://arxiv.org/abs/2408.15103): Layout-Aware LP Super-Resolution
- [PARSeq (2022)](https://arxiv.org/abs/2207.06966): Pre-training Autoregressive Objectively
