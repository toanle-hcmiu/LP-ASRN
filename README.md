# LP-ASRN

**Layout-Aware & Character-Driven Super-Resolution for License Plates**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A PyTorch implementation of the LP-ASRN architecture from Nascimento et al. (2024) for license plate super-resolution, featuring progressive training, TensorBoard integration, and character-driven optimization.

## Overview

LP-ASRN addresses the challenge of recognizing license plates from low-resolution surveillance footage by using a task-specific super-resolution approach. Unlike generic super-resolution methods that optimize for pixel-level metrics (PSNR, SSIM), LP-ASRN is explicitly designed to maximize license plate recognition accuracy.

### Key Features

- **Progressive Three-Stage Training**: Warm-up → LCOFL → Fine-tuning for stability
- **Layout and Character Oriented Focal Loss (LCOFL)**: Penalizes character confusion and layout violations
- **Enhanced Attention Module**: Deformable convolutions for adaptive character feature extraction
- **TensorBoard Integration**: Real-time visualization of metrics, images, and confusion matrices
- **Parseq OCR**: State-of-the-art scene text recognition, fine-tuned on license plates
- **Single Unified Model**: Handles both Brazilian (LLLNNNN) and Mercosur (LLLNLNN) layouts

## Results

Based on the original papers:

| Method | Dataset | Word Accuracy |
|--------|--------|---------------|
| Paper 1 (2023) | RodoSol-ALPR | 39.0% |
| Paper 2 (2024) | RodoSol-ALPR | **49.8%** |
| This implementation | LP-ASRN | TBD |

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA (for GPU training)

```bash
# Clone repository
git clone <repository-url>
cd LP-ASRN

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

```bash
# 1. Fine-tune Parseq on high-resolution license plates
python scripts/finetune_parseq.py --epochs 10 --batch-size 32

# 2. Train with progressive training (TensorBoard auto-starts on port 6007)
python scripts/train_progressive.py --stage all --config configs/lacd_srnn.yaml

# 3. Evaluate
python scripts/evaluate.py --checkpoint checkpoints/lacd_srnn/best.pth
```

## Usage

### Fine-tuning OCR

Before training the super-resolution model, fine-tune Parseq on your high-resolution license plate images:

```bash
python scripts/finetune_parseq.py \
    --data-root data/train \
    --epochs 10 \
    --batch-size 32 \
    --save-dir checkpoints/parseq
```

### Progressive Training

The progressive training approach consists of three stages:

```bash
python scripts/train_progressive.py \
    --config configs/lacd_srnn.yaml \
    --stage all \
    --tb-port 6007
```

Or train individual stages:

```bash
# Stage 1: Warm-up (L1 loss only)
python scripts/train_progressive.py --stage 1 --epochs 10

# Stage 2: LCOFL training
python scripts/train_progressive.py --stage 2 --resume checkpoints/stage1.pth

# Stage 3: Fine-tuning (joint OCR optimization)
python scripts/train_progressive.py --stage 3 --resume checkpoints/stage2.pth
```

### Training Stages

1. **Stage 1: Warm-up**
   - Loss: L1 (pixel reconstruction)
   - Purpose: Stabilize network before introducing complex losses
   - Duration: 5-10 epochs

2. **Stage 2: LCOFL Training**
   - Loss: L1 + LCOFL (character-driven)
   - Purpose: Optimize for character recognition
   - Duration: 50+ epochs

3. **Stage 3: Fine-tuning**
   - Loss: L1 + LCOFL (joint optimization)
   - Purpose: Refine with unfrozen OCR
   - Duration: 20+ epochs

### TensorBoard

TensorBoard automatically starts when training begins. Access at:
```
http://localhost:6007
```

Visualizations:
- **Scalars**: Losses, PSNR, SSIM, accuracy
- **Images**: LR/SR/HR comparisons
- **Histograms**: Weight distributions, gradient norms
- **Text**: Training logs and confusion reports

### Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/lacd_srnn/best.pth \
    --save-dir results/evaluation
```

Metrics reported:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- Character accuracy
- Word accuracy (full plate match)

## Architecture

### Generator Network

The LP-ASRN generator consists of:

1. **Shallow Feature Extractor**: PixelUnshuffle → Conv → PixelShuffle auto-encoder
2. **Deep Feature Extractor**: 16 Residual-in-Residual Dense Blocks with Enhanced Attention
3. **Enhanced Attention Module**:
   - Channel Attention (inter-channel relationships)
   - Geometrical Perception Unit (horizontal/vertical structure)
   - Deformable Convolutions (adaptive receptive fields)
4. **Upscaling Module**: PixelShuffle for 2x upscaling
5. **Reconstruction Layer**: Conv + Tanh activation

### LCOFL Loss

The Layout and Character Oriented Focal Loss consists of:

1. **Classification Loss**: Weighted cross-entropy with adaptive confusion-based weights
2. **LP Layout Penalty**: Penalizes digit/letter position mismatches
3. **SSIM Loss**: Structural similarity constraint

```
L_LCOFL = L_C + λ_layout * L_P + λ_ssim * L_S
```

## Dataset

### LP-ASRN Dataset

The dataset contains paired low-resolution and high-resolution license plate images:

```
data/train/
├── Scenario-A/          # Light degradation
│   ├── Brazilian/       # LLLNNNN layout (5,000 tracks)
│   └── Mercosur/        # LLLNLNN layout (5,000 tracks)
└── Scenario-B/          # Heavy degradation
    ├── Brazilian/       # (2,000 tracks)
    └── Mercosur/        # (8,000 tracks)
```

### Image Dimensions

- **LR**: ~31x17 pixels
- **HR**: ~60x32 pixels
- **Upscaling**: 2x

### Annotations

Each track includes:
- `lr-001.png` to `lr-005.png`: Low-resolution images
- `hr-001.png` to `hr-005.png`: High-resolution images
- `annotations.json`: Plate text, layout, and corner coordinates

## Configuration

Edit `configs/lacd_srnn.yaml` to customize:

```yaml
model:
  num_rrdb_blocks: 16
  num_filters: 64
  upscale_factor: 2
  use_deformable: true

progressive_training:
  stage1:
    epochs: 10
    lr: 0.0001
  stage2:
    epochs: 50
    lr: 0.0001
  stage3:
    epochs: 20
    lr: 0.00001

tensorboard:
  enabled: true
  log_dir: "logs/tensorboard"
```

## Project Structure

```
LP-ASRN/
├── configs/                 # Training configurations
│   └── lacd_srnn.yaml
├── src/
│   ├── data/               # Data loading
│   │   └── lp_dataset.py
│   ├── models/             # Model architectures
│   │   ├── generator.py
│   │   ├── attention.py
│   │   └── deform_conv.py
│   ├── losses/             # Loss functions
│   │   ├── lcofl.py
│   │   └── basic.py
│   ├── ocr/                # OCR integration
│   │   ├── parseq_wrapper.py
│   │   └── confusion_tracker.py
│   ├── training/           # Progressive training
│   │   └── progressive_trainer.py
│   └── utils/              # Utilities
│       ├── logger.py      # TensorBoard logger
│       └── visualizer.py  # Visualization tools
├── scripts/                # Training/evaluation scripts
│   ├── finetune_parseq.py
│   ├── train_progressive.py
│   └── evaluate.py
├── checkpoints/            # Model checkpoints
├── logs/                   # TensorBoard logs
└── results/                # Evaluation results
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{nascimento2024enhancing,
  title={Enhancing License Plate Super-Resolution: A Layout-Aware and Character-Driven Approach},
  author={Nascimento, Valfride and Laroca, Rayson and Ribeiro, Rafael O. and Schwartz, William R. and Menotti, David},
  journal={arXiv preprint arXiv:2408.15103},
  year={2024}
}
```

## References

- [Paper 1 (2023)](https://arxiv.org/abs/2305.17313): Super-Resolution of License Plate Images Using Attention Modules
- [Paper 2 (2024)](https://arxiv.org/abs/2408.15103): Enhancing License Plate Super-Resolution
- [Code Repository Paper 1](https://github.com/valfride/lpr-rsr-ext)
- [Code Repository Paper 2](https://github.com/valfride/lpsr-lacd)

## License

MIT License - see LICENSE file for details.
