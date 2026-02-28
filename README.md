# LP-ASRN

**Layout-Aware & Character-Driven Super-Resolution for License Plates**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A PyTorch implementation of LP-ASRN for license plate super-resolution using **RRDB-EA** (Residual-in-Residual Dense Block with Enhanced Attention) generator, **PARSeq OCR**, and progressive character-driven optimization.

## What's New in v3.1

| Feature | Description | Benefit |
|---------|-------------|---------|
| **PSNR Guardrail** | Dynamic LCOFL weight scaling when PSNR drops below floor | Prevents visual quality collapse |
| **Balanced Checkpoint** | `best_balanced.pth` combining word_acc × min(psnr/13, 1) | Best accuracy-quality tradeoff |
| **Standalone SSIM Loss** | Separate SSIM loss component in Stages 2-3 | Visual quality preservation |
| **Dual-Format Data** | PNG + JPG loading (Scenario-A + Scenario-B) | 2× training data (100K samples) |
| **Augmentation Pipeline** | JPEG compression, test-resolution, no-crop, aspect ratio | Bridges train-test domain gap |
| **Multi-Loss Stage 3** | L1 + LCOFL + SSIM + gradient + frequency + edge | Better convergence |
| **Diagnostic Inference** | `--diagnose`, `--multi-scale`, `--tta`, `--jpeg-deblock` | Detailed analysis tools |

## v3.0 Foundation

| Feature | Description | Benefit |
|---------|-------------|---------|
| **RRDB-EA Architecture** | Dense blocks + Enhanced Attention + deformable convolutions | Strong local feature extraction |
| **PARSeq OCR** | Pretrained attention-based OCR from HuggingFace | State-of-the-art recognition |
| **Five-Stage Progressive** | OCR Pretrain → Warm-up → LCOFL → Fine-tune → Hard Mining | Stable convergence |
| **~3.99M Parameters** | Efficient generator for fast inference | Accuracy with low footprint |

---

## Overview

LP-ASRN addresses the challenge of recognizing license plates from low-resolution surveillance footage using a task-specific super-resolution approach that maximizes recognition accuracy through character-level optimization.

### Key Features

- **Five-Stage Progressive Training**: OCR Pretrain → Warm-up → LCOFL → Fine-tune → Hard Mining
- **RRDB-EA Generator**: Dense blocks with Enhanced Attention Module + deformable convolutions (~3.99M params)
- **PARSeq OCR**: Pretrained attention-based text recognition (~51M params, frozen)
- **LCOFL Loss**: Classification + layout penalty with dynamic character confusion weights
- **Quality Guardrails**: PSNR floor, balanced checkpoint, standalone SSIM loss
- **Dual-Format Data**: Supports both PNG (Scenario-A) and JPG (Scenario-B) images
- **TensorBoard Integration**: Real-time visualization of metrics and images

---

## Results

| Method | Dataset | Word Accuracy |
|--------|---------|---------------|
| Paper 1 (2023) | RodoSol-ALPR | 39.0% |
| Paper 2 (2024) | RodoSol-ALPR | 49.8% |
| **LP-ASRN v3.1** | RodoSol-ALPR | **>60%** (target) |

---

## Installation

```bash
# Clone and install
git clone <repository-url>
cd LP-ASRN
pip install -r requirements.txt
```

---

## Quick Start

```bash
# Train all stages (TensorBoard auto-starts on :6007)
python scripts/train_progressive.py --config configs/lp_asrn.yaml

# Inference with best checkpoint
python scripts/inference.py --checkpoint outputs/run_XXXXX/best_balanced.pth --data-root data/test-public

# Inference with enhancements
python scripts/inference.py --checkpoint outputs/run_XXXXX/best_balanced.pth \
    --multi-scale --tta --jpeg-deblock --preserve-aspect
```

---

## Training Stages

| Stage | Name | Purpose | Epochs | Loss | OCR |
|-------|------|---------|--------|------|-----|
| 0 | OCR Pretrain | Fine-tune PARSeq on HR images | 50 | PARSeq PLM | Training |
| 1 | Warm-up | Stabilize RRDB-EA with L1 loss | 80 | L1 | Frozen |
| 2 | LCOFL | Character-driven + PSNR guardrail | 200 | L1+LCOFL+SSIM | Frozen |
| 3 | Fine-tune | Multi-loss extended optimization | 200 | L1+LCOFL+SSIM+Grad+Freq+Edge | Frozen |
| 4 | Hard Mining | Focus on difficult samples | 50 | L1+LCOFL | Frozen |

Train individual stages:
```bash
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 0  # OCR Pretrain
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 1  # Warm-up
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 2  # LCOFL
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 3  # Fine-tune
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 4  # Hard Mining
```

---

## Architecture

```
LR Image --> [RRDB-EA Generator] --> SR Image (2x)
               |
               +-- Shallow Feature Extractor (PixelUnshuffle autoencoder)
               +-- 12x RRDB-EA Blocks
               |   +-- Dense Connections (3 layers per block)
               |   +-- Enhanced Attention Module
               |   +-- Deformable Convolutions (optional)
               +-- Global Residual Connection
               +-- Upscaling Module (PixelShuffle 2x)
               +-- Reconstruction Layer (Conv3x3 + skip, weight=0.2)
```

### RRDB-EA Generator (~3.99M params)

**RRDB-EA (Residual-in-Residual Dense Block with Enhanced Attention)** for license plate super-resolution:
- **Dense Connections**: Rich feature reuse within each block
- **Enhanced Attention Module**: Channel + spatial attention per block
- **Deformable Convolutions**: Adaptive receptive field for geometric variation
- **PixelUnshuffle Autoencoder**: Efficient shallow feature extraction

> A SwinIR Transformer variant exists as backup (`src/models/generator_swinir_backup.py`). The inference script auto-detects architecture from checkpoint keys.

### OCR Model (PARSeq, ~51M params)

**Pretrained attention-based OCR** from HuggingFace:
- ViT encoder + autoregressive decoder
- Permutation Language Modeling (PLM) with 6 permutations
- 36-character LP vocabulary mapped from native 97-char PARSeq vocab
- Preprocessing: resize to 32x128 + ImageNet normalization
- PlateFormatValidator for Brazilian/Mercosur post-processing

---

## Configuration

Key settings in `configs/lp_asrn.yaml`:

```yaml
model:
  # RRDB-EA Architecture
  num_features: 64              # Feature channels
  num_blocks: 12                # Number of RRDB-EA blocks
  num_layers_per_block: 3       # Dense layers per block
  use_enhanced_attention: true  # Enhanced Attention Module
  use_deformable: true          # Deformable convolutions
  upscale_factor: 2             # 2x super-resolution

ocr:
  model_type: "parseq"          # PARSeq pretrained OCR
  pretrained_path: "baudm/parseq-base"
  freeze_ocr: true              # Frozen during SR training
  max_length: 7
  vocab: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

loss:
  lambda_lcofl: 0.5            # LCOFL weight (reduced from 1.5 to prevent PSNR collapse)
  lambda_layout: 0.5           # Layout penalty
  lambda_ssim: 0.2             # Standalone SSIM loss
  lambda_gradient: 0.05        # Gradient loss
  lambda_frequency: 0.05       # Frequency loss
  lambda_edge: 0.05            # Edge loss

data:
  jpeg_augment: true           # JPEG compression artifacts
  test_resolution_augment: true # Match test-public resolution
  no_crop_prob: 0.3            # Skip corner cropping
  aspect_ratio_augment: true   # Vary aspect ratios

progressive_training:
  stage2:
    psnr_floor: 12.5           # PSNR guardrail for LCOFL
  stage4:
    hard_mining:
      difficulty_alpha: 2.0     # Hard example weighting
```

---

## Inference

```bash
# Standard inference
python scripts/inference.py --checkpoint outputs/run_XXXXX/best.pth --data-root data/test-public

# With enhancements
python scripts/inference.py --checkpoint outputs/run_XXXXX/best_balanced.pth \
    --multi-scale --tta --jpeg-deblock --preserve-aspect

# Diagnostic mode
python scripts/inference.py --checkpoint outputs/run_XXXXX/best.pth --diagnose

# OCR-only (skip super-resolution)
python scripts/inference.py --checkpoint outputs/run_XXXXX/best.pth --ocr-only
```

| Argument | Description |
|----------|-------------|
| `--checkpoint` | Path to model checkpoint (.pth) |
| `--data-root` | Test data directory |
| `--multi-scale` | Multi-scale inference (0.8x, 1.0x, 1.2x) |
| `--tta` | Test-time augmentation |
| `--jpeg-deblock` | Gaussian deblocking for JPEGs |
| `--preserve-aspect` | Pad to preserve aspect ratio |
| `--beam-width` | Beam width for OCR decoding (default: 5) |
| `--diagnose` | Detailed per-track diagnostic output |
| `--diagnose-val` | Validation diagnostic with training data |
| `--ocr-only` / `--no-sr` | Skip SR, run OCR directly on LR |

---

## Project Structure

```
LP-ASRN/
+-- configs/lp_asrn.yaml               # Training configuration
+-- src/
|   +-- models/
|   |   +-- generator.py               # RRDB-EA Generator
|   |   +-- generator_swinir_backup.py # SwinIR backup variant
|   |   +-- swinir_blocks.py           # SwinIR building blocks
|   |   +-- attention.py               # Enhanced Attention Module
|   |   +-- character_attention.py     # MultiScale Character Attention
|   |   +-- deform_conv.py             # Deformable convolutions
|   |   +-- siamese_embedder.py        # Siamese network (disabled)
|   +-- ocr/
|   |   +-- ocr_model.py              # PARSeq OCR wrapper + PlateFormatValidator
|   |   +-- confusion_tracker.py       # Character confusion tracking
|   +-- losses/
|   |   +-- lcofl.py                   # LCOFL loss (classification + layout + SSIM)
|   |   +-- basic.py                   # L1, Perceptual, Gradient, Frequency, Edge losses
|   |   +-- embedding_loss.py          # Embedding consistency loss
|   |   +-- gan_loss.py                # GAN + feature matching loss (optional)
|   +-- training/
|   |   +-- progressive_trainer.py     # 5-stage progressive trainer
|   |   +-- hard_example_miner.py      # Hard example mining for Stage 4
|   +-- utils/
|   |   +-- logger.py                  # TensorBoard logger
|   |   +-- visualizer.py             # Image comparison grids
|   |   +-- adaptive_scheduler.py      # Adaptive LR scheduling
|   +-- data/
|       +-- lp_dataset.py              # License plate dataset (PNG + JPG)
+-- scripts/
|   +-- train_progressive.py           # Main training entry point
|   +-- inference.py                   # Inference + submission generation
|   +-- evaluate.py                    # Evaluation script
|   +-- finetune_ocr.py                # OCR fine-tuning
|   +-- test_pipeline.py               # Comprehensive pipeline tests
|   +-- analyze_data_mismatch.py       # Data analysis tools
+-- docs/
|   +-- architecture.md                # Detailed architecture docs
|   +-- training.md                    # Training guide
|   +-- CHANGES.md                     # Changelog
|   +-- CONTRIBUTING.md                # Contributing guidelines
+-- AGENTS.md                          # Training agents documentation
```

---

## TensorBoard

Access at `http://localhost:6007` during training:

- **Scalars**: Loss, PSNR, SSIM, word/char accuracy, LCOFL scale
- **Images**: LR | SR | HR comparisons
- **Stage 2 Metrics**: LCOFL scale factor, balanced score
- **Stage 4 Metrics**: Hard mining statistics

---

## Citation

```bibtex
@article{nascimento2024enhancing,
  title={Enhancing License Plate Super-Resolution: A Layout-Aware and Character-Driven Approach},
  author={Nascimento, Valfride and Laroca, Rayson and others},
  journal={arXiv preprint arXiv:2408.15103},
  year={2024}
}
```

---

## References

- [Paper 1 (2023)](https://arxiv.org/abs/2305.17313): Attention-based LP Super-Resolution
- [Paper 2 (2024)](https://arxiv.org/abs/2408.15103): Layout-Aware LP Super-Resolution
- [PARSeq (2022)](https://arxiv.org/abs/2207.06966): Pre-training Autoregressive Objectively

## License

MIT License - see LICENSE file for details.
