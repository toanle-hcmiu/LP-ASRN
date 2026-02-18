# LP-ASRN

**Layout-Aware & Character-Driven Super-Resolution for License Plates**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A PyTorch implementation of LP-ASRN for license plate super-resolution using **SwinIR Transformer** architecture, **PARSeq OCR**, and progressive character-driven optimization.

## ✨ What's New in v3.0

| Feature | Description | Benefit |
|---------|-------------|---------|
| **SwinIR Architecture** | Transformer-based generator with shifted window attention | Better long-range modeling |
| **Character Pyramid Attention** | Layout-aware multi-scale character attention | +5-10% accuracy |
| **PARSeq OCR** | Pretrained attention-based OCR from HuggingFace | State-of-the-art recognition |
| **12.8M Parameters** | Maximum configuration for best results | Prioritizes accuracy over speed |

---

## Overview

LP-ASRN addresses the challenge of recognizing license plates from low-resolution surveillance footage using a task-specific super-resolution approach that maximizes recognition accuracy through character-level optimization.

### Key Features

- **Five-Stage Progressive Training**: OCR Pretrain → Warm-up → LCOFL → Fine-tune → Hard Mining
- **SwinIR Generator**: Transformer-based architecture with shifted window attention
- **Character Pyramid Attention**: Layout-aware multi-scale character focus
- **PARSeq OCR**: Pretrained attention-based text recognition
- **TensorBoard Integration**: Real-time visualization of metrics and images

---

## Results

| Method | Dataset | Word Accuracy |
|--------|---------|---------------|
| Paper 1 (2023) | RodoSol-ALPR | 39.0% |
| Paper 2 (2024) | RodoSol-ALPR | 49.8% |
| **LP-ASRN v3.0** | RodoSol-ALPR | **>60%** (target) |

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

# Evaluate
python scripts/evaluate.py --checkpoint outputs/lp_asrn/best.pth --data-root data/test
```

---

## Training Stages

| Stage | Name | Purpose | Epochs |
|-------|------|---------|--------|
| 0 | OCR Pretrain | Fine-tune PARSeq on HR images | 50 |
| 1 | Warm-up | Stabilize with L1 loss | 30 |
| 2 | LCOFL | Character-driven training | 200 |
| 3 | Fine-tune | Joint OCR optimization | 100 |
| 4 | Hard Mining | Focus on difficult samples | 50 |

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
LR Image → [SwinIR Generator] → SR Image (2x)
              │
              ├── Shallow Feature Extractor (Conv)
              ├── 8× Residual Swin Transformer Blocks
              │   └── Window-based Multi-head Self Attention
              ├── Character Pyramid Attention (Layout-aware)
              ├── Upscaling Module (PixelShuffle)
              └── Reconstruction Layer
```

### SwinIR Generator

**Transformer-based architecture** for license plate super-resolution:
- **Shifted Window Attention**: Efficient self-attention with linear complexity
- **Hierarchical Representation**: Multi-scale feature extraction
- **Character Pyramid Attention**: Layout-aware character focus
- **Maximum Configuration**: 12.8M parameters for best accuracy

### OCR Model (PARSeq)

**Pretrained attention-based OCR** from HuggingFace:
- Autoregressive decoding with language modeling
- Pretrained on millions of text images
- Fine-tuned on license plate data

---

## Configuration

Key settings in `configs/lp_asrn.yaml`:

```yaml
model:
  # SwinIR Architecture (MAXIMUM for best accuracy)
  swinir_embed_dim: 144         # Embedding dimension
  swinir_num_rstb: 8            # Number of Residual Swin Transformer Blocks
  swinir_num_heads: 8           # Number of attention heads
  swinir_window_size: 6         # Window size for attention
  swinir_num_blocks_per_rstb: 3 # Swin blocks per RSTB
  swinir_mlp_ratio: 6.0         # MLP expansion ratio

  # Character Pyramid Attention
  use_pyramid_attention: true   # Layout-aware character attention
  pyramid_layout: "brazilian"   # "brazilian" or "mercocur"

ocr:
  model_type: "parseq"          # PARSeq pretrained OCR
  pretrained_path: "baudm/parseq-base"
  freeze_ocr: true              # Keep OCR frozen during SR training

progressive_training:
  stage4:
    epochs: 50
    hard_mining:
      difficulty_alpha: 2.0     # Hard example weighting
```

---

## Project Structure

```
LP-ASRN/
├── configs/lp_asrn.yaml           # Training configuration
├── src/
│   ├── models/
│   │   ├── generator.py           # SwinIR Generator
│   │   ├── swinir_blocks.py       # SwinIR building blocks
│   │   ├── character_attention.py # Character Pyramid Attention
│   │   └── attention.py           # Attention modules
│   ├── ocr/
│   │   └── ocr_model.py           # PARSeq OCR wrapper
│   ├── losses/
│   │   └── lcofl.py               # LCOFL loss
│   ├── training/
│   │   └── progressive_trainer.py # 5-stage trainer
│   └── data/
│       └── dataset.py             # License plate dataset
├── scripts/
│   ├── train_progressive.py
│   ├── evaluate.py
│   └── inference.py
└── docs/
    ├── architecture.md
    ├── training.md
    └── CHANGES.md
```

---

## TensorBoard

Access at `http://localhost:6007` during training:

- **Scalars**: Loss, PSNR, SSIM, word/char accuracy
- **Images**: LR | SR | HR comparisons
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

@inproceedings{liang2022swinir,
  title={SwinIR: Image Restoration Using Swin Transformer},
  author={Liang, Jingyun and Cao, Jiezhang and Sun, Guolei and others},
  booktitle={CVPR},
  year={2022}
}
```

---

## References

- [Paper 1 (2023)](https://arxiv.org/abs/2305.17313): Attention-based LP Super-Resolution
- [Paper 2 (2024)](https://arxiv.org/abs/2408.15103): Layout-Aware LP Super-Resolution
- [SwinIR (2022)](https://arxiv.org/abs/2109.15272): Image Restoration Using Swin Transformer
- [PARSeq (2022)](https://arxiv.org/abs/2207.06966): Pre-training Autoregressive Objectively

## License

MIT License - see LICENSE file for details.
