# LP-ASRN

**Layout-Aware & Character-Driven Super-Resolution for License Plates**

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A PyTorch implementation of LP-ASRN for license plate super-resolution with progressive training, TensorBoard integration, and character-driven optimization.

## ✨ What's New in v2.0

| Feature | Description | Benefit |
|---------|-------------|---------|
| **LCOFL-EC** | Embedding consistency loss with Siamese network | +3-5% accuracy |
| **DCNv4** | Flash-attention optimized deformable convs | 3x faster training |
| **MSCA** | Multi-scale character attention | Better text focus |
| **Stage 4** | Hard example mining curriculum | +1-2% accuracy |

---

## Overview

LP-ASRN addresses the challenge of recognizing license plates from low-resolution surveillance footage using a task-specific super-resolution approach that maximizes recognition accuracy.

### Key Features

- **Five-Stage Progressive Training**: OCR Pretrain → Warm-up → LCOFL → Fine-tune → Hard Mining
- **LCOFL-EC Loss**: Layout penalty + character classification + embedding consistency
- **Multi-Scale Character Attention**: Focus on text regions at multiple scales
- **DCNv4 Support**: 3x faster training with optional DCNv4
- **TensorBoard Integration**: Real-time visualization of metrics and images

---

## Results

| Method | Dataset | Word Accuracy |
|--------|---------|---------------|
| Paper 1 (2023) | RodoSol-ALPR | 39.0% |
| Paper 2 (2024) | RodoSol-ALPR | 49.8% |
| **LP-ASRN v2.0** | RodoSol-ALPR | **>55%** (target) |

---

## Installation

```bash
# Clone and install
git clone <repository-url>
cd LP-ASRN
pip install -r requirements.txt

# Optional: DCNv4 for 3x faster training
pip install dcnv4
```

---

## Quick Start

```bash
# Train all stages (TensorBoard auto-starts on :6007)
python scripts/train_progressive.py --stage all

# Evaluate
python scripts/evaluate.py --checkpoint checkpoints/lp_asrn/best.pth
```

---

## Training Stages

| Stage | Name | Purpose | Epochs |
|-------|------|---------|--------|
| 0 | OCR Pretrain | Train OCR on HR images | 50 |
| 1 | Warm-up | Stabilize with L1 loss | 30 |
| 2 | LCOFL | Character-driven training | 300 |
| 3 | Fine-tune | Joint OCR optimization | 150 |
| 4 | Hard Mining | Focus on difficult samples | 50 |

Train individual stages:
```bash
python scripts/train_progressive.py --stage 0  # OCR Pretrain
python scripts/train_progressive.py --stage 1  # Warm-up
python scripts/train_progressive.py --stage 2  # LCOFL
python scripts/train_progressive.py --stage 3  # Fine-tune
python scripts/train_progressive.py --stage 4  # Hard Mining
```

---

## Architecture

```
LR Image → [Generator] → SR Image (2x)
              │
              ├── Shallow Feature Extractor
              ├── 16× RRDB-EA Blocks (with DCNv4)
              ├── Multi-Scale Character Attention (NEW)
              ├── Upscaling Module
              └── Reconstruction Layer
```

### Loss Function (LCOFL-EC)

```
L = L1 + λ_lcofl × L_LCOFL + λ_embed × L_EC

Where:
- L_LCOFL = Classification + Layout Penalty + SSIM
- L_EC = Embedding Consistency (Siamese network)
```

---

## Configuration

Key settings in `configs/lp_asrn.yaml`:

```yaml
model:
  num_rrdb_blocks: 12
  use_dcnv4: true                    # 3x faster training
  use_character_attention: true       # Multi-scale attention

loss:
  lambda_embed: 0.3                  # Embedding consistency
  lambda_layout: 0.5                 # Layout penalty

progressive_training:
  stage4:
    epochs: 50
    hard_mining:
      difficulty_alpha: 2.0
```

---

## Project Structure

```
LP-ASRN/
├── configs/lp_asrn.yaml           # Training configuration
├── src/
│   ├── models/
│   │   ├── generator.py           # Main generator
│   │   ├── character_attention.py # MSCA module (NEW)
│   │   ├── siamese_embedder.py    # Embedding network (NEW)
│   │   └── deform_conv.py         # DCNv4/DCNv3
│   ├── losses/
│   │   ├── lcofl.py               # LCOFL-EC loss
│   │   └── embedding_loss.py      # Embedding loss (NEW)
│   ├── training/
│   │   ├── progressive_trainer.py # 5-stage trainer
│   │   └── hard_example_miner.py  # Stage 4 miner (NEW)
│   └── utils/
│       └── adaptive_scheduler.py  # Weight scheduling (NEW)
├── scripts/
│   ├── train_progressive.py
│   └── evaluate.py
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
```

---

## References

- [Paper 1 (2023)](https://arxiv.org/abs/2305.17313): Attention-based LP Super-Resolution
- [Paper 2 (2024)](https://arxiv.org/abs/2408.15103): Layout-Aware LP Super-Resolution

## License

MIT License - see LICENSE file for details.
