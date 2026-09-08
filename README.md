# LP-ASRN

### Layout-Aware & Character-Driven Super-Resolution for License Plate Recognition

**LP-ASRN** is a research project investigating **task-oriented super-resolution for low-resolution license plate recognition (LPR)**.

The project explores whether a super-resolution model can be optimized not only for visual reconstruction quality, but also for the **downstream recognition of license-plate characters**.

The final system was evaluated in the **ICPR 2026 Competition on Low-Resolution License Plate Recognition** on CodaBench.

> **Final Competition Result:** 71.27% Recognition Rate (Rank #51 / 99)

---

## Research Question

Low-resolution license plate images often contain insufficient visual information for reliable character recognition. Conventional super-resolution methods typically optimize image reconstruction objectives such as L1, PSNR, or SSIM.

However, a visually faithful reconstruction does not necessarily produce the most recognizable characters.

This project investigates the following question:

> **Can character-level recognition feedback improve super-resolution for downstream license plate recognition?**

Instead of treating super-resolution and OCR as independent tasks, LP-ASRN incorporates a pretrained OCR model into the training process and optimizes the super-resolution network with recognition-oriented objectives.

---

## Approach

The overall pipeline is:

```text
Low-Resolution License Plate
            │
            ▼
    ┌─────────────────┐
    │   RRDB-EA SR    │
    │     Network     │
    └────────┬────────┘
             │
             ▼
    Super-Resolved Image
             │
             ▼
    ┌─────────────────┐
    │     PARSeq      │
    │       OCR       │
    └────────┬────────┘
             │
             ▼
    Character Recognition
```

The key idea is to use the recognition model as a source of supervision for the super-resolution network.

---

## Main Contributions

### 1. Recognition-oriented Super-Resolution

Rather than optimizing only pixel-level reconstruction, LP-ASRN introduces character-driven objectives that explicitly consider the downstream OCR task.

### 2. RRDB-EA Generator

The super-resolution network is based on an enhanced RRDB architecture incorporating:

* Residual-in-Residual Dense Blocks
* channel and spatial attention
* deformable convolutions
* PixelUnshuffle-based feature extraction
* PixelShuffle-based 2× upsampling
* global residual connections

The final generator contains approximately **3.99M parameters**.

### 3. PARSeq-based Recognition Supervision

A pretrained **PARSeq** scene-text recognition model is adapted to the license plate domain.

During super-resolution training, the OCR model is kept frozen while recognition-oriented gradients are propagated to the super-resolution network.

### 4. Layout-Aware Character-Oriented Feature Loss

The project introduces **LCOFL (Layout-aware Character-Oriented Feature Loss)** to encourage reconstructions that preserve both:

* character-level information;
* spatial/layout relationships between characters.

### 5. Progressive Optimization

The training procedure separates reconstruction learning from recognition-oriented optimization through multiple stages:

```text
OCR Adaptation
      ↓
SR Warm-up
      ↓
Character-driven Optimization
      ↓
Multi-loss Fine-tuning
      ↓
Hard Example Mining
```

---

## Architecture

```text
                         ┌────────────────────┐
                         │      PARSeq OCR     │
                         │      (Frozen)       │
                         └─────────┬──────────┘
                                   │
                           Character Feedback
                                   │
                                   ▼
LR Image ──► Feature Extraction ──► RRDB-EA Blocks
                                      │
                                      ├── Dense Connections
                                      ├── Channel Attention
                                      ├── Spatial Attention
                                      └── Deformable Conv.
                                      │
                                      ▼
                              Feature Reconstruction
                                      │
                                      ▼
                               PixelShuffle ×2
                                      │
                                      ▼
                                  SR Image
```

### RRDB-EA

The generator consists of:

* 12 RRDB-EA blocks
* 64 feature channels
* 3 dense layers per block
* enhanced attention
* optional deformable convolutions
* 2× upsampling

The architecture is designed to preserve fine-grained structures that are particularly important for character recognition.

---

## Character-Driven Optimization

The training objective combines reconstruction and recognition-oriented losses.

Conceptually:

```text
L_total =
    λ1 L_reconstruction
  + λ2 L_character
  + λ3 L_layout
  + λ4 L_SSIM
  + λ5 L_gradient
  + λ6 L_frequency
  + λ7 L_edge
```

The different objectives provide complementary constraints:

| Loss           | Purpose                                     |
| -------------- | ------------------------------------------- |
| L1             | Pixel-level reconstruction                  |
| Character loss | Preserve recognizable character information |
| Layout loss    | Preserve character arrangement              |
| SSIM           | Structural similarity                       |
| Gradient loss  | Preserve local structures                   |
| Frequency loss | Preserve high-frequency information         |
| Edge loss      | Preserve character boundaries               |

The main research interest is not simply maximizing image quality, but understanding how these objectives affect **recognition performance**.

---

## Progressive Training

LP-ASRN uses a five-stage training procedure.

| Stage | Objective                     | OCR    |
| ----- | ----------------------------- | ------ |
| 0     | OCR adaptation                | Train  |
| 1     | SR reconstruction warm-up     | Frozen |
| 2     | Character-driven optimization | Frozen |
| 3     | Multi-loss fine-tuning        | Frozen |
| 4     | Hard example mining           | Frozen |

### Stage 0 — OCR Adaptation

PARSeq is adapted to the license plate recognition domain.

### Stage 1 — SR Warm-up

The super-resolution network is first trained using reconstruction loss to establish a stable image reconstruction baseline.

### Stage 2 — Character-driven Optimization

LCOFL and recognition-oriented objectives are introduced.

### Stage 3 — Multi-loss Fine-tuning

Additional structural and high-frequency objectives are introduced:

```text
L1
+ LCOFL
+ SSIM
+ Gradient
+ Frequency
+ Edge
```

### Stage 4 — Hard Example Mining

Samples that remain difficult for recognition are emphasized during further optimization.

---

## Experimental Evaluation

The project evaluates the system from two complementary perspectives.

### Image Reconstruction

* PSNR
* SSIM

### License Plate Recognition

* Character Accuracy
* Word Accuracy

The recognition metrics are particularly important because the ultimate objective of the system is **license plate recognition**, rather than image reconstruction alone.

---

## Baseline Comparison

The project is motivated by previous work on license plate super-resolution.

| Method                                              | Year |      Word Accuracy |
| --------------------------------------------------- | ---: | -----------------: |
| Attention-based LP Super-Resolution                 | 2023 |              39.0% |
| Layout-Aware & Character-Driven LP Super-Resolution | 2024 |              49.8% |
| **LP-ASRN**                                         | 2026 |              71.2% |

The comparison is intended to provide context for the development of recognition-oriented super-resolution methods.

---

## Ablation Studies

The repository is structured to investigate the contribution of individual components.

### Character-driven supervision

```text
Baseline reconstruction
        ↓
+ Character Loss
        ↓
+ Layout Loss
```

### Reconstruction objectives

```text
L1
↓
L1 + SSIM
↓
L1 + SSIM + Gradient
↓
L1 + SSIM + Gradient + Frequency + Edge
```

### Architecture

```text
RRDB
 ↓
RRDB + Attention
 ↓
RRDB + Deformable Convolution
 ↓
RRDB + Attention + Deformable Convolution
```

### Training strategy

The project also compares:

* direct end-to-end optimization;
* progressive optimization;
* progressive optimization with hard-example mining.

---

## Reproducibility

### Installation

```bash
git clone https://github.com/toanle-hcmiu/LP-ASRN.git
cd LP-ASRN
pip install -r requirements.txt
```

### Configuration

The main experimental configuration is:

```text
configs/lp_asrn.yaml
```

### Training

Run the complete progressive training procedure:

```bash
python scripts/train_progressive.py \
    --config configs/lp_asrn.yaml
```

Individual stages can be executed independently:

```bash
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 0
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 1
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 2
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 3
python scripts/train_progressive.py --config configs/lp_asrn.yaml --stage 4
```

---

## Evaluation

Standard evaluation:

```bash
python scripts/inference.py \
    --checkpoint <checkpoint> \
    --data-root <dataset>
```

OCR-only baseline:

```bash
python scripts/inference.py \
    --checkpoint <checkpoint> \
    --data-root <dataset> \
    --ocr-only
```

Diagnostic inference:

```bash
python scripts/inference.py \
    --checkpoint <checkpoint> \
    --data-root <dataset> \
    --diagnose
```

Additional inference options include:

```text
--multi-scale
--tta
--jpeg-deblock
--preserve-aspect
```

These are primarily intended for experimental analysis and evaluation.

---

## Repository Structure

```text
LP-ASRN/
├── configs/
│   └── lp_asrn.yaml
│
├── src/
│   ├── models/
│   │   ├── generator.py
│   │   ├── generator_swinir_backup.py
│   │   ├── attention.py
│   │   ├── character_attention.py
│   │   └── deform_conv.py
│   │
│   ├── ocr/
│   │   ├── ocr_model.py
│   │   └── confusion_tracker.py
│   │
│   ├── losses/
│   │   ├── lcofl.py
│   │   ├── basic.py
│   │   ├── embedding_loss.py
│   │   └── gan_loss.py
│   │
│   ├── training/
│   │   ├── progressive_trainer.py
│   │   └── hard_example_miner.py
│   │
│   └── data/
│       └── lp_dataset.py
│
├── scripts/
│   ├── train_progressive.py
│   ├── inference.py
│   ├── evaluate.py
│   └── finetune_ocr.py
│
├── docs/
│   ├── architecture.md
│   └── training.md
│
└── requirements.txt
```

---

## Research Context

This work builds upon research in:

* image super-resolution;
* license plate recognition;
* scene text recognition;
* character-driven image restoration;
* task-oriented image reconstruction.

The project particularly builds upon the idea that **the optimal reconstruction target should depend on the downstream task**.

For license plates, this means prioritizing information that allows an OCR system to distinguish visually ambiguous characters.

---

## References

### License Plate Super-Resolution

Nascimento, V., Laroca, R., et al.

*Enhancing License Plate Super-Resolution: A Layout-Aware and Character-Driven Approach.*

arXiv:2408.15103, 2024.

### PARSeq

*Scene Text Recognition with Permuted Autoregressive Sequence Models.*

2022.

