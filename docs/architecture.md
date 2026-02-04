# LP-ASRN Architecture

Detailed architectural documentation for the Layout-Aware and Character-Driven Super-Resolution Network.

## Overview

LP-ASRN (License Plate Super-Resolution Network) is a specialized super-resolution architecture designed specifically for license plate recognition. Unlike generic SR methods that optimize for pixel-level metrics, LP-ASRN incorporates character recognition supervision directly into the training process.

---

## High-Level Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │                    Generator                        │
Input: LR Image     │  ┌────────────┐  ┌────────────┐  ┌──────────────┐  │
   (B,3,H,W)  ───▶  │  │  Shallow   │  │    Deep    │  │    MSCA      │  │
                    │  │  Extractor │─▶│  Extractor │─▶│  (Optional)  │  │
                    │  └────────────┘  └────────────┘  └──────────────┘  │
                    │         │                               │          │
                    │         ▼                               ▼          │
                    │  ┌────────────┐  ┌────────────────────────────┐   │
                    │  │  Upscaler  │◀─│  Multi-Scale Char Attention│   │
                    │  └────────────┘  └────────────────────────────┘   │
                    │         │                                          │
                    │         ▼                                          │
                    │  ┌────────────┐                                    │──▶ SR Output
                    │  │ Reconstruct│                                    │    (B,3,2H,2W)
                    │  └────────────┘                                    │
                    └─────────────────────────────────────────────────────┘
```

---

## Generator Components

### 1. Shallow Feature Extractor

Extracts initial features using an auto-encoder structure with PixelShuffle.

```
LR Input → Conv5x5 → PixelUnshuffle → Conv → PixelShuffle → Conv → Shallow Features
                                                                         ↓
                                                              Skip Connection (+)
```

### 2. Deep Feature Extractor

16 RRDB-EA blocks (Residual-in-Residual Dense Block with Enhanced Attention).

```
Shallow Features → [RRDB-EA × 16] → Global Conv → Deep Features
                           ↓
              Each RRDB-EA block contains:
              - Dense layers with growth connections
              - Enhanced Attention Module (EAM)
              - Deformable Convolutions (DCNv4/DCNv3)
```

### 3. Enhanced Attention Module (EAM)

```
Input Features
      │
      ├──▶ Channel Attention (CA)
      │         - Conv1x1 parallel branches
      │         - PixelUnshuffle → Conv → PixelShuffle
      │
      ├──▶ Spatial/Positional Attention (POS)
      │         - DCNv4 (or DCNv3 fallback)
      │         - Adaptive sampling locations
      │
      └──▶ Geometrical Perception Unit (GP)
                - Global avg pool (H/V directions)
                - Point-wise convolutions

      Output = Sigmoid(CA × POS + GP) × DeformConv(Input)
```

### 4. Multi-Scale Character Attention (MSCA) - NEW

Character-aware attention module that focuses on text regions.

```
Deep Features (B, C, H, W)
      │
      ├──▶ Scale 1.0x ──▶ CharRegionDetector ──▶ GuidedAttention ──┐
      │                                                            │
      ├──▶ Scale 0.5x ──▶ CharRegionDetector ──▶ GuidedAttention ──┼──▶ Fusion ──▶ Enhanced Features
      │                                                            │
      └──▶ Scale 0.25x ──▶ CharRegionDetector ──▶ GuidedAttention ─┘
```

**CharacterRegionDetector**: Learns 36 character prototypes (0-9, A-Z) to identify text regions.

### 5. Upscaling Module

```
Features → Conv(C, 3×r²) → PixelShuffle(r=2) → Upscaled (2× resolution)
```

### 6. Reconstruction Layer

```
Upscaled → Conv3x3 → Tanh → Skip(LR upsampled) → SR Output [-1, 1]
```

---

## Deformable Convolution

### DCNv4 (Preferred) vs DCNv3

| Aspect | DCNv3 | DCNv4 |
|--------|-------|-------|
| Weight normalization | Softmax (bounded) | Unbounded |
| Skip connection | Internal | External |
| Memory access | Standard | Flash-attention optimized |
| Speed | Baseline | **~3x faster** |

### Implementation

DCNv4 is preferred when available, with automatic fallback to DCNv3:

```python
if DCNV4_AVAILABLE:
    self.deform_conv = DeformableConv2dV4(in_channels, out_channels)
else:
    self.deform_conv = DeformableConv2d(in_channels, out_channels)  # DCNv3
```

---

## Loss Functions

### LCOFL-EC (Extended with Embedding Consistency)

```
L_total = L_LCOFL + λ_embed × L_EC

Where:
- L_LCOFL = L_C + λ_layout × L_P + λ_ssim × L_S
- L_EC = max(m - D(V_SR, V_HR), 0)²
```

#### Classification Loss (L_C)
Weighted cross-entropy adapting to character confusions:
```
L_C = -(1/K) × Σ w_k × log(p(y_GT_k | x_SR))
w_k = 1 + α × confusion_count(k)
```

#### Layout Penalty (L_P)
Penalizes digit/letter position mismatches:
```
L_P = Σ [D(pred_i) × A(GT_i) + A(pred_i) × D(GT_i)]
```

#### Embedding Consistency Loss (L_EC) - NEW
Contrastive loss using Siamese network embeddings:
```
L_EC = max(margin - ManhattanDist(V_SR, V_HR), 0)²
```

**SiameseEmbedder Architecture**:
- Frozen ResNet-18 backbone
- 128-dim L2-normalized embeddings
- Manhattan distance for loss computation

---

## Five-Stage Progressive Training

```
┌──────────────────────────────────────────────────────────────────┐
│ Stage 0   │ Stage 1   │ Stage 2   │ Stage 3    │ Stage 4        │
│ OCR       │ Warm-up   │ LCOFL     │ Fine-tune  │ Hard Mining    │
│ Pretrain  │ (L1)      │ Training  │ (Joint)    │ (Curriculum)   │
│           │           │           │            │                │
│ 50 epochs │ 30 epochs │ 300 epochs│ 150 epochs │ 50 epochs      │
│ OCR only  │ Gen only  │ Gen only  │ Gen + OCR  │ Gen + Weighted │
└──────────────────────────────────────────────────────────────────┘
```

---

## Model Parameters

| Component | Parameters |
|-----------|------------|
| Shallow Extractor | ~50K |
| RRDB-EA Block (×16) | ~1.28M |
| MSCA Module | ~100K |
| SiameseEmbedder | ~11M (frozen backbone) |
| Upscaler + Reconstruction | ~6K |
| **Total Generator** | **~1.5M** |

---

## Key Design Decisions

### Why 2x Upscaling?
- More stable training than 4x
- Better matches real-world surveillance constraints
- Paper 2 achieved 49.8% vs Paper 1's 39.0% with 4x

### Why DCNv4?
- 3x faster training
- Better memory efficiency
- Unbounded weights learn more flexibly

### Why Multi-Scale Character Attention?
- Characters appear at different sizes in LR images
- Learned prototypes focus attention on text regions
- Improves recognition of small/blurry characters

### Why Embedding Consistency?
- Perceptual similarity beyond pixel metrics
- Frozen backbone provides stable gradients
- Contrastive loss prevents mode collapse
