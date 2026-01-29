# LACD-SRN Architecture

Detailed architectural documentation for the Layout-Aware and Character-Driven Super-Resolution Network.

## Overview

LACD-SRN (Layout-Aware and Character-Driven Super-Resolution Network) is a specialized super-resolution architecture designed specifically for license plate recognition. Unlike generic SR methods that optimize for pixel-level metrics, LACD-SRN incorporates character recognition supervision directly into the training process.

---

## Generator Architecture

### High-Level Structure

```
Input: LR Image (B, 3, H, W) → [Generator] → SR Image (B, 3, 2H, 2W)
```

### Components

#### 1. Shallow Feature Extractor

```python
LR Input (B, 3, H, W)
    ↓
Conv 5x5, 64 filters
    ↓
PixelUnshuffle (compress spatial → channels)
    ↓
Conv 3x3, compressed channels
    ↓
PixelShuffle (expand channels → spatial)
    ↓
Conv 3x3, 64 filters
    ↓
Skip Connection
    ↓
Shallow Features (B, 64, H, W)
```

**Purpose**: Extract and reorganize initial features, eliminating less significant information early.

#### 2. Deep Feature Extractor

```
Shallow Features (B, 64, H, W)
    ↓
[16x RRDB-EA Blocks]
    ↓
Global Conv + Skip Connection
    ↓
Deep Features (B, 64, H, W)
```

Each RRDB-EA (Residual-in-Residual Dense Block with Enhanced Attention):

```
Input Features
    ↓
[Dense Conv Layers (3-4 layers)]
    ↓
Enhanced Attention Module (EAM)
    ↓
Local Conv
    ↓
Residual Connection
```

#### 3. Enhanced Attention Module (EAM)

```
Input Features (B, 64, H, W)
    ↓
┌─────────────────────────────────────────┐
│  Channel Unit (CA)                      │
│  - Conv 1x1 (parallel branches)           │
│  - PixelUnshuffle → Conv → PixelShuffle   │
│  - Outputs: channel importance weights    │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Positional Unit (POS)                    │
│  - Deformable Conv (adaptive sampling)     │
│  - PixelUnshuffle → Conv → PixelShuffle   │
│  - Outputs: spatial importance weights     │
└─────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────┐
│  Geometrical Perception Unit (GP)         │
│  - Global avg pool (horizontal/vertical)  │
│  - Point-wise conv                       │
│  - Outputs: structural feature importance │
└─────────────────────────────────────────┘
    ↓
Combine: CA × POS + GP (element-wise operations)
    ↓
Sigmoid activation
    ↓
Attention Mask
    ↓
Deformable Conv
    ↓
Enhanced Features
```

#### 4. Upscaling Module

```
Deep Features (B, 64, H, W)
    ↓
Conv (B, 3 × r², H, W)  # r=2 for 2x upscaling
    ↓
PixelShuffle (rearrange channels → spatial)
    ↓
Upscaled Features (B, 3, 2H, 2W)
```

#### 5. Reconstruction Layer

```
Upscaled Features (B, 3, 2H, 2W)
    ↓
Conv 3x3, 3 filters
    ↓
Tanh activation (clamp to [-1, 1])
    ↓
Skip Connection (from input LR upsampled)
    ↓
SR Output (B, 3, 2H, 2W)
```

---

## Deformable Convolution

### Standard vs Deformable Convolution

**Standard Convolution**:
- Fixed sampling grid (e.g., 3×3)
- Samples at predetermined positions relative to each pixel
- Limited ability to adapt to irregular shapes

**Deformable Convolution**:
- Learns offset parameters for each sampling location
- Adapts receptive field to match character shapes
- Better captures curved and irregular character strokes

### Implementation

```
Input Features (B, C, H, W)
    ↓
Offset Conv (2 × k × k filters) → Offsets (B, 2k², H', W')
    ↓
Deformable Conv (k × k filters with learned offsets)
    ↓
Output Features (B, C, H', W')
```

---

## Loss Functions

### LCOFL (Layout and Character Oriented Focal Loss)

```
L_LCOFL = L_C + λ_layout × L_P + λ_ssim × L_S
```

#### Classification Loss (L_C)

Weighted cross-entropy that adapts to character confusions:

```
L_C = -(1/K) × Σ w_k × log(p(y_GT_k | x_SR))

w_k = 1 + α × confusion_count(k)
```

#### Layout Penalty (L_P)

Penalizes digit/letter position mismatches:

```
L_P = Σ [D(pred_i) × A(GT_i) + A(pred_i) × D(GT_i)]

D(c) = β if c is digit
A(c) = β if c is letter
```

---

## OCR Integration

### Parseq Model

- **Pre-trained**: baudm/parseq-base from HuggingFace
- **Fine-tuned**: On HR license plate images
- **Frozen during SR training**: Provides stable gradients

### OCR as Discriminator

In the GAN-inspired training paradigm:

1. **Generator**: Creates super-resolved images
2. **Discriminator (OCR)**: Evaluates character recognizability
3. **Objective**: Generator produces images that OCR can classify correctly

---

## Training Stability Features

### 1. Progressive Training

Three stages prevent instability from complex loss landscapes:
- **Stage 1**: Warm-up with simple L1 loss
- **Stage 2**: Introduce LCOFL gradually
- **Stage 3**: Joint optimization (if needed)

### 2. Learning Rate Scheduling

StepLR based on **recognition rate** (not loss):
```
Every 5 epochs:
  IF recognition_rate NOT improved:
    learning_rate *= 0.9
```

### 3. Gradient Clipping

Prevents exploding gradients in deep networks:
```
IF gradient_norm > 1.0:
    gradient = gradient / gradient_norm × 1.0
```

### 4. Early Stopping

Monitors metrics and stops when:
- 20 epochs without improvement (Stage 1)
- Recognition rate plateaus (Stage 2)
- Word accuracy plateaus (Stage 3)

---

## Model Parameters

| Component | Parameters |
|-----------|------------|
| Shallow Extractor | ~50K |
| RRDB-EA Block | ~80K |
| Total (16 blocks) | ~1.28M |
| EAM per block | ~10K |
| Upscaler | ~3K |
| Reconstruction | ~3K |
| **Total Generator** | **~1.38M** |

---

## Input/Output Specifications

### Input
- **Format**: RGB images
- **Range**: [-1, 1] (normalized)
- **Size**: Variable (e.g., 17×31, 31×17)

### Output
- **Format**: RGB images
- **Range**: [-1, 1] (normalized)
- **Size**: 2× input (e.g., 34×62, 62×34)

### Supported Formats

| Layout | Pattern | Example |
|--------|--------|---------|
| Brazilian | LLLNNNN | ABC1234 |
| Mercosur | LLLNLNN | ABC1D23 |

---

## Key Design Decisions

### Why 2x Upscaling?

Paper 2 achieved better recognition with 2x than Paper 1 with 4x:
- More stable training
- Better matches real-world surveillance constraints
- Sufficient for most OCR systems

### Why Progressive Training?

1. **Stability**: Warm-up prevents early loss landscape exploration failures
2. **Convergence**: Each stage focuses on specific objectives
3. **Performance**: Fine-tuning extracts final improvements

### Why OCR-Guided Training?

- Direct optimizes for the end task (recognition)
- Avoids optimizing irrelevant features (textures, backgrounds)
- Handles character confusion explicitly through LCOFL
