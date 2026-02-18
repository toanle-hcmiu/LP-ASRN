# LP-ASRN Architecture

Detailed architectural documentation for the Layout-Aware and Character-Driven Super-Resolution Network with SwinIR Transformer.

## Overview

LP-ASRN (License Plate Super-Resolution Network) is a specialized super-resolution architecture designed specifically for license plate recognition. Unlike generic SR methods that optimize for pixel-level metrics, LP-ASRN incorporates character recognition supervision directly into the training process using **SwinIR Transformer** for the generator and **PARSeq** for OCR.

---

## High-Level Architecture

```
                    ┌─────────────────────────────────────────────────────┐
                    │              SwinIR Generator (12.8M params)        │
Input: LR Image     │  ┌────────────┐  ┌──────────────────┐  ┌─────────┐  │
   (B,3,H,W)  ───▶  │  │  Shallow   │  │  SwinIR Deep     │  │ Pyramid │  │
                    │  │  Extractor │─▶│  Feature Extract.│─▶│ Attention│  │
                    │  │  (Conv)    │  │  (8× RSTB)       │  │ (Optional│  │
                    │  └────────────┘  └──────────────────┘  └─────────┘  │
                    │         │                               │          │
                    │         ▼                               ▼          │
                    │  ┌────────────┐  ┌────────────────────────────┐   │
                    │  │  Upscaler  │◀─│  Character Pyramid Attention│   │
                    │  └────────────┘  └────────────────────────────┘   │
                    │         │                                          │
                    │         ▼                                          │
                    │  ┌────────────┐                                    │──▶ SR Output
                    │  │ Reconstruct│                                    │    (B,3,2H,2W)
                    │  └────────────┘                                    │
                    └─────────────────────────────────────────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │   PARSeq OCR    │
                    │  (Pretrained)    │
                    └─────────────────┘
```

---

## SwinIR Generator Components

### 1. Shallow Feature Extractor

Extracts initial features using a simple convolution:

```
LR Input → Conv3x3 → Shallow Features (embed_dim channels)
```

### 2. SwinIR Deep Feature Extractor

**Residual Swin Transformer Blocks (RSTB)** for efficient long-range modeling:

```
Shallow Features → Conv_First → [RSTB × 8] → Conv_After → Deep Features
                              ↓
                    Each RSTB contains:
                    - Swin Transformer Blocks (3 per RSTB)
                    - Window-based Multi-head Self Attention
                    - Shifted Window Attention for cross-window connections
                    - MLP with GELU activation
                    - Residual connection
```

#### Window-based Multi-head Self Attention (W-MSA)

```
Input Features (B, C, H, W)
      │
      ├──▶ Window Partition (6×6 windows)
      │         └── Each window: (window_size², C)
      │
      ├──▶ QKV Projection → Q, K, V
      │
      ├──▶ Attention: softmax(QK^T / √d + relative_pos_bias) × V
      │
      ├──▶ Window Reverse → Merge Windows
      │
      └──▶ Output Projection + Residual
```

**Relative Position Bias**: Learnable biases for each relative position in the window.

#### Shifted Window Attention (SW-MSA)

Alternates between regular and shifted windows to enable cross-window connections:
- Even layers: Regular W-MSA
- Odd layers: Shifted W-MSA (shifted by half window size)

### 3. Character Pyramid Attention (Optional)

Layout-aware multi-scale character attention:

```
Deep Features (B, C, H, W)
      │
      ├──▶ Stroke Detection (H/V/Diagonal)
      │         └── 4 learnable stroke kernels
      │
      ├──▶ Gap Detection
      │         └── Detect spaces between characters
      │
      ├──▶ Multi-Scale Processing
      │         ├── Scale 1.0×: Full resolution
      │         ├── Scale 0.5×: Half resolution
      │         └── Scale 0.25×: Quarter resolution
      │
      ├──▶ Layout-Aware Positional Encoding
      │         └── Brazilian: LLLNNNN (7 positions)
      │         └── Mercosur: LLLNLNN (7 positions)
      │
      └──▶ Fusion → Enhanced Features
```

**Layout Types**:
- **Brazilian**: LLLNNNN (3 letters + 4 digits)
- **Mercosur**: LLLNLNN (3 letters + digit + letter + 2 digits)

### 4. Upscaling Module

```
Features → Conv(embed_dim, embed_dim × 4) → PixelShuffle(2) → Upscaled (2× resolution)
```

**Progressive Refinement**: Intermediate convolutions and attention after upscaling.

### 5. Reconstruction Layer

```
Upscaled → Conv3x3 → Tanh → Skip(LR upsampled) → SR Output [-1, 1]
```

---

## PARSeq OCR Model

**Pretrained attention-based OCR** from HuggingFace (`baudm/parseq-base`).

### Architecture

```
Input Image (B, 3, 32, 128)
      │
      ├──▶ ViT Encoder
      │         └── Patch embedding + Transformer layers
      │
      ├──▶ Autoregressive Decoder
      │         ├── Cross-attention (encoder features)
      │         ├── Self-attention (target tokens)
      │         ├── Permutation Language Modeling (PLM)
      │         └── Character prediction head
      │
      └──▶ Output: Character sequence (7 chars)
```

### Training Protocol

**Permutation Language Modeling (PLM)**:
- Teacher forcing during training
- Multiple permutation orderings per batch
- Canonical + reverse + random permutations

### Fine-tuning

1. **Stage 0**: Fine-tune PARSeq on license plate HR images
2. **Stages 1-2**: Freeze OCR weights, use for LCOFL loss
3. **Stage 3**: Unfreeze for joint optimization
4. **Stage 4**: Refreeze for hard example mining

---

## Loss Functions

### LCOFL (Layout-Constrained Optical Flow Loss)

```
L_LCOFL = L_C + λ_layout × L_P + λ_ssim × L_S

Where:
- L_C = Classification Loss (weighted cross-entropy)
- L_P = Layout Penalty (position mismatches)
- L_S = SSIM Loss (structural similarity)
```

#### Classification Loss (L_C)

Weighted cross-entropy adapting to character confusions:
```
L_C = -(1/K) × Σ w_k × log(p(y_GT_k | x_SR))
w_k = 1 + α × confusion_count(k)
```

- Confused characters get higher weight
- Alpha increases over time (adaptive)

#### Layout Penalty (L_P)

Penalizes digit/letter position mismatches:
```
L_P = Σ [D(pred_i) × A(GT_i) + A(pred_i) × D(GT_i)]
```

- `D(c)`: Is character a digit?
- `A(c)`: Is character a letter?
- Penalizes letter↔digit confusion at each position

#### SSIM Loss (L_S)

Structural similarity for perceptual quality:
```
L_S = 1 - SSIM(SR, HR)
```

---

## Five-Stage Progressive Training

```
┌──────────────────────────────────────────────────────────────────┐
│ Stage 0   │ Stage 1   │ Stage 2   │ Stage 3    │ Stage 4        │
│ PARSeq    │ Warm-up   │ LCOFL     │ Fine-tune  │ Hard Mining    │
│ Fine-tune │ (L1)      │ Training  │ (Joint)    │ (Curriculum)   │
│           │           │           │            │                │
│ 50 epochs │ 30 epochs │ 200 epochs│ 100 epochs │ 50 epochs      │
│ OCR only  │ Gen only  │ Gen only  │ Gen + OCR  │ Gen + Weighted │
└──────────────────────────────────────────────────────────────────┘
```

### Stage 0: PARSeq Pretraining
- Fine-tune PARSeq on HR license plate images
- PLM training with teacher forcing
- Result: `checkpoints/ocr/best.pth`

### Stage 1: Warm-up
- Generator with L1 loss only
- Stabilizes training before complex losses
- Frozen OCR

### Stage 2: LCOFL Training
- Character-driven optimization
- L1 + LCOFL loss
- Update confusion weights
- Frozen OCR

### Stage 3: Fine-tuning
- Joint optimization of generator + OCR
- Lower learning rate
- Unfrozen OCR

### Stage 4: Hard Example Mining
- Focus on difficult samples
- Weighted sampling by OCR confidence
- Frozen OCR

---

## Model Parameters

| Component | Parameters |
|-----------|------------|
| Shallow Extractor | ~100K |
| SwinIR Deep Features (8 RSTB) | ~12.5M |
| Character Pyramid Attention | ~200K |
| Upscaler + Reconstruction | ~10K |
| **Total Generator** | **~12.8M** |
| PARSeq OCR (frozen) | ~51M (pretrained) |

---

## Key Design Decisions

### Why SwinIR over CNN?

| Aspect | CNN (RRDB) | SwinIR |
|--------|-----------|--------|
| Long-range modeling | Limited (receptive field) | Excellent (global attention) |
| Parameter efficiency | Moderate | High |
| Training stability | Good | Better |
| Recognition accuracy | ~50% | **Target: 60%+** |

### Why 2x Upscaling?
- More stable training than 4x
- Better matches real-world surveillance constraints
- Paper 2 achieved 49.8% vs Paper 1's 39.0% with 2x

### Why PARSeq OCR?
- Pretrained on millions of text images
- Attention-based architecture
- Autoregressive decoding with language modeling
- State-of-the-art accuracy on text recognition

### Why Character Pyramid Attention?
- Layout-aware positional encoding
- Multi-scale stroke detection
- Focus on character regions
- Adapts to different plate formats (Brazilian/Mercosur)

### Why Shifted Window Attention?
- Linear complexity (O(n) vs O(n²) for global attention)
- Efficient implementation
- Cross-window connections via shifting
- Better for small license plate images

---

## Configuration Reference

### Maximum Configuration (Best Accuracy)

```yaml
model:
  swinir_embed_dim: 144         # High capacity
  swinir_num_rstb: 8            # Deep transformer
  swinir_num_heads: 8           # 144 / 8 = 18
  swinir_window_size: 6         # Fine-grained attention
  swinir_num_blocks_per_rstb: 3 # More depth per RSTB
  swinir_mlp_ratio: 6.0         # Wide MLP
  use_pyramid_attention: true   # Character-aware
```

### Lightweight Configuration (Faster Training)

```yaml
model:
  swinir_embed_dim: 96          # Reduced
  swinir_num_rstb: 4            # Shallower
  swinir_num_heads: 6
  swinir_window_size: 8
  swinir_num_blocks_per_rstb: 2
  swinir_mlp_ratio: 4.0
  use_pyramid_attention: false  # Disable for speed
```
