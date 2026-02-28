# LP-ASRN Architecture

Detailed architectural documentation for the Layout-Aware and Character-Driven Super-Resolution Network with RRDB-EA generator and PARSeq OCR.

## Overview

LP-ASRN (License Plate Super-Resolution Network) is a specialized super-resolution architecture designed specifically for license plate recognition. Unlike generic SR methods that optimize for pixel-level metrics, LP-ASRN incorporates character recognition supervision directly into the training process using **RRDB-EA** (Residual-in-Residual Dense Block with Enhanced Attention) for the generator and **PARSeq** for OCR.

---

## High-Level Architecture

```
                    +-----------------------------------------------------+
                    |            RRDB-EA Generator (~3.99M params)         |
Input: LR Image    |  +------------+  +------------------+  +---------+   |
   (B,3,H,W)  ---> |  |  Shallow   |  |  Deep Feature    |  | Upscale |   |
                    |  |  Feature   |->|  Extractor       |->| Module  |   |
                    |  |  Extractor |  |  (12x RRDB-EA)   |  | (PShufl)|   |
                    |  +------------+  +------------------+  +---------+   |
                    |         |                                    |        |
                    |         |            Global Residual         |        |
                    |         +-----------------------------------+        |
                    |                                              |        |
                    |  +--------------------------------------------+      |
                    |  |  Reconstruction (Conv3x3 + skip, w=0.2)   |      |--> SR Output
                    |  +--------------------------------------------+      |    (B,3,2H,2W)
                    +-----------------------------------------------------+
                              |
                              v
                    +------------------+
                    |    PARSeq OCR    |
                    |   (~51M, frozen) |
                    +------------------+
```

---

## RRDB-EA Generator Components

### 1. Shallow Feature Extractor

PixelUnshuffle autoencoder for efficient initial feature extraction:

```
LR Input (B, 3, H, W)
    --> PixelUnshuffle (factor=2) --> (B, 12, H/2, W/2)
    --> Conv3x3 --> (B, num_features, H/2, W/2)
    --> Conv3x3 --> (B, num_features, H/2, W/2)
    --> PixelShuffle (factor=2) --> Shallow Features (B, num_features/4, H, W)
```

### 2. Deep Feature Extractor (12x RRDB-EA Blocks)

**Residual-in-Residual Dense Blocks with Enhanced Attention** for deep feature extraction:

```
Shallow Features --> [RRDB-EA x 12] --> Conv3x3 --> Deep Features
                            |
                  Each RRDB-EA contains:
                  - 3 Dense Layers with growth connections
                  - Enhanced Attention Module (channel + spatial)
                  - Optional Deformable Convolution
                  - Residual scaling (beta = 0.2)
```

#### Dense Connections

Each RRDB-EA block has 3 dense layers where each layer receives features from all previous layers:

```
Layer 1: x1 = LeakyReLU(Conv(x0))
Layer 2: x2 = LeakyReLU(Conv(cat(x0, x1)))
Layer 3: x3 = LeakyReLU(Conv(cat(x0, x1, x2)))
Output:  x0 + beta * attention(x3)     # beta = 0.2
```

#### Enhanced Attention Module

Channel attention + spatial attention applied to each block's output:

```
Input Features
    --> Channel Attention (global avg pool --> FC --> sigmoid)
    --> Spatial Attention (conv --> sigmoid)
    --> Attended Features
```

#### Deformable Convolutions (Optional)

Adaptive receptive field for handling geometric variation in license plates:
- Learns spatial offsets per pixel
- Better handles perspective distortion and rotation
- Enabled via `use_deformable: true` in config

#### Global Residual Connection

```
Deep Features = Shallow Features + Conv(RRDB_output)
```

### 3. Upscaling Module

```
Deep Features --> Conv(num_features, num_features * 4) --> PixelShuffle(2) --> Upscaled (2x)
```

### 4. Reconstruction Layer

```
Upscaled --> Conv3x3(num_features, 3) --> Raw Output
LR --> Bilinear Upsample(2x) --> LR_up

# Skip connection with soft clamp
if |Raw Output| > 1: output = tanh(Raw Output)
SR = LR_up + skip_weight * output       # skip_weight = 0.2
```

---

## PARSeq OCR Model

**Pretrained attention-based OCR** from HuggingFace (`baudm/parseq`).

### Architecture

```
Input Image (B, 3, 32, 128)
      |
      +--> ViT Encoder
      |         +-- Patch embedding + Transformer layers
      |
      +--> Autoregressive Decoder
      |         +-- Cross-attention (encoder features)
      |         +-- Self-attention (target tokens)
      |         +-- Permutation Language Modeling (PLM)
      |         +-- Character prediction head
      |
      +--> Output: Character sequence (max 7 chars)
```

### Vocabulary Mapping

PARSeq natively uses a 95-character vocabulary (97 with special tokens). LP-ASRN maps to a 36-character LP vocabulary:
- **Digits**: 0-9 (PARSeq indices 1-10)
- **Letters**: A-Z (PARSeq indices 37-62)

### Training Protocol

**Permutation Language Modeling (PLM)**:
- 6 permutations per batch (forward + mirrored + random)
- Teacher forcing during training
- Label smoothing: 0.1

### Preprocessing

- Resize to **32 x 128** pixels
- ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

### PlateFormatValidator

Post-processing correction for Brazilian/Mercosur formats:
- **Brazilian**: `LLLNNNN` (3 letters + 4 digits)
- **Mercosur**: `LLLNLNN` (3 letters + 1 digit + 1 letter + 2 digits)
- Uses character confusion pairs (e.g., O<->0, I<->1, B<->8) for correction

### Fine-tuning Protocol

1. **Stage 0**: Fine-tune PARSeq on license plate HR images (50 epochs)
2. **Stages 1-4**: Freeze OCR weights — used for LCOFL loss computation only

---

## Loss Functions

### LCOFL (Layout-Constrained Optical Flow Loss)

```
L_LCOFL = L_C + lambda_layout * L_P

Where:
- L_C = Classification Loss (weighted cross-entropy)
- L_P = Layout Penalty (position mismatches)
```

#### Classification Loss (L_C)

Weighted cross-entropy adapting to character confusions:
```
L_C = -(1/K) * sum( w_k * log(p(y_GT_k | x_SR)) )
w_k = 1 + alpha * (row_sum - diagonal)   # from confusion matrix
```

- Character weights stored on `ClassificationLoss.weights` (shape: 36)
- Updated each epoch via confusion matrix analysis
- Supports PARSeq's 97-dim logits via automatic vocabulary mapping

#### Layout Penalty (L_P)

Differentiable penalty using OCR logits (not decoded text):
```
L_P = sum( digit_prob_at_letter_positions + letter_prob_at_digit_positions )
```

- Uses probability mass from softmax logits
- Digit/letter masks as registered buffers
- `lambda_layout: 0.5`

### Standalone SSIM Loss (v3.1)

Separate SSIM component added to `loss_components` (distinct from SSIM inside LCOFL):
```
L_SSIM = 1 - SSIM(SR, HR)
```

- Gaussian window (size=11, sigma=1.5)
- `lambda_ssim: 0.2`
- Prevents visual quality collapse during aggressive LCOFL optimization

### Additional Loss Components

| Loss | Lambda | Stages | Description |
|------|--------|--------|-------------|
| L1 | 1.0 | 1-4 | Pixel-wise reconstruction |
| LCOFL | 0.5 | 2-4 | Character-driven classification + layout |
| SSIM | 0.2 | 2-3 | Standalone structural similarity |
| Gradient | 0.05 | 3 | Edge-preserving gradient loss |
| Frequency | 0.05 | 3 | FFT-based frequency domain loss |
| Edge | 0.05 | 3 | Canny-style edge awareness |
| Perceptual | 0.1 | (optional) | VGG19 feature matching |
| GAN | 0.01 | (optional) | Adversarial training (disabled by default) |

---

## Quality Guardrails (v3.1)

### PSNR Guardrail

Prevents LCOFL from collapsing visual quality (observed: PSNR 13.88 -> 12.0 with lambda_lcofl=1.5):

```python
if config.psnr_floor > 0 and val_psnr < config.psnr_floor:
    lcofl_scale = max(0.1, val_psnr / config.psnr_floor)
    effective_lcofl_weight = lambda_lcofl * lcofl_scale
else:
    lcofl_scale = 1.0
```

- Configured via `psnr_floor: 12.5` in Stage 2
- Dynamically reduces LCOFL weight when visual quality drops
- Minimum scale: 0.1 (never fully disables LCOFL)

### Balanced Checkpoint

Saves the best model combining both recognition accuracy AND visual quality:

```python
balanced_score = word_acc * min(val_psnr / 13.0, 1.0)
```

- Output: `best_balanced.pth`
- Use for submission instead of `best.pth` if PSNR collapsed
- PSNR threshold of 13.0 chosen based on observed quality thresholds

---

## Five-Stage Progressive Training

```
+----------------------------------------------------------------+
| Stage 0   | Stage 1   | Stage 2     | Stage 3    | Stage 4     |
| PARSeq    | Warm-up   | LCOFL       | Fine-tune  | Hard Mining |
| Fine-tune | (L1)      | + Guardrail | (Multi-L)  | (Curriculum)|
|           |           |             |            |             |
| 50 epochs | 80 epochs | 200 epochs  | 200 epochs | 50 epochs   |
| OCR only  | Gen only  | Gen only    | Gen only   | Gen+Weighted|
| lr=5e-4   | lr=1e-4   | lr=1e-4     | lr=1e-5    | lr=5e-6     |
+----------------------------------------------------------------+
```

### Stage 0: PARSeq Pretraining
- Fine-tune PARSeq on HR license plate images (PNG + JPG)
- PLM training with teacher forcing
- Result: `checkpoints/ocr/best.pth`

### Stage 1: Warm-up
- RRDB-EA generator with L1 loss only
- Stabilizes training before complex losses
- OCR frozen, used for monitoring only

### Stage 2: LCOFL Training
- Character-driven optimization
- L1 + LCOFL + standalone SSIM loss
- PSNR guardrail: auto-scales LCOFL weight
- Balanced checkpoint: saves best accuracy-quality tradeoff
- Update confusion weights each epoch
- OCR frozen

### Stage 3: Fine-tuning
- Extended optimization with six loss components
- L1 + LCOFL + SSIM + gradient + frequency + edge
- OCR remains **frozen** for stability
- Narrowed aspect ratio range [0.25, 0.45] matching test distribution

### Stage 4: Hard Example Mining
- Focus on difficult samples
- Weighted sampling by OCR confidence
- OCR frozen

---

## Data Loading

### Dual-Format Support (v3.1)

Dataset loads both PNG and JPG images:
- **Scenario-A**: PNG images (lighter degradation)
- **Scenario-B**: JPG images (heavier degradation)
- Tries `.png` first, falls back to `.jpg`
- 5 image pairs per track x 20,000 tracks = ~100,000 samples

### Augmentation Pipeline

| Augmentation | Target | Config Key | Description |
|-------------|--------|------------|-------------|
| JPEG Compression | LR only | `jpeg_augment: true` | Simulates JPEG artifacts (quality 60-95) |
| Test Resolution | LR+HR | `test_resolution_augment: true` | Downsamples to match test distribution |
| No-Crop | LR+HR | `no_crop_prob: 0.3` | Skips corner cropping (test images lack corners) |
| Aspect Ratio | LR+HR | `aspect_ratio_augment: true` | Varies aspect ratios [0.25, 0.45] |
| Geometric | LR+HR | Always | RandomAffine + RandomPerspective (no H-flip) |
| Photometric | LR only | Always | ColorJitter + GaussianBlur + GaussianNoise |

---

## Model Parameters

| Component | Parameters |
|-----------|------------|
| Shallow Feature Extractor | ~50K |
| Deep Features (12x RRDB-EA) | ~3.8M |
| Enhanced Attention (12x) | ~100K |
| Upscaler + Reconstruction | ~40K |
| **Total Generator** | **~3.99M** |
| PARSeq OCR (frozen) | ~51M (pretrained) |

---

## Key Design Decisions

### Why RRDB-EA over SwinIR?

| Aspect | SwinIR (backup) | RRDB-EA (active) |
|--------|-----------------|------------------|
| Parameters | ~12.8M | ~3.99M |
| Training speed | Slower | Faster |
| Memory usage | Higher | Lower |
| Small image handling | Extra padding needed | Native |
| Dense feature reuse | Limited | Excellent |
| Deformable conv | N/A | Supported |

> **Note**: A SwinIR variant exists at `src/models/generator_swinir_backup.py`. The inference script auto-detects architecture from checkpoint keys and supports both models.

### Why 2x Upscaling?
- More stable training than 4x
- Better matches real-world surveillance constraints
- Paper 2 achieved 49.8% vs Paper 1's 39.0% with 2x

### Why PARSeq OCR?
- Pretrained on millions of text images
- Attention-based architecture
- Autoregressive decoding with language modeling
- State-of-the-art accuracy on text recognition

### Why Frozen OCR in All SR Stages?
- Original plan unfroze OCR in Stage 3 — caused instability
- Frozen OCR provides stable gradient signal for generator
- Generator learns to produce images OCR can read, without OCR co-adapting

### Why PSNR Guardrail?
- LCOFL with high weight (lambda=1.5) caused PSNR collapse: 13.88 -> 12.0
- Quality collapse creates artifacts that fool OCR on training set but generalize poorly
- Dynamic scaling prevents this while still allowing character-driven optimization

---

## Configuration Reference

### Active Configuration (RRDB-EA)

```yaml
model:
  num_features: 64
  num_blocks: 12
  num_layers_per_block: 3
  use_enhanced_attention: true
  use_deformable: true
  upscale_factor: 2
  use_character_attention: false

loss:
  lambda_lcofl: 0.5
  lambda_layout: 0.5
  lambda_ssim: 0.2
  lambda_gradient: 0.05
  lambda_frequency: 0.05
  lambda_edge: 0.05
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

### SwinIR Backup Configuration

```yaml
model:
  swinir_embed_dim: 144
  swinir_num_rstb: 8
  swinir_num_heads: 8
  swinir_window_size: 6
  swinir_num_blocks_per_rstb: 3
  swinir_mlp_ratio: 6.0
  use_pyramid_attention: false
```
