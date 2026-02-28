# Changelog

All notable changes to LP-ASRN will be documented in this file.

## [3.1.0] - 2025-02-26

### Critical Bug Fixes

- **Fixed 50% data loss**: Dataset hardcoded `.png` extension, silently skipping all Scenario-B tracks (JPG images). Added PNG-then-JPG fallback — training data doubled from ~50K to ~100K samples.
- **Fixed LCOFL weights crash**: `save_checkpoint` referenced `self.lcofl_loss.weights` but weights live on inner `ClassificationLoss`. Fixed all 3 references to `self.lcofl_loss.classification_loss.weights`.
- **Fixed LCOFL quality collapse**: PSNR dropped from 13.88 to 12.0 during LCOFL training with `lambda_lcofl=1.5`. Reduced to 0.5 and added guardrails.

### Added

- **PSNR Guardrail** (`psnr_floor: 12.5`): Dynamically scales LCOFL weight when PSNR drops below floor. Formula: `lcofl_scale = max(0.1, val_psnr / psnr_floor)`.
- **Balanced Checkpoint** (`best_balanced.pth`): Saves model with best `word_acc * min(psnr/13.0, 1.0)` score — combines recognition accuracy with visual quality.
- **Standalone SSIM Loss**: Separate SSIM loss component in Stages 2-3 (`lambda_ssim: 0.2`), independent from SSIM inside LCOFL.
- **JPEG Compression Augmentation**: Simulates JPEG artifacts (quality 60-95) to bridge PNG training → JPG test gap.
- **Test-Resolution Augmentation**: Downsamples training images to match test-public resolution distribution (`test_resolution_prob: 0.7`).
- **No-Crop Probability**: Skips corner cropping (`no_crop_prob: 0.3`) since test images have no corner annotations.
- **Aspect Ratio Augmentation**: Varies aspect ratios to match test distribution (`test_aspect_range: [0.25, 0.45]`).
- **Inference Diagnostic Modes**: `--diagnose` for per-track analysis, `--diagnose-val` for validation comparison.
- **Multi-Scale Inference**: `--multi-scale` flag for 0.8x/1.0x/1.2x ensemble.
- **Test-Time Augmentation**: `--tta` flag for horizontal flip ensemble.
- **JPEG Deblocking**: `--jpeg-deblock` flag for Gaussian deblocking preprocessing.
- **Aspect Ratio Preservation**: `--preserve-aspect` flag to pad instead of stretch.
- **Comprehensive Pipeline Tests**: `scripts/test_pipeline.py` with 14 tests covering all modified code paths.
- **`StageConfig.psnr_floor`**: Quality guardrail configuration per stage.
- **Test-Like Validation**: Simulates test conditions during training validation (`test_like_val: true`).

### Changed

- **`lambda_lcofl`**: 1.5 → 0.5 (prevents PSNR collapse)
- **Stage 1 epochs**: 30 → 80 (better generator stabilization)
- **Stage 3 epochs**: 100 → 200 (extended optimization)
- **Stage 3 `freeze_ocr`**: false → **true** (OCR stays frozen for stability)
- **Stage 3 `loss_components`**: `[l1, lcofl]` → `[l1, lcofl, ssim, gradient, frequency, edge]` (six loss components)
- **Stage 2 `loss_components`**: `[l1, lcofl]` → `[l1, lcofl, ssim]` (added standalone SSIM)
- **Stage 2 LR**: 2e-4 → 1e-4
- **`val_split`**: 0.05 → 0.10 (larger validation set)
- **Checkpoint saving**: `save_checkpoint` now accepts `save_path` parameter for balanced checkpoint
- **Checkpoint state**: Now saves/loads `best_balanced_score`, `last_val_psnr`, `_lcofl_scale`

---

## [3.0.0] - 2025-02-18

### Major Architecture Changes

**Added:**
- **PARSeq OCR** (pretrained from HuggingFace)
  - Pretrained on millions of text images
  - Attention-based autoregressive decoding
  - Permutation Language Modeling (PLM) training
  - 36-character LP vocabulary mapped from 97-char PARSeq vocab
  - `PlateFormatValidator` for Brazilian/Mercosur format correction
- **SwinIR Transformer Generator** (backup variant)
  - Available at `src/models/generator_swinir_backup.py`
  - Shifted Window Attention (W-MSA/SW-MSA)
  - Residual Swin Transformer Blocks (RSTB)
  - Inference auto-detects RRDB vs SwinIR from checkpoint keys
- **Five-Stage Progressive Training**
  - Stage 0: PARSeq pretraining
  - Stage 1: Warm-up (L1 only)
  - Stage 2: LCOFL character-driven training
  - Stage 3: Fine-tuning with multi-loss
  - Stage 4: Hard example mining with curriculum learning

**Retained (primary architecture):**
- **RRDB-EA Generator** (~3.99M parameters)
  - Residual-in-Residual Dense Blocks with Enhanced Attention
  - PixelUnshuffle autoencoder shallow feature extraction
  - Deformable convolution support
  - PixelShuffle 2x upscaling with skip connections

**Removed:**
- SimpleCRNN OCR model (replaced by PARSeq)

### Configuration Changes

**Active config keys (RRDB-EA):**
```yaml
model:
  num_features: 64
  num_blocks: 12
  num_layers_per_block: 3
  use_enhanced_attention: true
  use_deformable: true
  upscale_factor: 2
  use_character_attention: false

ocr:
  model_type: "parseq"
  pretrained_path: "baudm/parseq-base"
  freeze_ocr: true
  max_length: 7
  vocab: "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
```

### Training Changes

- Stage 0 fine-tunes PARSeq instead of training SimpleCRNN
- Inference script auto-detects RRDB-EA vs SwinIR architecture from checkpoint keys
- Hard example mining (Stage 4) with `HardExampleMiner` and `CurriculumSampler`

### Breaking Changes

- **SimpleCRNN OCR checkpoints are NOT compatible** (now using PARSeq)
- Must retrain OCR from Stage 0

---

## [2.0.0] - 2024-08-15 (v2.0.0 - Hybrid Improvements)

### Added

- **LCOFL-EC Loss**: Embedding consistency loss with Siamese network
- **DCNv4 Support**: Deformable convolutions
- **Multi-Scale Character Attention (MSCA)**: Character-aware attention at multiple scales
- **Hard Example Mining (Stage 4)**: Curriculum learning focused on difficult samples

### Changed

- Improved progressive training pipeline
- TensorBoard integration
- Better data augmentation

---

## [1.0.0] - 2023-05-20

### Initial Release

- RRDB-EA based generator
- SimpleCRNN OCR
- LCOFL loss function
- Three-stage progressive training
