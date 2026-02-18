# Changelog

All notable changes to LP-ASRN will be documented in this file.

## [3.0.0] - 2025-02-18

### Major Architecture Changes

**Removed:**
- RRDB-EA (Residual-in-Residual Dense Block with Enhanced Attention) generator
- SimpleCRNN OCR model
- DCNv4/DCNv3 deformable convolution support
- Multi-Scale Character Attention (MSCA) module
- Siamese Embedder and embedding consistency loss

**Added:**
- **SwinIR Transformer Generator** (12.8M parameters)
  - Shifted Window Attention (W-MSA/SW-MSA)
  - Residual Swin Transformer Blocks (RSTB)
  - 8 RSTB with 3 Swin blocks each
  - 144 embedding dimension, 8 attention heads
  - 6x6 window size for fine-grained attention
- **PARSeq OCR** (pretrained from HuggingFace)
  - Pretrained on millions of text images
  - Attention-based autoregressive decoding
  - Permutation Language Modeling (PLM) training
- **Character Pyramid Attention**
  - Layout-aware positional encoding
  - Multi-scale stroke detection
  - Gap detection between characters
  - Support for Brazilian (LLLNNNN) and Mercosur (LLLNLNN) layouts

### Configuration Changes

**New config options:**
```yaml
model:
  swinir_embed_dim: 144
  swinir_num_rstb: 8
  swinir_num_heads: 8
  swinir_window_size: 6
  swinir_num_blocks_per_rstb: 3
  swinir_mlp_ratio: 6.0
  use_pyramid_attention: true
  pyramid_layout: "brazilian"

ocr:
  model_type: "parseq"
  pretrained_path: "baudm/parseq-base"
  freeze_ocr: true
```

**Removed config options:**
- `num_rrdb_blocks`
- `num_filters`
- `use_deformable`
- `use_dcnv4`
- `use_character_attention`
- `msca_scales`
- `msca_num_prototypes`
- OCR `backbone_channels`, `lstm_hidden_size`, `lstm_num_layers`, `rnn_dropout`

### Training Changes

- Stage 0 now fine-tunes PARSeq instead of training SimpleCRNN
- All training scripts updated to use SwinIR Generator parameters
- Inference script updated with SwinIR architecture detection

### Breaking Changes

- **Old RRDB checkpoints are NOT compatible** with v3.0
- Must retrain from scratch with new SwinIR architecture
- SimpleCRNN OCR checkpoints are NOT compatible (now using PARSeq)

### Migration Guide

To migrate from v2.0 to v3.0:

1. Update config file (`configs/lp_asrn.yaml`)
2. Remove old checkpoints
3. Retrain from Stage 0:
   ```bash
   python scripts/train_progressive.py --config configs/lp_asrn.yaml
   ```

---

## [2.0.0] - 2024-08-15 (v2.0.0 - Hybrid Improvements)

### Added

- **LCOFL-EC Loss**: Embedding consistency loss with Siamese network
- **DCNv4 Support**: 3x faster deformable convolutions
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
