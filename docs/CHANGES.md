# Recent Changes and Fixes

## v2.0.0 - Hybrid Improvements (2026-02-04)

### New Features

#### Phase 1: Enhanced Loss Function (LCOFL-EC)
- **Embedding Consistency Loss**: Added contrastive loss component using Siamese network
- **SiameseEmbedder**: Uses frozen ResNet-18 backbone for stable semantic features
- **EmbeddingConsistencyLoss**: Manhattan distance-based contrastive loss with margin
- **AdaptiveWeightScheduler**: Gradual warm-up of embedding loss weight (0 → 0.3)

**New Files:**
- `src/models/siamese_embedder.py`: SiameseEmbedder, LightweightSiameseEmbedder
- `src/losses/embedding_loss.py`: EmbeddingConsistencyLoss, TripletEmbeddingLoss, CosineEmbeddingLoss
- `src/utils/adaptive_scheduler.py`: AdaptiveWeightScheduler

#### Phase 2: DCNv4 Integration
- **DCNv4 Support**: 3x faster training with flash-attention inspired memory access
- **Fallback Support**: Graceful fallback to DCNv3 if DCNv4 not available
- **Updated Attention**: EnhancedAttentionModule now prefers DCNv4 when available

**Modified Files:**
- `src/models/deform_conv.py`: Added DCNv4 implementation with fallback

#### Phase 3: Multi-Scale Character Attention (MSCA)
- **CharacterRegionDetector**: Learns character-like spatial patterns
- **MultiScaleCharacterAttention**: Processes features at multiple scales (1x, 0.5x, 0.25x)
- **Generator Integration**: MSCA applied before upscaling for character-aware reconstruction

**New Files:**
- `src/models/character_attention.py`: CharacterRegionDetector, MultiScaleCharacterAttention

#### Phase 4: OCR-Driven Curriculum (Stage 4)
- **HardExampleMiner**: Tracks per-sample OCR accuracy for weighted sampling
- **CharacterConfusionTracker**: Analyzes character-level confusion patterns
- **CurriculumSampler**: Gradually transitions from easy to hard examples
- **Stage 4 Training**: Focus on samples that OCR struggles with

**New Files:**
- `src/training/hard_example_miner.py`: HardExampleMiner, CharacterConfusionTracker, CurriculumSampler

---

## v1.1.0 - Deformable Convolution Fix (2026-01-29)

### Bug Fixes

#### cuDNN Contiguous Tensor Fix
- Added `.contiguous()` calls after `reshape()` operations in deformable convolution
- Fixed `RuntimeError: cuDNN error: CUDNN_STATUS_NOT_SUPPORTED`

**Files Modified:**
- `src/models/deform_conv.py`: Lines 180, 189, 197, 416, 424, 431, 453

### Configuration Updates
- Changed default `num_rrdb_blocks` from 16 to 12 for stability

---

## v1.0.0 - OCR Model Change (2026-01-29)

### Breaking Changes

#### SimpleCRNN as Primary OCR
- Switched from Parseq to SimpleCRNN for vocabulary compatibility
- 36-character vocabulary (0-9, A-Z) optimized for license plates
- CNN+BiLSTM architecture designed specifically for license plates

**Files Modified:**
- `src/ocr/parseq_wrapper.py`: SimpleCRNN fallback
- `requirements.txt`: Removed Parseq-specific dependencies
