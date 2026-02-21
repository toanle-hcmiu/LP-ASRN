"""
Inference Script for LP-ASRN Competition Submission

Runs the trained SR + OCR pipeline on test data and produces:
- predictions.txt  (track_id,plate_text;confidence)
- submission.zip   (ready to upload to CodaBench)

Usage:
    python scripts/inference.py --checkpoint outputs/run_XXX/best.pth
    python scripts/inference.py --checkpoint outputs/run_XXX/best.pth --data-root data/test-public
    python scripts/inference.py --checkpoint outputs/run_XXX/best.pth --ocr-only  # skip SR, OCR directly on LR
"""

import argparse
import os
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.generator import Generator
from src.ocr.ocr_model import OCRModel, PlateFormatValidator


def parse_args():
    parser = argparse.ArgumentParser(description="LP-ASRN Inference for Competition Submission")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint (.pth)")
    parser.add_argument("--config", type=str, default="configs/lp_asrn.yaml",
                        help="Path to config YAML")
    parser.add_argument("--data-root", type=str, default="data/test-public",
                        help="Root directory of test data")
    parser.add_argument("--output-dir", type=str, default="submission",
                        help="Output directory for predictions.txt and submission.zip")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference")
    parser.add_argument("--beam-width", type=int, default=5,
                        help="Beam width for CTC decoding (1=greedy, 5=accurate)")
    parser.add_argument("--ocr-only", action="store_true",
                        help="Skip SR generator, run OCR directly on LR images")
    parser.add_argument("--no-sr", action="store_true",
                        help="Alias for --ocr-only")
    parser.add_argument("--tta", action="store_true",
                        help="Enable test-time augmentation (horizontal flip + multi-scale)")
    parser.add_argument("--no-resize", action="store_true",
                        help="Do NOT resize test images to lr_size (use native resolution)")
    parser.add_argument("--lr-size", type=int, nargs=2, default=None, metavar=("H", "W"),
                        help="Override LR size (default: from config or no resize)")
    parser.add_argument("--preserve-aspect", action="store_true",
                        help="Pad images to match target aspect ratio before resizing (avoids distortion)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Run diagnostic mode: compare strategies, print samples, check pipeline")
    parser.add_argument("--diagnose-val", type=str, default=None,
                        help="Path to training data root to validate inference vs training accuracy")
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    print(f"Warning: Config file {config_path} not found. Using defaults.")
    return {}


def _strip_ddp_prefix(state_dict):
    """Remove 'module.' prefix from DDP-wrapped state dict keys."""
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def _detect_architecture_from_checkpoint(gen_state):
    """
    Auto-detect generator architecture from checkpoint keys.

    Supports both SwinIR (new) and RRDB (old) architectures.

    Returns:
        dict of detected architecture parameters
    """
    detected = {}

    # Detect architecture type from key patterns
    has_swinir = any(k.startswith("deep_features.layers.") for k in gen_state)
    has_rrdb = any(k.startswith("deep_extractor.blocks.") for k in gen_state)
    detected["architecture"] = "swinir" if has_swinir else ("rrdb" if has_rrdb else "unknown")

    # For SwinIR: detect SwinIR-specific parameters
    if has_swinir:
        # Detect embed_dim from conv_first weight shape
        if "conv_first.weight" in gen_state:
            detected["embed_dim"] = gen_state["conv_first.weight"].shape[0]

        # Detect num_rstb from deep_features.layers keys
        rstb_indices = set()
        for k in gen_state:
            if k.startswith("deep_features.layers."):
                parts = k.split(".")
                if len(parts) >= 3 and parts[2].isdigit():
                    rstb_indices.add(int(parts[2]))
        if rstb_indices:
            detected["num_rstb"] = max(rstb_indices) + 1

        # Detect num_heads from window attention keys
        for k in gen_state:
            if "attn.qkv.weight" in k:
                # qkv has shape [embed_dim * 3, embed_dim]
                detected["embed_dim"] = gen_state[k].shape[1]
                break

    # For RRDB: detect RRDB-specific parameters (legacy support)
    elif has_rrdb:
        # Detect use_character_attention: look for character_attention.* keys
        has_char_attn = any(k.startswith("character_attention.") for k in gen_state)
        detected["use_character_attention"] = has_char_attn

        # Detect num_features from first conv weight shape
        if "shallow_extractor.conv1.weight" in gen_state:
            detected["num_features"] = gen_state["shallow_extractor.conv1.weight"].shape[0]

        # Detect num_blocks from deep_extractor block keys
        block_indices = set()
        for k in gen_state:
            if k.startswith("deep_extractor.blocks."):
                parts = k.split(".")
                if len(parts) >= 3 and parts[2].isdigit():
                    block_indices.add(int(parts[2]))
        if block_indices:
            detected["num_blocks"] = max(block_indices) + 1

        # Detect use_deformable: look for deform conv keys in attention modules
        has_deformable = any("deform" in k.lower() or "offset" in k.lower() for k in gen_state)
        detected["use_deformable"] = has_deformable

    # Detect use_pyramid_attention from pyramid_attention keys
    has_pyramid = any(k.startswith("pyramid_attention.") for k in gen_state)
    detected["use_pyramid_attention"] = has_pyramid

    # Detect upscale_factor from upscaler weights
    if "upscaler.pre_conv.weight" in gen_state:
        # For 2x: out_channels = embed_dim * 4 (for conv_first) or 3 * 4 (for output)
        out_ch = gen_state["upscaler.pre_conv.weight"].shape[0]
        # Check if it's the feature upscaler or output upscaler
        if out_ch in [12, 48, 192, 576]:  # 12=3*4(2x), 48=3*16(4x), 192=48*4(2x), 576=144*16(4x)
            if out_ch == 12 or out_ch == 192:
                detected["upscale_factor"] = 2
            elif out_ch == 48 or out_ch == 576:
                detected["upscale_factor"] = 4

    return detected


def load_models(args, config, device):
    """
    Load generator and OCR models from checkpoint.

    Auto-detects architecture from checkpoint keys to avoid mismatches
    between config and what was actually trained.

    Returns:
        (generator, ocr) – generator may be None if --ocr-only
    """
    model_config = config.get("model", {})
    ocr_config = config.get("ocr", {})

    # ── Load checkpoint ──────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"  Checkpoint keys: {list(checkpoint.keys())}")
    if "stage" in checkpoint:
        print(f"  Stage: {checkpoint['stage']}")
    if "best_word_acc" in checkpoint:
        print(f"  Best word accuracy: {checkpoint['best_word_acc']:.4f}")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")

    # ── Generator ────────────────────────────────────────────────────
    generator = None
    skip_sr = args.ocr_only or args.no_sr

    if not skip_sr and "generator_state_dict" in checkpoint:
        gen_state = _strip_ddp_prefix(checkpoint["generator_state_dict"])

        # Auto-detect architecture from checkpoint (NOT from config!)
        detected = _detect_architecture_from_checkpoint(gen_state)
        print(f"  Auto-detected architecture: {detected}")

        # Check if SwinIR or RRDB architecture
        is_swinir = detected.get("architecture") == "swinir"

        if is_swinir:
            # SwinIR architecture
            embed_dim = detected.get("embed_dim", model_config.get("swinir_embed_dim", 144))
            num_rstb = detected.get("num_rstb", model_config.get("swinir_num_rstb", 8))
            upscale_factor = detected.get("upscale_factor", model_config.get("upscale_factor", 2))
            use_pyramid_attn = detected.get("use_pyramid_attention", model_config.get("use_pyramid_attention", True))

            print(f"  Creating SwinIR generator: embed_dim={embed_dim}, num_rstb={num_rstb}, "
                  f"upscale={upscale_factor}, pyramid_attn={use_pyramid_attn}")

            generator = Generator(
                in_channels=3,
                out_channels=3,
                embed_dim=embed_dim,
                num_rstb=num_rstb,
                upscale_factor=upscale_factor,
                use_pyramid_attention=use_pyramid_attn,
                pyramid_layout=model_config.get("pyramid_layout", "brazilian"),
            )
        else:
            # RRDB architecture (legacy support)
            num_features = detected.get("num_features", model_config.get("num_filters", 64))
            num_blocks = detected.get("num_blocks", model_config.get("num_rrdb_blocks", 16))
            upscale_factor = detected.get("upscale_factor", model_config.get("upscale_factor", 2))
            use_char_attn = detected.get("use_character_attention", model_config.get("use_character_attention", False))

            # Get additional parameters from detection
            use_enhanced_attn = detected.get("use_enhanced_attention", True)
            use_deformable = detected.get("use_deformable", True)
            use_pyramid = detected.get("use_pyramid_attention", False)

            print(f"  Creating RRDB Generator: features={num_features}, blocks={num_blocks}, "
                  f"upscale={upscale_factor}, enhanced_attn={use_enhanced_attn}, deformable={use_deformable}")

            from src.models.generator import Generator
            generator = Generator(
                in_channels=3,
                out_channels=3,
                num_features=num_features,
                num_blocks=num_blocks,
                num_layers_per_block=3,
                upscale_factor=upscale_factor,
                use_enhanced_attention=use_enhanced_attn,
                use_deformable=use_deformable,
                use_character_attention=use_char_attn,
            )

        # Load with strict=True to catch any mismatch
        try:
            generator.load_state_dict(gen_state, strict=True)
            print(f"  ✓ Generator loaded (strict=True, all weights matched)")
        except RuntimeError as e:
            print(f"  ⚠ Strict loading failed: {e}")
            print(f"  Falling back to strict=False...")
            result = generator.load_state_dict(gen_state, strict=False)
            if result.missing_keys:
                print(f"  ⚠ Missing keys ({len(result.missing_keys)}): "
                      f"{result.missing_keys[:5]}{'...' if len(result.missing_keys) > 5 else ''}")
            if result.unexpected_keys:
                print(f"  ⚠ Unexpected keys ({len(result.unexpected_keys)}): "
                      f"{result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}")

        generator = generator.to(device).eval()

        total_params = sum(p.numel() for p in generator.parameters())
        print(f"  Generator parameters: {total_params:,}")
    elif not skip_sr:
        print("Warning: No generator_state_dict in checkpoint. Running OCR-only mode.")

    # ── OCR ──────────────────────────────────────────────────────────
    # Load OCR weights from checkpoint (PARSeq only - SimpleCRNN removed)
    ocr_state = None
    ocr_source = None
    if "ocr_state_dict" in checkpoint:
        ocr_state = _strip_ddp_prefix(checkpoint["ocr_state_dict"])
        ocr_source = "ocr_state_dict"
    elif "model_state_dict" in checkpoint:
        ocr_state = _strip_ddp_prefix(checkpoint["model_state_dict"])
        ocr_source = "model_state_dict"

    # Detect if checkpoint has PARSeq OCR
    has_parseq = False
    if ocr_state is not None:
        # Check for PARSeq-specific keys
        parseq_keys = [k for k in ocr_state.keys() if 'parseq_system' in k]
        has_parseq = len(parseq_keys) > 0
        print(f"  Auto-detected OCR type: {'PARSeq' if has_parseq else 'Unknown'}")

    print("Creating OCR model...")
    ocr = OCRModel(
        pretrained_path=ocr_config.get("pretrained_path", "baudm/parseq-base"),
        vocab=ocr_config.get("vocab", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        max_length=ocr_config.get("max_length", 7),
        frozen=True,
    )

    if ocr_state is not None and has_parseq:
        try:
            ocr.load_state_dict(ocr_state, strict=True)
            print(f"  ✓ OCR loaded (strict=True) from {ocr_source}")
        except RuntimeError as e:
            print(f"  ⚠ OCR strict loading failed: {e}")
            print(f"  Falling back to strict=False...")
            result = ocr.load_state_dict(ocr_state, strict=False)
            if result.missing_keys:
                print(f"  ⚠ Missing keys: {result.missing_keys[:5]}")
            if result.unexpected_keys:
                print(f"  ⚠ Unexpected keys: {result.unexpected_keys[:5]}")
    else:
        print("  ⚠ WARNING: No PARSeq OCR weights in checkpoint! Using pretrained weights.")

    ocr = ocr.to(device).eval()

    return generator, ocr


def discover_tracks(data_root: str):
    """
    Discover all track folders and their images.

    Returns:
        List of (track_id, [image_paths]) sorted by track_id
    """
    data_root = Path(data_root)
    tracks = []

    for track_dir in sorted(data_root.iterdir()):
        if not track_dir.is_dir():
            continue
        track_id = track_dir.name  # e.g. "track_10005"

        # Collect all LR images (jpg or png)
        images = sorted(
            [p for p in track_dir.iterdir()
             if p.suffix.lower() in (".jpg", ".jpeg", ".png")
             and p.stem.startswith("lr-")]
        )

        if images:
            tracks.append((track_id, images))

    print(f"Discovered {len(tracks)} tracks with {sum(len(imgs) for _, imgs in tracks)} total images")
    return tracks


def load_and_preprocess(image_path: Path, lr_size=None, normalize=True, preserve_aspect=False):
    """
    Load a single image and convert to tensor.

    Args:
        image_path: Path to image file
        lr_size: Optional (H, W) to resize to
        normalize: If True, normalize to [-1, 1]
        preserve_aspect: If True, pad to target aspect ratio BEFORE resizing.
                         This avoids distorting characters when test images have
                         different aspect ratios than training data.

    Returns:
        Tensor of shape (3, H, W)
    """
    img = Image.open(image_path).convert("RGB")

    if lr_size is not None:
        target_h, target_w = lr_size
        target_ratio = target_h / target_w  # e.g. 34/62 = 0.548

        if preserve_aspect:
            # Pad image to match target aspect ratio before resizing
            # This preserves character shapes without distortion
            orig_w, orig_h = img.size
            orig_ratio = orig_h / orig_w

            if orig_ratio < target_ratio:
                # Image is too wide — pad height
                new_h = int(orig_w * target_ratio)
                pad_top = (new_h - orig_h) // 2
                pad_bottom = new_h - orig_h - pad_top
                import numpy as np
                img_arr = np.array(img)
                # Pad with edge pixels (better than black for plates)
                img_arr = np.pad(img_arr,
                                 ((pad_top, pad_bottom), (0, 0), (0, 0)),
                                 mode='edge')
                img = Image.fromarray(img_arr)
            elif orig_ratio > target_ratio:
                # Image is too tall — pad width
                new_w = int(orig_h / target_ratio)
                pad_left = (new_w - orig_w) // 2
                pad_right = new_w - orig_w - pad_left
                import numpy as np
                img_arr = np.array(img)
                img_arr = np.pad(img_arr,
                                 ((0, 0), (pad_left, pad_right), (0, 0)),
                                 mode='edge')
                img = Image.fromarray(img_arr)

        # Now resize to exact target size
        try:
            img = img.resize((target_w, target_h), Image.Resampling.BICUBIC)
        except AttributeError:
            img = img.resize((target_w, target_h), Image.BICUBIC)

    tensor = transforms.ToTensor()(img)  # [0, 1]

    if normalize:
        tensor = tensor * 2.0 - 1.0  # [-1, 1]

    return tensor


def indices_to_text(indices, ocr):
    """Convert a list of character indices to a plate text string."""
    text = ""
    for idx in indices:
        if 0 <= idx < ocr.blank_idx:
            char = ocr.tokenizer.idx_to_char.get(idx, "")
            if char in ocr.tokenizer.vocab:
                text += char
    return text


def format_aware_beam_search(logits_single, ocr, beam_width=5, length_norm=0.7):
    """
    CTC beam search for a single sample with format-aware re-scoring.

    Instead of returning only the top beam, returns all beam_width candidates,
    applies PlateFormatValidator.score_candidates() to re-score them with
    Brazilian/Mercosur format bonuses, then picks the best.

    Args:
        logits_single: (T, C) logits for one sample
        ocr: OCRModel instance (for blank_idx, tokenizer)
        beam_width: Number of beams
        length_norm: Length normalization exponent

    Returns:
        (best_text, confidence)
    """
    T, C = logits_single.shape
    log_probs = F.log_softmax(logits_single, dim=-1)
    blank_idx = ocr.model.blank_idx

    # ── Run beam search ──────────────────────────────────────────────
    # Each beam: (sequence_indices, log_prob, prev_idx, blank_count)
    beam = [([], 0.0, None, 0)]

    for t in range(T):
        new_beam = []
        for seq, log_prob, prev_idx, blank_count in beam:
            top_k_lp, top_k_idx = log_probs[t].topk(beam_width)

            for k in range(beam_width):
                idx = top_k_idx[k].item()
                prob = top_k_lp[k].item()

                new_seq = seq.copy()
                new_blank_count = blank_count

                if idx == blank_idx:
                    new_blank_count += 1
                    new_prev_idx = None
                elif prev_idx == idx and blank_count == 0:
                    continue  # CTC collapse
                else:
                    new_seq.append(idx)
                    new_blank_count = 0
                    new_prev_idx = idx

                new_beam.append((new_seq, log_prob + prob, new_prev_idx, new_blank_count))

        # Keep top beams by length-normalized score
        new_beam.sort(
            key=lambda x: x[1] / (max(len(x[0]), 1) ** length_norm),
            reverse=True,
        )
        beam = new_beam[:beam_width]

    # ── Convert all beam candidates to (text, score) ─────────────────
    candidates = []
    for seq, log_prob, _, _ in beam:
        text = indices_to_text(seq, ocr)
        # Length-normalized score
        norm_score = log_prob / (max(len(seq), 1) ** length_norm)
        candidates.append((text, norm_score))

    if not candidates:
        return "UNKNOWN", 0.0

    # ── Format-aware re-scoring ──────────────────────────────────────
    # PlateFormatValidator.score_candidates adds +2.0 bonus to valid
    # Brazilian (LLLNNNN) or Mercosur (LLLNLNN) formats
    candidates = PlateFormatValidator.score_candidates(candidates)

    best_text, best_score = candidates[0]

    # Also apply character-level format correction as a safety net
    best_text = PlateFormatValidator.correct(best_text)

    # ── Confidence from normalized log-probability ───────────────────
    # Convert log-prob score to [0, 1] confidence
    # Typical CTC log-probs per char are in [-0.01, -3.0] range
    import math
    confidence = math.exp(max(best_score, -20.0))  # Clamp to avoid underflow
    confidence = min(confidence, 1.0)

    return best_text, confidence


def predict_with_confidence(ocr, logits, beam_width=1):
    """
    Decode OCR logits with format-aware beam search and confidence scores.

    Args:
        ocr: OCRModel instance
        logits: (B, T, C) raw logits
        beam_width: Beam width (1=greedy, >1=beam search with format re-scoring)

    Returns:
        List of (text, confidence) tuples
    """
    B, T, C = logits.shape
    is_parseq = ocr.use_parseq  # Check if using PARSeq
    is_ctc = not is_parseq and hasattr(ocr.model, 'use_ctc') and ocr.model.use_ctc

    results = []

    if is_parseq:
        # ── PARSeq decoding (uses native tokenizer) ───────────────────
        probs = logits.softmax(-1)
        preds, _ = ocr._parseq_tokenizer.decode(probs)
        # Filter to uppercase + digits (our LP vocabulary)
        allowed = set(ocr.vocab)
        for b, pred in enumerate(preds):
            text = ''.join(c.upper() if c.upper() in allowed else '' for c in pred)
            text = text[:ocr.max_length]
            text = PlateFormatValidator.correct(text)
            # Confidence: average probability of predicted characters
            char_probs = []
            for i, c in enumerate(text):
                if i < T and c in ocr._parseq_tokenizer._stoi:
                    char_idx = ocr._parseq_tokenizer._stoi[c]
                    char_probs.append(probs[b, i, char_idx].item())
            confidence = sum(char_probs) / len(char_probs) if char_probs else 0.0
            results.append((text, confidence))

    elif is_ctc and beam_width > 1:
        # ── Format-aware beam search (best quality) ──────────────────
        for b in range(B):
            text, confidence = format_aware_beam_search(
                logits[b], ocr, beam_width=beam_width
            )
            results.append((text, confidence))

    elif is_ctc:
        # ── Greedy CTC decoding (fast) ───────────────────────────────
        probs = F.softmax(logits, dim=-1)
        decoded_lists = ocr.model.ctc_decode_greedy(logits)

        for b in range(B):
            decoded_indices = decoded_lists[b]
            text = ""
            char_probs = []

            for idx in decoded_indices:
                if 0 <= idx < ocr.blank_idx:
                    char = ocr.tokenizer.idx_to_char.get(idx, "")
                    if char in ocr.tokenizer.vocab:
                        text += char
                        max_prob = probs[b, :, idx].max().item()
                        char_probs.append(max_prob)

            text = PlateFormatValidator.correct(text)

            if char_probs:
                import math
                log_conf = sum(math.log(max(p, 1e-10)) for p in char_probs) / len(char_probs)
                confidence = math.exp(log_conf)
            else:
                confidence = 0.0

            results.append((text, confidence))
    else:
        # ── Argmax decoding (non-CTC models) ─────────────────────────
        pred_indices = logits.argmax(dim=-1)
        max_probs = F.softmax(logits, dim=-1).max(dim=-1).values

        for b in range(B):
            text = ocr.tokenizer.decode(pred_indices[b])
            text = PlateFormatValidator.correct(text)
            valid_len = min(len(text), T)
            if valid_len > 0:
                confidence = max_probs[b, :valid_len].mean().item()
            else:
                confidence = 0.0
            results.append((text, confidence))

    return results


def aggregate_track_predictions(predictions):
    """
    Aggregate predictions from multiple images of the same track.

    Uses majority voting on each character position, weighted by confidence.

    Args:
        predictions: List of (text, confidence) tuples from different images

    Returns:
        (final_text, final_confidence)
    """
    if not predictions:
        return "UNKNOWN", 0.0

    if len(predictions) == 1:
        return predictions[0]

    # ── Strategy: weighted majority voting per character position ────
    max_len = max(len(text) for text, _ in predictions)
    final_chars = []
    final_confs = []

    for pos in range(max_len):
        # Collect characters at this position with their confidence
        char_weights = defaultdict(float)
        for text, conf in predictions:
            if pos < len(text):
                char_weights[text[pos]] += conf

        if char_weights:
            best_char = max(char_weights, key=char_weights.get)
            # Confidence for this position = weighted proportion
            total_weight = sum(char_weights.values())
            pos_conf = char_weights[best_char] / total_weight if total_weight > 0 else 0.0
            final_chars.append(best_char)
            final_confs.append(pos_conf)

    final_text = "".join(final_chars)

    # Also consider full-plate voting (most common full prediction)
    full_plate_votes = Counter()
    full_plate_conf = defaultdict(list)
    for text, conf in predictions:
        full_plate_votes[text] += 1
        full_plate_conf[text].append(conf)

    # If a single prediction appears in majority of images, prefer it
    most_common_text, most_common_count = full_plate_votes.most_common(1)[0]
    if most_common_count >= len(predictions) / 2:
        final_text = most_common_text
        final_confs = full_plate_conf[most_common_text]

    # Apply format correction
    final_text = PlateFormatValidator.correct(final_text)

    # Overall confidence
    if final_confs:
        final_confidence = sum(final_confs) / len(final_confs)
    else:
        final_confidence = 0.0

    return final_text, final_confidence


def tta_augment_batch(images: torch.Tensor) -> list:
    """
    Create TTA augmented versions of a batch of images.

    Augmentations that preserve text readability:
    - Original (identity)
    - Brightness +/- 10%
    - Contrast +/- 15%
    - Slight scale (0.95x and 1.05x with padding)

    Args:
        images: (B, 3, H, W) tensor in [-1, 1] range

    Returns:
        List of (augmented_batch, weight) tuples. Each augmented_batch
        has the same shape as input. Weight indicates confidence in
        this augmentation (1.0 = original, <1.0 = augmented).
    """
    augmented = [(images, 1.0)]  # Original always included with highest weight

    # Convert to [0, 1] for augmentations
    imgs_01 = (images + 1.0) / 2.0

    # Brightness variations
    for delta in [-0.08, 0.08]:
        bright = torch.clamp(imgs_01 + delta, 0, 1)
        augmented.append((bright * 2.0 - 1.0, 0.8))

    # Contrast variations
    mean = imgs_01.mean(dim=(2, 3), keepdim=True)
    for factor in [0.85, 1.15]:
        contrast = torch.clamp(mean + (imgs_01 - mean) * factor, 0, 1)
        augmented.append((contrast * 2.0 - 1.0, 0.8))

    # Slight scale variations (zoom in/out by 5%)
    B, C, H, W = images.shape
    for scale in [0.95, 1.05]:
        new_h, new_w = int(H * scale), int(W * scale)
        if new_h >= 4 and new_w >= 4:  # Safety check
            scaled = F.interpolate(images, size=(new_h, new_w), mode='bilinear', align_corners=False)
            # Pad/crop back to original size
            if scale < 1.0:
                pad_h = H - new_h
                pad_w = W - new_w
                scaled = F.pad(scaled, (pad_w // 2, pad_w - pad_w // 2,
                                        pad_h // 2, pad_h - pad_h // 2), mode='reflect')
            else:
                crop_h = new_h - H
                crop_w = new_w - W
                scaled = scaled[:, :, crop_h // 2:crop_h // 2 + H, crop_w // 2:crop_w // 2 + W]
            augmented.append((scaled, 0.7))

    return augmented


def tta_predict(generator, ocr, images, beam_width=5):
    """
    Run TTA: augment images, get logits from each, average logits, then decode.

    Logit-level fusion is much more powerful than text-level majority voting
    because it combines information before the argmax bottleneck.

    Args:
        generator: SR generator (or None for OCR-only)
        ocr: OCR model
        images: (B, 3, H, W) input tensor
        beam_width: Beam width for final decoding

    Returns:
        List of (text, confidence) tuples
    """
    augmented_batches = tta_augment_batch(images)
    all_logits = []
    all_weights = []

    for aug_images, weight in augmented_batches:
        if generator is not None:
            sr = generator(aug_images)
        else:
            sr = aug_images
        logits = ocr(sr, return_logits=True)
        all_logits.append(logits)
        all_weights.append(weight)

    # Weighted average of logits (soft fusion before argmax)
    total_weight = sum(all_weights)
    fused_logits = sum(l * w for l, w in zip(all_logits, all_weights)) / total_weight

    # Decode the fused logits
    return predict_with_confidence(ocr, fused_logits, beam_width=beam_width)


@torch.no_grad()
def run_diagnose(args):
    """
    Diagnostic mode: compare strategies, print samples, find pipeline bugs.

    Tests:
    1. Image size statistics across all test images
    2. Compare: resize(34x62) vs native resolution vs OCR-only
    3. Print first 10 track predictions for manual inspection
    4. If --diagnose-val given: reproduce training validation accuracy
    """
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    generator, ocr = load_models(args, config, device)
    data_config = config.get("data", {})
    config_lr_size = tuple(data_config.get("lr_size", [34, 62]))

    tracks = discover_tracks(args.data_root)
    if not tracks:
        print("No tracks found!")
        return

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: Image size statistics
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TEST 1: Image Size Statistics")
    print(f"{'='*60}")
    sizes = []
    for _, image_paths in tracks[:100]:  # Sample first 100 tracks
        for p in image_paths:
            img = Image.open(p)
            sizes.append((img.height, img.width))

    heights = [s[0] for s in sizes]
    widths = [s[1] for s in sizes]
    ratios = [h / w for h, w in sizes]
    unique_sizes = set(sizes)

    print(f"  Sampled {len(sizes)} images from {min(100, len(tracks))} tracks")
    print(f"  Height range: {min(heights)} - {max(heights)} (mean={sum(heights)/len(heights):.1f})")
    print(f"  Width range:  {min(widths)} - {max(widths)} (mean={sum(widths)/len(widths):.1f})")
    print(f"  Aspect ratio: {min(ratios):.3f} - {max(ratios):.3f} (mean={sum(ratios)/len(ratios):.3f})")
    print(f"  Unique sizes: {len(unique_sizes)}")
    if len(unique_sizes) <= 20:
        from collections import Counter
        size_counts = Counter(sizes).most_common(20)
        for (h, w), count in size_counts:
            print(f"    {h}×{w} (ratio {h/w:.3f}): {count} images")
    print(f"  Config lr_size: {config_lr_size[0]}×{config_lr_size[1]} (ratio {config_lr_size[0]/config_lr_size[1]:.3f})")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: Compare strategies on first 20 tracks
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("TEST 2: Strategy Comparison (first 20 tracks)")
    print(f"{'='*60}")

    test_tracks = tracks[:20]

    strategies = {}

    # Strategy A: Resize to config lr_size (current default)
    print("\n  Running Strategy A: resize to config lr_size...")
    strat_a = {}
    for track_id, image_paths in test_tracks:
        preds = []
        tensors = [load_and_preprocess(p, lr_size=config_lr_size) for p in image_paths]
        batch = torch.stack(tensors).to(device)
        if generator is not None:
            sr = generator(batch)
        else:
            sr = batch
        logits = ocr(sr, return_logits=True)
        preds = predict_with_confidence(ocr, logits, beam_width=args.beam_width)
        text, conf = aggregate_track_predictions(preds)
        strat_a[track_id] = (text, conf)
    strategies["A: resize(34×62)+SR"] = strat_a

    # Strategy B: No resize (native resolution) + SR
    print("  Running Strategy B: native resolution + SR...")
    strat_b = {}
    for track_id, image_paths in test_tracks:
        tensors = [load_and_preprocess(p, lr_size=None) for p in image_paths]
        # Pad to same size within track
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)
        # Make divisible by 2 for PixelShuffle
        max_h = max_h + (max_h % 2)
        max_w = max_w + (max_w % 2)
        padded = [F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1]), mode='reflect') for t in tensors]
        batch = torch.stack(padded).to(device)
        if generator is not None:
            sr = generator(batch)
        else:
            sr = batch
        logits = ocr(sr, return_logits=True)
        preds = predict_with_confidence(ocr, logits, beam_width=args.beam_width)
        text, conf = aggregate_track_predictions(preds)
        strat_b[track_id] = (text, conf)
    strategies["B: native+SR"] = strat_b

    # Strategy C: OCR only (no SR), resize to OCR input size
    print("  Running Strategy C: OCR only (no SR), resize to OCR-friendly size...")
    strat_c = {}
    for track_id, image_paths in test_tracks:
        # OCR internally resizes to 68×124; feed larger images
        tensors = [load_and_preprocess(p, lr_size=(68, 124)) for p in image_paths]
        batch = torch.stack(tensors).to(device)
        logits = ocr(batch, return_logits=True)
        preds = predict_with_confidence(ocr, logits, beam_width=args.beam_width)
        text, conf = aggregate_track_predictions(preds)
        strat_c[track_id] = (text, conf)
    strategies["C: OCR-only(68×124)"] = strat_c

    # Strategy D: OCR only, native resolution
    print("  Running Strategy D: OCR only, native resolution...")
    strat_d = {}
    for track_id, image_paths in test_tracks:
        tensors = [load_and_preprocess(p, lr_size=None) for p in image_paths]
        max_h = max(t.shape[1] for t in tensors)
        max_w = max(t.shape[2] for t in tensors)
        padded = [F.pad(t, (0, max_w - t.shape[2], 0, max_h - t.shape[1]), mode='reflect') for t in tensors]
        batch = torch.stack(padded).to(device)
        logits = ocr(batch, return_logits=True)
        preds = predict_with_confidence(ocr, logits, beam_width=args.beam_width)
        text, conf = aggregate_track_predictions(preds)
        strat_d[track_id] = (text, conf)
    strategies["D: OCR-only(native)"] = strat_d

    # Strategy E: No format correction (raw predictions)
    print("  Running Strategy E: SR + NO format correction...")
    strat_e = {}
    for track_id, image_paths in test_tracks:
        tensors = [load_and_preprocess(p, lr_size=config_lr_size) for p in image_paths]
        batch = torch.stack(tensors).to(device)
        if generator is not None:
            sr = generator(batch)
        else:
            sr = batch
        logits = ocr(sr, return_logits=True)

        # Decode WITHOUT format correction (raw predictions)
        # PARSeq: use native tokenizer without format correction
        probs = logits.softmax(-1)
        preds_list, _ = ocr._parseq_tokenizer.decode(probs)
        allowed = set(ocr.vocab)
        preds = []
        for pred in preds_list:
            text = ''.join(c.upper() if c.upper() in allowed else '' for c in pred)
            # NO PlateFormatValidator.correct() here
            preds.append((text[:ocr.max_length], 1.0))

        text, conf = aggregate_track_predictions(preds)
        strat_e[track_id] = (text, conf)
    strategies["E: SR+raw(no fmt fix)"] = strat_e

    # Strategy F: Aspect-ratio-preserving resize + SR
    print("  Running Strategy F: aspect-preserving resize + SR...")
    strat_f = {}
    for track_id, image_paths in test_tracks:
        tensors = [load_and_preprocess(p, lr_size=config_lr_size, preserve_aspect=True) for p in image_paths]
        batch = torch.stack(tensors).to(device)
        if generator is not None:
            sr = generator(batch)
        else:
            sr = batch
        logits = ocr(sr, return_logits=True)
        preds = predict_with_confidence(ocr, logits, beam_width=args.beam_width)
        text, conf = aggregate_track_predictions(preds)
        strat_f[track_id] = (text, conf)
    strategies["F: aspect-pad+SR"] = strat_f

    # Print comparison table
    print(f"\n  {'Track':<15}", end="")
    for name in strategies:
        print(f" | {name:<25}", end="")
    print()
    print("  " + "-" * (15 + 28 * len(strategies)))

    for track_id, _ in test_tracks:
        print(f"  {track_id:<15}", end="")
        for name, strat in strategies.items():
            text, conf = strat[track_id]
            print(f" | {text:<18} ({conf:.2f})", end="")
        print()

    # Agreement analysis
    print(f"\n  Strategy agreement:")
    strat_names = list(strategies.keys())
    for i, name_i in enumerate(strat_names):
        for j, name_j in enumerate(strat_names):
            if i >= j:
                continue
            agree = sum(
                1 for tid, _ in test_tracks
                if strategies[name_i][tid][0] == strategies[name_j][tid][0]
            )
            print(f"    {name_i} vs {name_j}: {agree}/{len(test_tracks)} agree")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: Reproduce training validation (if --diagnose-val given)
    # ═══════════════════════════════════════════════════════════════════
    if args.diagnose_val:
        print(f"\n{'='*60}")
        print("TEST 3: Reproduce Training Validation Accuracy")
        print(f"{'='*60}")

        from src.data.lp_dataset import LicensePlateDataset
        from torch.utils.data import DataLoader

        val_dataset = LicensePlateDataset(
            root_dir=args.diagnose_val,
            image_size=tuple(config_lr_size),
            augment=False,
        )

        if len(val_dataset) == 0:
            print("  No validation data found!")
        else:
            val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=0)

            word_correct = 0
            char_correct = 0
            total_chars = 0
            total_samples = 0
            mismatches = []

            for batch in tqdm(val_loader, desc="  Validating"):
                lr_images = batch["lr"].to(device)
                gt_texts = batch["plate_text"]

                if generator is not None:
                    sr_images = generator(lr_images)
                else:
                    sr_images = lr_images

                logits = ocr(sr_images, return_logits=True)
                preds = predict_with_confidence(ocr, logits, beam_width=args.beam_width)

                for (pred_text, conf), gt in zip(preds, gt_texts):
                    total_samples += 1
                    if pred_text == gt:
                        word_correct += 1
                    else:
                        if len(mismatches) < 20:
                            mismatches.append((gt, pred_text, conf))

                    for p, g in zip(pred_text, gt):
                        if p == g:
                            char_correct += 1
                        total_chars += 1

            word_acc = word_correct / total_samples if total_samples > 0 else 0
            char_acc = char_correct / total_chars if total_chars > 0 else 0

            # Load checkpoint to get stored accuracy
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            ckpt_word_acc = ckpt.get("best_word_acc", 0.0)

            print(f"\n  Inference on training data:")
            print(f"    Samples: {total_samples}")
            print(f"    Word accuracy: {word_acc:.4f} (checkpoint says {ckpt_word_acc:.4f})")
            print(f"    Char accuracy: {char_acc:.4f}")
            # Pipeline is OK if inference acc >= checkpoint acc (tested on full data including train split)
            # or within 10% tolerance (val split may differ)
            pipeline_ok = word_acc >= ckpt_word_acc * 0.9
            print(f"    Pipeline: {'OK ✓ (accuracy matches or exceeds checkpoint)' if pipeline_ok else 'POSSIBLE BUG ✗ — accuracy much lower than checkpoint!'}")
            if word_acc > ckpt_word_acc:
                print(f"    Note: Higher accuracy is expected since diagnose-val includes training data")

            if mismatches:
                print(f"\n  Sample mismatches (first 20):")
                print(f"    {'Ground Truth':<12} {'Prediction':<12} {'Conf':>6}  Diff")
                for gt, pred, conf in mismatches:
                    diff = "".join(
                        "^" if (i < len(pred) and i < len(gt) and pred[i] != gt[i]) else " "
                        for i in range(max(len(pred), len(gt)))
                    )
                    print(f"    {gt:<12} {pred:<12} {conf:>6.3f}  {diff}")

    print(f"\n{'='*60}")
    print("DIAGNOSIS COMPLETE")
    print(f"{'='*60}")


@torch.no_grad()
def run_inference(args):
    """Main inference pipeline."""
    # ── Run diagnostics if requested ─────────────────────────────────
    if args.diagnose:
        return run_diagnose(args)

    # ── Setup ────────────────────────────────────────────────────────
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    generator, ocr = load_models(args, config, device)

    # Data config
    data_config = config.get("data", {})

    # Determine resize behavior
    if args.lr_size is not None:
        lr_size = tuple(args.lr_size)
    elif args.no_resize:
        lr_size = None
    else:
        lr_size = tuple(data_config.get("lr_size", [34, 62]))

    # Discover test tracks
    tracks = discover_tracks(args.data_root)

    if not tracks:
        print(f"Error: No tracks found in {args.data_root}")
        return

    # ── Diagnostic: Check a sample image to detect size mismatch ─────
    sample_img = Image.open(tracks[0][1][0]).convert("RGB")
    native_w, native_h = sample_img.size
    print(f"\n  Sample test image size (native): {native_h}×{native_w} (H×W)")
    if lr_size is not None:
        print(f"  Will resize to: {lr_size[0]}×{lr_size[1]} (H×W)")
        if abs(native_h / native_w - lr_size[0] / lr_size[1]) > 0.1:
            print(f"  ⚠ Aspect ratio mismatch! Native={native_h/native_w:.2f}, "
                  f"Target={lr_size[0]/lr_size[1]:.2f}")
    else:
        print(f"  Using native resolution (no resize)")

    # ── Inference ────────────────────────────────────────────────────
    all_predictions = {}  # track_id -> (text, confidence)

    # Process tracks in batches
    print(f"\nRunning inference on {len(tracks)} tracks...")
    print(f"  SR: {'enabled' if generator is not None else 'disabled (OCR-only)'}")
    print(f"  Beam width: {args.beam_width}")
    print(f"  TTA: {'enabled (7 augmentations, logit-level fusion)' if args.tta else 'disabled'}")
    print(f"  LR size: {lr_size}")

    for track_id, image_paths in tqdm(tracks, desc="Processing tracks"):
        track_predictions = []

        # Process images in mini-batches
        for i in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[i:i + args.batch_size]

            # Load and preprocess images
            batch_tensors = []
            for img_path in batch_paths:
                tensor = load_and_preprocess(
                    img_path, lr_size=lr_size, normalize=True,
                    preserve_aspect=args.preserve_aspect,
                )
                batch_tensors.append(tensor)

            # If no resize, images may have different sizes — pad to largest
            if lr_size is None:
                max_h = max(t.shape[1] for t in batch_tensors)
                max_w = max(t.shape[2] for t in batch_tensors)
                padded = []
                for t in batch_tensors:
                    pad_h = max_h - t.shape[1]
                    pad_w = max_w - t.shape[2]
                    if pad_h > 0 or pad_w > 0:
                        t = F.pad(t, (0, pad_w, 0, pad_h), mode='reflect')
                    padded.append(t)
                batch_tensors = padded

            batch = torch.stack(batch_tensors).to(device)  # (B, 3, H, W)

            if args.tta:
                # TTA: logit-level fusion across augmented versions
                preds_with_conf = tta_predict(
                    generator, ocr, batch, beam_width=args.beam_width
                )
            else:
                # Standard: single forward pass
                if generator is not None:
                    sr_batch = generator(batch)
                else:
                    sr_batch = batch
                logits = ocr(sr_batch, return_logits=True)
                preds_with_conf = predict_with_confidence(ocr, logits, beam_width=args.beam_width)

            track_predictions.extend(preds_with_conf)

        # Aggregate predictions across all images of this track
        final_text, final_confidence = aggregate_track_predictions(track_predictions)
        all_predictions[track_id] = (final_text, final_confidence)

    # ── Write predictions.txt ────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.txt"
    with open(predictions_path, "w") as f:
        for track_id in sorted(all_predictions.keys()):
            text, confidence = all_predictions[track_id]
            f.write(f"{track_id},{text};{confidence:.4f}\n")

    print(f"\nPredictions written to: {predictions_path}")
    print(f"  Total tracks: {len(all_predictions)}")

    # ── Create submission.zip ────────────────────────────────────────
    zip_path = output_dir / "submission.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(predictions_path, "predictions.txt")

    print(f"Submission ZIP created: {zip_path}")

    # ── Summary statistics ───────────────────────────────────────────
    confidences = [conf for _, conf in all_predictions.values()]
    text_lengths = [len(text) for text, _ in all_predictions.values()]
    valid_format_count = sum(
        1 for text, _ in all_predictions.values()
        if PlateFormatValidator.validate(text)[0]
    )

    print(f"\n{'='*60}")
    print(f"INFERENCE SUMMARY")
    print(f"{'='*60}")
    print(f"  Tracks processed:    {len(all_predictions)}")
    print(f"  Avg confidence:      {sum(confidences)/len(confidences):.4f}")
    print(f"  Min confidence:      {min(confidences):.4f}")
    print(f"  Max confidence:      {max(confidences):.4f}")
    print(f"  Avg text length:     {sum(text_lengths)/len(text_lengths):.1f}")
    print(f"  Valid format plates:  {valid_format_count}/{len(all_predictions)} "
          f"({100*valid_format_count/len(all_predictions):.1f}%)")
    print(f"{'='*60}")

    return all_predictions


if __name__ == "__main__":
    args = parse_args()
    run_inference(args)

