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
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    print(f"Warning: Config file {config_path} not found. Using defaults.")
    return {}


def load_models(args, config, device):
    """
    Load generator and OCR models from checkpoint.

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
        print("Creating generator...")
        generator = Generator(
            in_channels=3,
            out_channels=3,
            num_features=model_config.get("num_filters", 64),
            num_blocks=model_config.get("num_rrdb_blocks", 16),
            upscale_factor=model_config.get("upscale_factor", 2),
            use_deformable=model_config.get("use_deformable", True),
            use_character_attention=model_config.get("use_character_attention", False),
            msca_scales=tuple(model_config.get("msca_scales", [1.0, 0.5, 0.25])),
            msca_num_prototypes=model_config.get("msca_num_prototypes", 36),
        )

        # Handle DDP-wrapped state dicts (keys prefixed with "module.")
        gen_state = checkpoint["generator_state_dict"]
        gen_state = {k.replace("module.", ""): v for k, v in gen_state.items()}
        generator.load_state_dict(gen_state, strict=False)
        generator = generator.to(device).eval()

        total_params = sum(p.numel() for p in generator.parameters())
        print(f"  Generator loaded ({total_params:,} params)")
    elif not skip_sr:
        print("Warning: No generator_state_dict in checkpoint. Running OCR-only mode.")

    # ── OCR ──────────────────────────────────────────────────────────
    print("Creating OCR model...")
    ocr = OCRModel(
        vocab=ocr_config.get("vocab", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        max_length=ocr_config.get("max_length", 7),
        frozen=True,
        rnn_dropout=ocr_config.get("rnn_dropout", 0.3),
        use_parseq=ocr_config.get("use_pretrained", False),
    )

    # Load OCR weights from checkpoint if available
    if "ocr_state_dict" in checkpoint:
        ocr_state = checkpoint["ocr_state_dict"]
        ocr_state = {k.replace("module.", ""): v for k, v in ocr_state.items()}
        ocr.load_state_dict(ocr_state, strict=False)
        print("  OCR loaded from checkpoint (ocr_state_dict)")
    elif "model_state_dict" in checkpoint:
        # OCR-only checkpoint (from finetune_ocr.py)
        ocr_state = checkpoint["model_state_dict"]
        ocr_state = {k.replace("module.", ""): v for k, v in ocr_state.items()}
        ocr.load_state_dict(ocr_state, strict=False)
        print("  OCR loaded from checkpoint (model_state_dict)")
    else:
        print("  Warning: No OCR weights in checkpoint. Using randomly initialized OCR.")

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


def load_and_preprocess(image_path: Path, lr_size=None, normalize=True):
    """
    Load a single image and convert to tensor.

    Args:
        image_path: Path to image file
        lr_size: Optional (H, W) to resize to
        normalize: If True, normalize to [-1, 1]

    Returns:
        Tensor of shape (3, H, W)
    """
    img = Image.open(image_path).convert("RGB")

    if lr_size is not None:
        h, w = lr_size
        try:
            img = img.resize((w, h), Image.Resampling.BICUBIC)
        except AttributeError:
            img = img.resize((w, h), Image.BICUBIC)

    tensor = transforms.ToTensor()(img)  # [0, 1]

    if normalize:
        tensor = tensor * 2.0 - 1.0  # [-1, 1]

    return tensor


def predict_with_confidence(ocr, logits):
    """
    Decode OCR logits and compute per-prediction confidence.

    Args:
        ocr: OCRModel instance
        logits: (B, T, C) raw logits

    Returns:
        List of (text, confidence) tuples
    """
    B, T, C = logits.shape
    is_ctc = isinstance(ocr.model, type(ocr.model)) and hasattr(ocr.model, 'use_ctc') and ocr.model.use_ctc

    # Softmax probabilities
    probs = F.softmax(logits, dim=-1)  # (B, T, C)

    results = []

    if is_ctc:
        # CTC decoding
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
                        # Get max prob for this character across timesteps
                        # Use the max probability of this index across all timesteps
                        max_prob = probs[b, :, idx].max().item()
                        char_probs.append(max_prob)

            # Apply plate format correction
            text = PlateFormatValidator.correct(text)

            # Confidence = geometric mean of character probabilities
            if char_probs:
                import math
                log_conf = sum(math.log(max(p, 1e-10)) for p in char_probs) / len(char_probs)
                confidence = math.exp(log_conf)
            else:
                confidence = 0.0

            results.append((text, confidence))
    else:
        # Argmax decoding
        pred_indices = logits.argmax(dim=-1)  # (B, T)
        max_probs = probs.max(dim=-1).values  # (B, T)

        for b in range(B):
            text = ocr.tokenizer.decode(pred_indices[b])
            text = PlateFormatValidator.correct(text)
            # Average confidence over non-padding positions
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


@torch.no_grad()
def run_inference(args):
    """Main inference pipeline."""
    # ── Setup ────────────────────────────────────────────────────────
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load models
    generator, ocr = load_models(args, config, device)

    # Data config
    data_config = config.get("data", {})
    lr_size = tuple(data_config.get("lr_size", [34, 62]))

    # Discover test tracks
    tracks = discover_tracks(args.data_root)

    if not tracks:
        print(f"Error: No tracks found in {args.data_root}")
        return

    # ── Inference ────────────────────────────────────────────────────
    all_predictions = {}  # track_id -> (text, confidence)

    # Process tracks in batches
    print(f"\nRunning inference on {len(tracks)} tracks...")
    print(f"  SR: {'enabled' if generator is not None else 'disabled (OCR-only)'}")
    print(f"  Beam width: {args.beam_width}")
    print(f"  LR size: {lr_size}")

    for track_id, image_paths in tqdm(tracks, desc="Processing tracks"):
        track_predictions = []

        # Process images in mini-batches
        for i in range(0, len(image_paths), args.batch_size):
            batch_paths = image_paths[i:i + args.batch_size]

            # Load and preprocess images
            batch_tensors = []
            for img_path in batch_paths:
                tensor = load_and_preprocess(img_path, lr_size=lr_size, normalize=True)
                batch_tensors.append(tensor)

            batch = torch.stack(batch_tensors).to(device)  # (B, 3, H, W)

            # Super-resolution
            if generator is not None:
                sr_batch = generator(batch)
            else:
                sr_batch = batch  # OCR on LR directly

            # OCR with logits for confidence
            logits = ocr(sr_batch, return_logits=True)
            preds_with_conf = predict_with_confidence(ocr, logits)

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

