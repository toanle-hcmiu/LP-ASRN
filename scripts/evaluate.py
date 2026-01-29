"""
Evaluation Script for LACD-SRN

Evaluates a trained super-resolution model on license plate data.
Reports:
- Image quality metrics (PSNR, SSIM)
- OCR accuracy (character-level and word-level)
"""

import argparse
import yaml
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.lp_dataset import LicensePlateDataset
from src.models.generator import Generator
from src.ocr.parseq_wrapper import ParseqOCR
from src.ocr.confusion_tracker import ConfusionTracker


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LACD-SRN")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/lacd_srnn.yaml")
    parser.add_argument("--data-root", type=str, default="data/train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-dir", type=str, default="results/evaluation")
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def compute_psnr(pred, target):
    """Compute PSNR between pred and target."""
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    return 20 * torch.log10(1.0 / torch.sqrt(mse))


def compute_ssim(pred, target, window_size=11):
    """Compute SSIM between pred and target."""
    # Convert from [-1, 1] to [0, 1]
    pred = (pred + 1.0) / 2.0
    target = (target + 1.0) / 2.0

    pred = torch.clamp(pred, 0, 1)
    target = torch.clamp(target, 0, 1)

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu_pred = F.avg_pool2d(pred, window_size, stride=1, padding=window_size // 2)
    mu_target = F.avg_pool2d(target, window_size, stride=1, padding=window_size // 2)

    mu_pred_sq = mu_pred ** 2
    mu_target_sq = mu_target ** 2
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.avg_pool2d(pred ** 2, window_size, stride=1, padding=window_size // 2) - mu_pred_sq
    sigma_target_sq = F.avg_pool2d(target ** 2, window_size, stride=1, padding=window_size // 2) - mu_target_sq
    sigma_pred_target = F.avg_pool2d(pred * target, window_size, stride=1, padding=window_size // 2) - mu_pred_target

    ssim_map = ((2 * mu_pred_target + C1) * (2 * sigma_pred_target + C2)) / (
        (mu_pred_sq + mu_target_sq + C1) * (sigma_pred_sq + sigma_target_sq + C2)
    )

    return ssim_map.mean().item()


def evaluate(args):
    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu")

    # Create generator
    model_config = config.get("model", {})
    generator = Generator(
        in_channels=3,
        out_channels=3,
        num_features=model_config.get("num_filters", 64),
        num_blocks=model_config.get("num_rrdb_blocks", 16),
        upscale_factor=model_config.get("upscale_factor", 2),
        use_deformable=model_config.get("use_deformable", True),
    ).to(device)

    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    # Create OCR model
    ocr_config = config.get("ocr", {})
    ocr = ParseqOCR(
        vocab=ocr_config.get("vocab", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        max_length=ocr_config.get("max_length", 7),
    ).to(device)
    ocr.eval()

    # Create dataset
    data_config = config.get("data", {})
    dataset = LicensePlateDataset(
        root_dir=args.data_root,
        image_size=tuple(data_config.get("lr_size", [17, 31])),
        augment=False,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    print(f"Evaluating on {len(dataset)} samples")

    # Evaluation metrics
    total_psnr = 0.0
    total_ssim = 0.0
    word_correct = 0
    char_correct = 0
    total_chars = 0

    # For confusion tracking
    confusion_tracker = ConfusionTracker(
        vocab=ocr_config.get("vocab", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    )

    # Results storage
    results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            lr_images = batch["lr"].to(device)
            hr_images = batch["hr"].to(device)
            gt_texts = batch["plate_text"]

            # Super-resolution
            sr_images = generator(lr_images)

            # Compute image quality metrics
            for i in range(sr_images.shape[0]):
                psnr = compute_psnr(sr_images[i:i+1], hr_images[i:i+1]).item()
                ssim = compute_ssim(sr_images[i:i+1], hr_images[i:i+1])
                total_psnr += psnr
                total_ssim += ssim

            # OCR predictions
            sr_texts = ocr.predict(sr_images)

            # Collect results
            for lr_path, hr_path, sr_text, gt_text in zip(
                batch["lr_path"], batch["hr_path"], sr_texts, gt_texts
            ):
                results.append({
                    "lr_path": lr_path,
                    "hr_path": hr_path,
                    "gt_text": gt_text,
                    "sr_text": sr_text,
                    "word_correct": sr_text == gt_text,
                })

            # Update accuracy metrics
            for sr_text, gt_text in zip(sr_texts, gt_texts):
                if sr_text == gt_text:
                    word_correct += 1

                # Character accuracy
                for s_char, g_char in zip(sr_text, gt_text):
                    if s_char == g_char:
                        char_correct += 1
                    total_chars += 1

            # Update confusion tracker
            confusion_tracker.update(sr_texts, gt_texts)

    # Compute averages
    avg_psnr = total_psnr / len(dataset)
    avg_ssim = total_ssim / len(dataset)
    word_acc = word_correct / len(dataset)
    char_acc = char_correct / total_chars if total_chars > 0 else 0

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"PSNR: {avg_psnr:.2f} dB")
    print(f"SSIM: {avg_ssim:.4f}")
    print(f"Character Accuracy: {char_acc:.4f}")
    print(f"Word Accuracy (Full Plate): {word_acc:.4f}")
    print("=" * 60)

    # Print confusion report
    print("\n" + confusion_tracker.get_report())

    # Save results
    output = {
        "checkpoint": args.checkpoint,
        "num_samples": len(dataset),
        "psnr": avg_psnr,
        "ssim": avg_ssim,
        "char_accuracy": char_acc,
        "word_accuracy": word_acc,
    }

    with open(save_dir / "metrics.json", "w") as f:
        json.dump(output, f, indent=2)

    # Save detailed results
    with open(save_dir / "detailed_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save confusion report
    with open(save_dir / "confusion_report.txt", "w") as f:
        f.write(confusion_tracker.get_report())

    print(f"\nResults saved to {save_dir}")

    return output


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
