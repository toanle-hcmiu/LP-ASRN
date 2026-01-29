#!/usr/bin/env python
"""
Progressive Training Script for LACD-SRN

Implements three-stage progressive training with automatic TensorBoard startup.

Usage:
    python scripts/train_progressive.py --stage all --config configs/lacd_srnn.yaml

Stages:
    Stage 1 (warmup): L1 loss only, 5-10 epochs
    Stage 2 (lcofl): Full LCOFL training, 50+ epochs
    Stage 3 (finetune): Joint OCR optimization, 20+ epochs
"""

import argparse
import yaml
import signal
import subprocess
import sys
import time
from pathlib import Path
from threading import Thread

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.progressive_trainer import ProgressiveTrainer, TrainingStage
from src.data.lp_dataset import create_dataloaders
from src.models.generator import Generator
from src.ocr.parseq_wrapper import ParseqOCR
from src.utils.logger import TensorBoardLogger


class TensorBoardLauncher:
    """Manages TensorBoard process in a separate thread."""

    def __init__(self, log_dir: str, port: int = 6006):
        """
        Initialize TensorBoard launcher.

        Args:
            log_dir: TensorBoard log directory
            port: Port to run TensorBoard on
        """
        self.log_dir = log_dir
        self.port = port
        self.process = None
        self.thread = None

    def start(self):
        """Start TensorBoard in a background thread."""
        self.thread = Thread(target=self._run_tensorboard, daemon=True)
        self.thread.start()

        # Wait a bit for TensorBoard to start
        time.sleep(2)

        if self.process.poll() is not None:
            print(f"Warning: TensorBoard failed to start. Process exited with code {self.process.returncode}")
            return False

        print(f"\n{'='*60}")
        print(f"TensorBoard running at: http://localhost:{self.port}")
        print(f"Log directory: {self.log_dir}")
        print(f"{'='*60}\n")

        return True

    def _run_tensorboard(self):
        """Run TensorBoard process."""
        try:
            cmd = [
                sys.executable, "-m", "tensorboard.main",
                "--logdir", self.log_dir,
                "--port", str(self.port),
                "--host", "0.0.0.0",
            ]

            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )

            # Wait for process to complete
            self.process.wait()

        except Exception as e:
            print(f"Error starting TensorBoard: {e}")

    def stop(self):
        """Stop TensorBoard process."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LACD-SRN with progressive training"
    )
    parser.add_argument("--config", type=str, default="configs/lacd_srnn.yaml")
    parser.add_argument("--data-root", type=str, default="data/train")
    parser.add_argument("--resume", type=str, default=None)

    # Stage selection
    parser.add_argument(
        "--stage",
        type=str,
        choices=["1", "2", "3", "all", "warmup", "lcofl", "finetune"],
        default="all",
        help="Training stage to run",
    )

    # Epoch overrides
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--lcofl-epochs", type=int, default=None)
    parser.add_argument("--finetune-epochs", type=int, default=None)

    # TensorBoard
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        default=True,
        help="Enable TensorBoard logging (default: enabled)",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_false",
        dest="tensorboard",
        help="Disable TensorBoard",
    )
    parser.add_argument("--tb-port", type=int, default=6007, help="TensorBoard port")
    parser.add_argument("--tb-dir", type=str, default="logs/tensorboard")

    # Training overrides
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--debug", action="store_true")

    return parser.parse_args()


def load_config(config_path: str, args) -> dict:
    """Load and update configuration from file and command-line arguments."""
    config_path = Path(config_path)

    if config_path.exists():
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config file {config_path} not found. Using defaults.")
        config = {}

    # Apply command-line overrides
    if args.batch_size:
        config.setdefault("data", {})["batch_size"] = args.batch_size

    # Configure progressive training if not in config
    if "progressive_training" not in config:
        config["progressive_training"] = {
            "enabled": True,
            "stage1": {"epochs": 10, "lr": 1e-4},
            "stage2": {"epochs": 50, "lr": 1e-4},
            "stage3": {"epochs": 20, "lr": 1e-5},
        }

    # Apply epoch overrides
    if args.warmup_epochs:
        config["progressive_training"]["stage1"]["epochs"] = args.warmup_epochs
    if args.lcofl_epochs:
        config["progressive_training"]["stage2"]["epochs"] = args.lcofl_epochs
    if args.finetune_epochs:
        config["progressive_training"]["stage3"]["epochs"] = args.finetune_epochs

    # Configure TensorBoard
    config["tensorboard"] = {
        "enabled": args.tensorboard,
        "log_dir": args.tb_dir,
    }

    # Set device
    config["training"] = config.get("training", {})
    config["training"]["device"] = args.device

    # Debug mode
    if args.debug:
        config["training"]["epochs"] = 1
        config["progressive_training"]["stage1"]["epochs"] = 1
        config["progressive_training"]["stage2"]["epochs"] = 1
        config["progressive_training"]["stage3"]["epochs"] = 1
        config["data"]["num_workers"] = 0

    return config


def map_stage_name(stage: str) -> TrainingStage:
    """Map string stage name to TrainingStage enum."""
    stage_map = {
        "1": TrainingStage.WARMUP,
        "warmup": TrainingStage.WARMUP,
        "2": TrainingStage.LCOFL,
        "lcofl": TrainingStage.LCOFL,
        "3": TrainingStage.FINETUNE,
        "finetune": TrainingStage.FINETUNE,
        "all": None,
    }
    return stage_map.get(stage.lower())


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config, args)

    # Determine device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # TensorBoard launcher
    tb_launcher = None
    if config["tensorboard"]["enabled"]:
        tb_launcher = TensorBoardLauncher(
            log_dir=config["tensorboard"]["log_dir"],
            port=args.tb_port,
        )
        tb_launcher.start()

        # Handle Ctrl+C gracefully
        def cleanup_handler(signum, frame):
            print("\nStopping training...")
            if tb_launcher:
                tb_launcher.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)

    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader = create_dataloaders(
        root_dir=args.data_root,
        batch_size=config["data"].get("batch_size", 16),
        num_workers=config["data"].get("num_workers", 4),
        image_size=tuple(config["data"].get("lr_size", [17, 31])),
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create generator
    print("\nCreating generator...")
    model_config = config.get("model", {})
    generator = Generator(
        num_features=model_config.get("num_filters", 64),
        num_blocks=model_config.get("num_rrdb_blocks", 16),
        upscale_factor=model_config.get("upscale_factor", 2),
        use_deformable=model_config.get("use_deformable", True),
    )

    # Count parameters
    total_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator parameters: {total_params:,}")

    # Create OCR
    print("\nCreating OCR model...")
    ocr_config = config.get("ocr", {})
    ocr = ParseqOCR(
        vocab=ocr_config.get("vocab", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        max_length=ocr_config.get("max_length", 7),
    )

    # Load fine-tuned OCR if available
    if ocr_config.get("finetuned_path") and Path(ocr_config["finetuned_path"]).exists():
        print(f"Loading fine-tuned OCR from {ocr_config['finetuned_path']}")
        ocr.load(ocr_config["finetuned_path"])

    # Create logger
    logger = TensorBoardLogger(log_dir=config["tensorboard"]["log_dir"])

    # Create trainer
    trainer = ProgressiveTrainer(
        generator=generator,
        ocr=ocr,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
        device=device,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Run training
    if args.stage.lower() == "all":
        results = trainer.train_full_progressive()
    else:
        stage = map_stage_name(args.stage)
        if stage is None:
            print(f"Invalid stage: {args.stage}")
            return

        final_acc = trainer.train_stage(stage)
        results = {"final_acc": final_acc}

    # Close logger
    logger.close()

    # Stop TensorBoard
    if tb_launcher:
        print("\nStopping TensorBoard...")
        tb_launcher.stop()

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
