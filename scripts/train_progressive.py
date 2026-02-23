#!/usr/bin/env python
"""
Progressive Training Script for LP-ASRN

Implements four-stage progressive training with automatic TensorBoard startup.

Usage:
    python scripts/train_progressive.py --stage all --config configs/lp_asrn.yaml

Stages:
    Stage 0 (pretrain): OCR pretraining on HR images, 150 epochs
    Stage 1 (warmup): L1 loss only, 30 epochs
    Stage 2 (lcofl): Full LCOFL training, 300 epochs
    Stage 3 (finetune): Joint OCR optimization, 150 epochs
"""

import argparse
import yaml
import signal
import subprocess
import sys
import time
import datetime
import os
from pathlib import Path
from threading import Thread

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.progressive_trainer import ProgressiveTrainer, TrainingStage
from src.data.lp_dataset import create_dataloaders
from src.models.generator import Generator
from src.ocr.ocr_model import OCRModel
from src.utils.logger import TensorBoardLogger, TextLogger


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
        description="Train LP-ASRN with progressive training"
    )
    parser.add_argument("--config", type=str, default="configs/lp_asrn.yaml")
    parser.add_argument("--data-root", type=str, default="data/train")
    parser.add_argument("--resume", type=str, default=None)

    # Stage selection
    parser.add_argument(
        "--stage",
        type=str,
        choices=["0", "1", "2", "3", "all", "pretrain", "warmup", "lcofl", "finetune"],
        default="all",
        help="Training stage to run",
    )

    # Epoch overrides
    parser.add_argument("--pretrain-epochs", type=int, default=None)
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
    parser.add_argument("--tb-dir", type=str, default=None, help="TensorBoard log directory (default: <save_dir>/logs)")
    parser.add_argument("--save-dir", type=str, default=None, help="Output directory for checkpoints and logs (auto-timestamped if None)")

    # Training overrides
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--gpus", type=str, default="0", help="GPU IDs to use for DDP (comma-separated, e.g., '0,1')")
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
            "stage0": {"epochs": 150, "lr": 1e-4},
            "stage1": {"epochs": 30, "lr": 1e-4},
            "stage2": {"epochs": 300, "lr": 1e-4},
            "stage3": {"epochs": 150, "lr": 1e-5},
        }

    # Apply epoch overrides
    if args.pretrain_epochs:
        config["progressive_training"]["stage0"]["epochs"] = args.pretrain_epochs
    if args.warmup_epochs:
        config["progressive_training"]["stage1"]["epochs"] = args.warmup_epochs
    if args.lcofl_epochs:
        config["progressive_training"]["stage2"]["epochs"] = args.lcofl_epochs
    if args.finetune_epochs:
        config["progressive_training"]["stage3"]["epochs"] = args.finetune_epochs

    # Configure single output directory with timestamped run folder
    # All outputs (checkpoints, logs, etc.) go into one folder
    if args.save_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.save_dir = f"outputs/run_{timestamp}"

    config.setdefault("training", {})["save_dir"] = args.save_dir

    # TensorBoard logs go into a subdirectory of the output folder
    if args.tb_dir is None:
        args.tb_dir = f"{args.save_dir}/logs"

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
        config["progressive_training"]["stage0"]["epochs"] = 1
        config["progressive_training"]["stage1"]["epochs"] = 1
        config["progressive_training"]["stage2"]["epochs"] = 1
        config["progressive_training"]["stage3"]["epochs"] = 1
        config["data"]["num_workers"] = 0

    return config


def map_stage_name(stage: str) -> TrainingStage:
    """Map string stage name to TrainingStage enum."""
    stage_map = {
        "0": TrainingStage.PRETRAIN,
        "pretrain": TrainingStage.PRETRAIN,
        "1": TrainingStage.WARMUP,
        "warmup": TrainingStage.WARMUP,
        "2": TrainingStage.LCOFL,
        "lcofl": TrainingStage.LCOFL,
        "3": TrainingStage.FINETUNE,
        "finetune": TrainingStage.FINETUNE,
        "all": None,
    }
    return stage_map.get(stage.lower())


def setup_ddp(rank, world_size):
    """Initialize DDP environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # NCCL settings for better error handling and longer timeout
    os.environ["NCCL_BLOCKING_WAIT"] = "1"  # Better error messages
    os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"  # Async error for debugging
    os.environ["NCCL_DEBUG"] = "WARN"  # Enable NCCL warnings for debugging

    # Additional NCCL settings to prevent timeouts
    os.environ["NCCL_P2P_LEVEL"] = "NVL"  # Use NVLink for faster P2P
    os.environ["NCCL_IB_DISABLE"] = "1"  # Disable InfiniBand for single-node
    os.environ["NCCL_SOCKET_NTHREADS"] = "4"  # More threads for socket comm
    os.environ["NCCL_NSOCKS_PERTHREAD"] = "4"  # More sockets per thread

    # Use longer timeout (120 minutes) to handle long validation phases and potential hangs
    import datetime
    timeout = datetime.timedelta(minutes=120)

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
        timeout=timeout  # Explicit timeout to prevent watchdog kills
    )
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup DDP environment."""
    dist.destroy_process_group()


def train_ddp(rank, world_size, args, config):
    """DDP training function for each process."""
    # Setup DDP
    setup_ddp(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Only rank 0 prints and logs
    is_main = rank == 0

    if is_main:
        print(f"[Rank {rank}] Using DDP with {world_size} GPUs")
        print(f"[Rank {rank}] Using device: {device}")

    # Enable CuDNN autotuner for A100 optimization
    torch.backends.cudnn.benchmark = True

    # TensorBoard launcher (only on rank 0)
    tb_launcher = None
    if is_main and config["tensorboard"]["enabled"]:
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
            cleanup_ddp()
            sys.exit(0)

        signal.signal(signal.SIGINT, cleanup_handler)
        signal.signal(signal.SIGTERM, cleanup_handler)

    # Create data loaders with DistributedSampler
    if is_main:
        print("\nLoading data...")

    from src.data.lp_dataset import create_dataloaders
    test_like_val = config.get("training", {}).get("test_like_val", False)
    dataloader_result = create_dataloaders(
        root_dir=args.data_root,
        batch_size=config["data"].get("batch_size", 16),
        num_workers=config["data"].get("num_workers", 4),
        image_size=tuple(config["data"].get("lr_size", [17, 31])),
        distributed=True,
        rank=rank,
        world_size=world_size,
        ocr_pretrain_mode=config["data"].get("ocr_pretrain_augmentation", False),
        aspect_ratio_augment=config["data"].get("aspect_ratio_augment", False),
        test_aspect_range=tuple(config["data"].get("test_aspect_range", [0.29, 0.40])),
        test_resolution_augment=config["data"].get("test_resolution_augment", False),
        test_resolution_prob=config["data"].get("test_resolution_prob", 0.5),
        jpeg_augment=config["data"].get("jpeg_augment", False),
        jpeg_quality_range=tuple(config["data"].get("jpeg_quality_range", [60, 95])),
        no_crop_prob=config["data"].get("no_crop_prob", 0.0),
        test_like_val=test_like_val,
        test_like_val_fraction=config.get("training", {}).get("test_like_val_fraction", 0.1),
    )
    # Unpack: 4-tuple when test_like_val=True, 3-tuple otherwise
    test_like_val_loader = None
    if test_like_val and len(dataloader_result) == 4:
        train_loader, val_loader, train_sampler, test_like_val_loader = dataloader_result
    else:
        train_loader, val_loader, train_sampler = dataloader_result

    if is_main:
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")

    # Create generator
    if is_main:
        print("\nCreating generator...")

    model_config = config.get("model", {})
    generator = Generator(
        in_channels=3,
        out_channels=3,
        num_features=model_config.get("num_features", 64),
        num_blocks=model_config.get("num_blocks", 12),
        num_layers_per_block=model_config.get("num_layers_per_block", 3),
        upscale_factor=model_config.get("upscale_factor", 2),
        use_enhanced_attention=model_config.get("use_enhanced_attention", True),
        use_deformable=model_config.get("use_deformable", False),
        use_autoencoder=True,
    )

    # Count parameters (only rank 0)
    if is_main:
        total_params = sum(p.numel() for p in generator.parameters())
        print(f"Generator parameters: {total_params:,}")

    # Create OCR
    if is_main:
        print("\nCreating OCR model...")

    ocr_config = config.get("ocr", {})
    ocr = OCRModel(
        pretrained_path=ocr_config.get("pretrained_path", "baudm/parseq-base"),
        vocab=ocr_config.get("vocab", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
        max_length=ocr_config.get("max_length", 7),
        frozen=ocr_config.get("freeze_ocr", True),
    )

    # Load fine-tuned OCR if available (all ranks need this)
    if ocr_config.get("finetuned_path") and Path(ocr_config["finetuned_path"]).exists():
        if is_main:
            print(f"Loading fine-tuned OCR from {ocr_config['finetuned_path']}")
        ocr.load(ocr_config["finetuned_path"])

    # Wrap models with DDP
    generator = nn.parallel.DistributedDataParallel(
        generator.to(device),
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True,  # For OCR frozen periods
    )

    # Only wrap OCR with DDP if it has trainable parameters
    # (OCR is frozen by default, so no need for DDP)
    has_trainable_params = any(p.requires_grad for p in ocr.parameters())
    if has_trainable_params:
        ocr_model = nn.parallel.DistributedDataParallel(
            ocr.to(device),
            device_ids=[rank],
            output_device=rank,
        )
        if is_main:
            print("OCR wrapped with DDP (has trainable parameters)")
    else:
        # OCR is frozen, no need for DDP wrapper
        ocr_model = ocr.to(device)
        if is_main:
            print("OCR not wrapped with DDP (frozen, no trainable parameters)")

    # Store the wrapped model reference (only for DDP-wrapped models)
    # This avoids circular reference when OCR is frozen
    if has_trainable_params:
        ocr.model = ocr_model

    # Create logger (only rank 0)
    logger = None
    text_logger = None
    if is_main:
        logger = TensorBoardLogger(log_dir=config["tensorboard"]["log_dir"])
        text_logger = TextLogger(log_dir=config["tensorboard"]["log_dir"], filename="training.log")

        # Log comprehensive system and configuration info
        text_logger.log_system_info()
        text_logger.log_model_summary("Generator", generator.module)
        text_logger.log_model_summary("OCR", ocr_model.module if has_trainable_params else ocr_model)
        text_logger.log_training_config(config)

        text_logger.info(f"DDP Training with {world_size} GPUs")

    # Create trainer
    trainer = ProgressiveTrainer(
        generator=generator,
        ocr=ocr_model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        logger=logger,
        device=device,
        distributed=True,
        rank=rank,
        world_size=world_size,
        train_sampler=train_sampler,
        test_like_val_loader=test_like_val_loader,
    )
    if is_main:
        trainer.set_text_logger(text_logger)

    # Resume from checkpoint if specified (ALL ranks must load for consistent state)
    if args.resume:
        if is_main:
            print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Synchronize all ranks before training
    dist.barrier()

    # Run training
    if args.stage.lower() == "all":
        results = trainer.train_full_progressive()
    else:
        stage = map_stage_name(args.stage)
        if stage is None:
            if is_main:
                print(f"Invalid stage: {args.stage}")
            return
        final_acc = trainer.train_stage(stage)
        results = {"final_acc": final_acc}

    # Close loggers (only rank 0)
    if is_main:
        logger.close()
        text_logger.close()

        # Stop TensorBoard
        if tb_launcher:
            print("\nStopping TensorBoard...")
            tb_launcher.stop()

        print("\nTraining complete!")

    # Cleanup DDP
    cleanup_ddp()


def find_latest_checkpoint(save_dir: str) -> str:
    """Find the latest checkpoint in the save directory."""
    save_path = Path(save_dir)
    if not save_path.exists():
        return None

    # Check for emergency_latest first (most recent crash recovery)
    emergency_path = save_path / "emergency_latest.pth"
    if emergency_path.exists():
        return str(emergency_path)

    # Check for mid-epoch checkpoints
    mid_epoch_files = list(save_path.glob("mid_epoch_*_step_*.pth"))
    if mid_epoch_files:
        # Sort by step number and get the latest
        latest = max(mid_epoch_files, key=lambda p: int(p.stem.split('_')[-1]))
        return str(latest)

    # Check for best.pth
    best_path = save_path / "best.pth"
    if best_path.exists():
        return str(best_path)

    return None


def main():
    args = parse_args()

    # Load configuration
    config = load_config(args.config, args)

    # Auto-resume from latest checkpoint if no explicit resume path and save_dir exists
    save_dir = Path(config["training"]["save_dir"])
    if not args.resume and save_dir.exists():
        latest_checkpoint = find_latest_checkpoint(config["training"]["save_dir"])
        if latest_checkpoint:
            print(f"\n{'='*60}")
            print(f"Found existing checkpoint: {latest_checkpoint}")
            print(f"Resume training? (Press Enter to resume, or 'n' to start fresh)")
            print(f"{'='*60}")
            # For automation, you can also set an environment variable
            if os.environ.get("LP_ASRN_AUTO_RESUME", "1") == "1":
                args.resume = latest_checkpoint
                print(f"Auto-resuming from: {latest_checkpoint}\n")
            else:
                response = input().strip().lower()
                if response != 'n':
                    args.resume = latest_checkpoint
                    print(f"Resuming from: {latest_checkpoint}\n")

    # Check if using DDP (multiple GPUs)
    gpu_ids = args.gpus.split(',')
    world_size = len(gpu_ids)

    if world_size > 1:
        # Use DDP for multi-GPU training
        print(f"Using DDP with {world_size} GPUs: {args.gpus}")
        mp.spawn(
            train_ddp,
            args=(world_size, args, config),
            nprocs=world_size,
            join=True
        )
    else:
        # Single GPU training (original code path)
        # Use the specified GPU ID from --gpus argument
        gpu_id = gpu_ids[0]
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(int(gpu_id))
        print(f"Using device: {device}")

        # Enable CuDNN autotuner for A100 optimization
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            print("CuDNN benchmark enabled for A100 optimization")

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
        test_like_val = config.get("training", {}).get("test_like_val", False)
        dataloader_result = create_dataloaders(
            root_dir=args.data_root,
            batch_size=config["data"].get("batch_size", 16),
            num_workers=config["data"].get("num_workers", 4),
            image_size=tuple(config["data"].get("lr_size", [17, 31])),
            distributed=False,
            ocr_pretrain_mode=config["data"].get("ocr_pretrain_augmentation", False),
            aspect_ratio_augment=config["data"].get("aspect_ratio_augment", False),
            test_aspect_range=tuple(config["data"].get("test_aspect_range", [0.29, 0.40])),
            test_resolution_augment=config["data"].get("test_resolution_augment", False),
            test_resolution_prob=config["data"].get("test_resolution_prob", 0.5),
            jpeg_augment=config["data"].get("jpeg_augment", False),
            jpeg_quality_range=tuple(config["data"].get("jpeg_quality_range", [60, 95])),
            no_crop_prob=config["data"].get("no_crop_prob", 0.0),
            test_like_val=test_like_val,
            test_like_val_fraction=config.get("training", {}).get("test_like_val_fraction", 0.1),
        )
        # Unpack: 4-tuple when test_like_val=True, 3-tuple otherwise
        test_like_val_loader = None
        if test_like_val and len(dataloader_result) == 4:
            train_loader, val_loader, _, test_like_val_loader = dataloader_result
        else:
            train_loader, val_loader, _ = dataloader_result

        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")

        # Create generator
        print("\nCreating generator...")
        model_config = config.get("model", {})
        generator = Generator(
            in_channels=3,
            out_channels=3,
            num_features=model_config.get("num_features", 64),
            num_blocks=model_config.get("num_blocks", 12),
            num_layers_per_block=model_config.get("num_layers_per_block", 3),
            upscale_factor=model_config.get("upscale_factor", 2),
            use_enhanced_attention=model_config.get("use_enhanced_attention", True),
            use_deformable=model_config.get("use_deformable", False),
            use_character_attention=model_config.get("use_character_attention", False),
            msca_scales=tuple(model_config.get("msca_scales", (1.0, 0.5, 0.25))),
            msca_num_prototypes=model_config.get("msca_num_prototypes", 36),
            use_autoencoder=True,
        )

        # Count parameters
        total_params = sum(p.numel() for p in generator.parameters())
        print(f"Generator parameters: {total_params:,}")

        # Create OCR
        print("\nCreating OCR model...")
        ocr_config = config.get("ocr", {})
        ocr = OCRModel(
            pretrained_path=ocr_config.get("pretrained_path", "baudm/parseq-base"),
            vocab=ocr_config.get("vocab", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            max_length=ocr_config.get("max_length", 7),
            frozen=ocr_config.get("freeze_ocr", True),
        )

        # Load fine-tuned OCR if available
        if ocr_config.get("finetuned_path") and Path(ocr_config["finetuned_path"]).exists():
            print(f"Loading fine-tuned OCR from {ocr_config['finetuned_path']}")
            ocr.load(ocr_config["finetuned_path"])

        # Create logger
        logger = TensorBoardLogger(log_dir=config["tensorboard"]["log_dir"])
        text_logger = TextLogger(log_dir=config["tensorboard"]["log_dir"], filename="training.log")

        # Log comprehensive system and configuration info
        text_logger.log_system_info()
        text_logger.log_model_summary("Generator", generator)
        text_logger.log_model_summary("OCR", ocr)
        text_logger.log_training_config(config)

        # Move models to device
        generator = generator.to(device)
        ocr = ocr.to(device)

        # Create trainer
        trainer = ProgressiveTrainer(
            generator=generator,
            ocr=ocr,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            logger=logger,
            device=device,
            test_like_val_loader=test_like_val_loader,
        )
        trainer.set_text_logger(text_logger)

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

        # Close loggers
        logger.close()
        text_logger.close()

        # Stop TensorBoard
        if tb_launcher:
            print("\nStopping TensorBoard...")
            tb_launcher.stop()

        print("\nTraining complete!")


if __name__ == "__main__":
    main()
