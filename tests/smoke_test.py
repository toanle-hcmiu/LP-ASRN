#!/usr/bin/env python
"""
Smoke Test for LP-ASRN Training Pipeline

Tests all modified components work end-to-end using SYNTHETIC data.
This avoids the slow dataset scan on local disk while testing the same code paths.

Run: python tests/smoke_test.py
"""

import sys
import os
import io
import json
import tempfile
import shutil
import traceback
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

# Tee stdout to a log file
class TeeWriter:
    def __init__(self, *writers):
        self.writers = writers
    def write(self, text):
        for w in self.writers:
            w.write(text)
            w.flush()
    def flush(self):
        for w in self.writers:
            w.flush()

log_file = open(Path(__file__).parent / "smoke_output.txt", "w", encoding="utf-8")
sys.stdout = TeeWriter(sys.__stdout__, log_file)
sys.stderr = TeeWriter(sys.__stderr__, log_file)

import torch
import yaml
import numpy as np
from PIL import Image

# Track test results
passed = []
failed = []


def test(name):
    """Decorator to track test results."""
    def decorator(func):
        def wrapper():
            print(f"\n{'='*60}")
            print(f"TEST: {name}")
            print(f"{'='*60}")
            try:
                func()
                print(f"  PASSED: {name}")
                passed.append(name)
            except Exception as e:
                print(f"  FAILED: {name}")
                print(f"    Error: {e}")
                traceback.print_exc()
                failed.append((name, str(e)))
        return wrapper
    return decorator


def create_synthetic_dataset(num_tracks=5):
    """
    Create a minimal synthetic dataset matching the real directory structure.
    Returns the temp dir path (caller must clean up).
    """
    tmp_dir = tempfile.mkdtemp(prefix="lp_asrn_test_")
    
    plates = ["ABC1234", "DEF5678", "GHI9012", "JKL3456", "MNO7890",
              "PQR1234", "STU5678", "VWX9012", "ABC4D56", "DEF7E89"]
    layouts = ["Brazilian", "Mercosur"]

    for scenario in ["Scenario-A", "Scenario-B"]:
        for layout in layouts:
            layout_dir = Path(tmp_dir) / scenario / layout
            layout_dir.mkdir(parents=True, exist_ok=True)

            for t in range(num_tracks):
                track_dir = layout_dir / f"track_{t:05d}"
                track_dir.mkdir(exist_ok=True)

                plate_text = plates[t % len(plates)]
                plate_layout = layout

                # Create corners (4 points for each image)
                corners = {}

                for i in range(1, 6):
                    lr_name = f"lr-{i:03d}.png"
                    hr_name = f"hr-{i:03d}.png"

                    # Create LR image (30x87 - typical train size)
                    lr_img = Image.fromarray(
                        np.random.randint(0, 255, (30, 87, 3), dtype=np.uint8)
                    )
                    lr_img.save(track_dir / lr_name)

                    # Create HR image (60x174 - 2x LR)
                    hr_img = Image.fromarray(
                        np.random.randint(0, 255, (60, 174, 3), dtype=np.uint8)
                    )
                    hr_img.save(track_dir / hr_name)

                    # Corners: dict with top-left, top-right, bottom-right, bottom-left
                    lr_w, lr_h = 87, 30
                    hr_w, hr_h = 174, 60
                    corners[lr_name] = {
                        "top-left": [2, 2],
                        "top-right": [lr_w - 2, 2],
                        "bottom-right": [lr_w - 2, lr_h - 2],
                        "bottom-left": [2, lr_h - 2],
                    }
                    corners[hr_name] = {
                        "top-left": [4, 4],
                        "top-right": [hr_w - 4, 4],
                        "bottom-right": [hr_w - 4, hr_h - 4],
                        "bottom-left": [4, hr_h - 4],
                    }

                annotations = {
                    "plate_text": plate_text,
                    "plate_layout": plate_layout,
                    "corners": corners,
                }
                with open(track_dir / "annotations.json", "w") as f:
                    json.dump(annotations, f)

    return tmp_dir


# ============================================================
# Test 1: Config loading
# ============================================================
@test("Config loading with new parameters")
def test_config():
    with open("configs/lp_asrn.yaml", "r") as f:
        config = yaml.safe_load(f)

    assert config["data"]["jpeg_augment"] is True
    assert config["data"]["jpeg_quality_range"] == [60, 95]
    assert config["data"]["no_crop_prob"] == 0.3
    assert config["training"]["test_like_val"] is True
    assert config["training"]["test_like_val_fraction"] == 0.1

    print(f"  Config OK: jpeg_augment={config['data']['jpeg_augment']}, "
          f"no_crop={config['data']['no_crop_prob']}, "
          f"test_like_val={config['training']['test_like_val']}")


# ============================================================
# Test 2: JPEGCompression class
# ============================================================
@test("JPEGCompression augmentation")
def test_jpeg_compression():
    from src.data.lp_dataset import JPEGCompression

    jpeg = JPEGCompression(quality_range=(60, 95))
    img = Image.fromarray(np.random.randint(0, 255, (34, 62, 3), dtype=np.uint8))
    compressed = jpeg(img)

    assert isinstance(compressed, Image.Image)
    assert compressed.size == img.size
    print(f"  Input size: {img.size}, Output size: {compressed.size}")


# ============================================================
# Test 3: Dataset creation with all new params (synthetic data)
# ============================================================
@test("Dataset creation with JPEG + no-crop + test-like-val")
def test_dataset_creation():
    from src.data.lp_dataset import create_dataloaders

    tmp_dir = create_synthetic_dataset(num_tracks=5)
    try:
        result = create_dataloaders(
            root_dir=tmp_dir,
            batch_size=4,
            num_workers=0,
            image_size=(34, 62),
            distributed=False,
            ocr_pretrain_mode=True,
            aspect_ratio_augment=True,
            test_aspect_range=(0.25, 0.45),
            test_resolution_augment=True,
            test_resolution_prob=0.7,
            jpeg_augment=True,
            jpeg_quality_range=(60, 95),
            no_crop_prob=0.3,
            test_like_val=True,
            test_like_val_fraction=0.5,  # Higher fraction for small dataset
        )

        assert len(result) == 4, f"Expected 4-tuple (test_like_val=True), got {len(result)}-tuple"
        train_loader, val_loader, _, test_like_val_loader = result

        print(f"  Train: {len(train_loader.dataset)} samples, {len(train_loader)} batches")
        print(f"  Val: {len(val_loader.dataset)} samples")
        print(f"  Test-like val: {len(test_like_val_loader.dataset)} samples")

        # Get one batch
        batch = next(iter(train_loader))
        assert "lr" in batch and "hr" in batch and "plate_text" in batch
        print(f"  Train batch: LR={batch['lr'].shape}, HR={batch['hr'].shape}")
        print(f"  Plate texts: {batch['plate_text']}")

        # Test-like val batch
        tl_batch = next(iter(test_like_val_loader))
        assert "lr" in tl_batch and "hr" in tl_batch
        print(f"  Test-like batch: LR={tl_batch['lr'].shape}, HR={tl_batch['hr'].shape}")

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# Test 4: Dataset creation WITHOUT test_like_val (3-tuple return)
# ============================================================
@test("Dataset creation without test-like-val (3-tuple)")
def test_dataset_creation_no_testlike():
    from src.data.lp_dataset import create_dataloaders

    tmp_dir = create_synthetic_dataset(num_tracks=3)
    try:
        result = create_dataloaders(
            root_dir=tmp_dir,
            batch_size=4,
            num_workers=0,
            image_size=(34, 62),
            distributed=False,
            jpeg_augment=True,
            jpeg_quality_range=(60, 95),
            no_crop_prob=0.3,
            test_like_val=False,
        )

        assert len(result) == 3, f"Expected 3-tuple, got {len(result)}-tuple"
        train_loader, val_loader, sampler = result
        assert sampler is None
        print(f"  3-tuple returned correctly (no test_like_val)")
        print(f"  Train: {len(train_loader.dataset)} samples")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# Test 5: Generator forward pass
# ============================================================
@test("Generator forward pass")
def test_generator():
    from src.models.generator import Generator

    with open("configs/lp_asrn.yaml", "r") as f:
        config = yaml.safe_load(f)
    model_config = config.get("model", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(
        in_channels=3, out_channels=3,
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
    ).to(device)

    total_params = sum(p.numel() for p in generator.parameters())
    print(f"  Generator parameters: {total_params:,}")

    lr = torch.randn(2, 3, 34, 62).to(device)
    with torch.no_grad():
        sr = generator(lr)
    assert sr.shape == (2, 3, 68, 124), f"Expected (2,3,68,124), got {sr.shape}"
    print(f"  Input: {lr.shape} -> Output: {sr.shape}")


# ============================================================
# Test 6: PARSeq OCR forward pass
# ============================================================
@test("PARSeq OCR forward + predict")
def test_ocr():
    from src.ocr.ocr_model import OCRModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ocr = OCRModel(
        pretrained_path="baudm/parseq-base",
        vocab="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        max_length=7, frozen=True,
    ).to(device)

    total_params = sum(p.numel() for p in ocr.parameters())
    print(f"  OCR parameters: {total_params:,}")

    hr = torch.randn(2, 3, 68, 124).to(device)
    with torch.no_grad():
        preds = ocr.predict(hr, beam_width=1)
    assert len(preds) == 2
    print(f"  Predictions: {preds}")


# ============================================================
# Test 7: PARSeq forward_train (Stage 0 training loss)
# ============================================================
@test("PARSeq forward_train loss computation")
def test_parseq_forward_train():
    from src.ocr.ocr_model import OCRModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ocr = OCRModel(
        pretrained_path="baudm/parseq-base",
        vocab="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        max_length=7, frozen=False,
    ).to(device)

    hr = torch.randn(4, 3, 68, 124).to(device)
    gt_texts = ["ABC1234", "DEF5678", "GHI9012", "JKL3456"]

    ocr.train()
    loss = ocr.forward_train(hr, gt_texts, label_smoothing=0.1)

    assert loss.requires_grad, "Loss should require grad for backprop"
    loss.backward()
    print(f"  forward_train loss: {loss.item():.4f}")
    print(f"  Grad computed: {any(p.grad is not None for p in ocr.parameters())}")


# ============================================================
# Test 8: _simulate_test_pipeline
# ============================================================
@test("_simulate_test_pipeline (Stage 0 degradation)")
def test_simulate_test_pipeline():
    from src.training.progressive_trainer import ProgressiveTrainer
    from src.models.generator import Generator
    from src.ocr.ocr_model import OCRModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("configs/lp_asrn.yaml", "r") as f:
        config = yaml.safe_load(f)

    generator = Generator(
        in_channels=3, out_channels=3, num_features=64,
        num_blocks=4, num_layers_per_block=2, upscale_factor=2,
        use_enhanced_attention=False, use_deformable=False,
        use_autoencoder=True,
    ).to(device)

    ocr = OCRModel(
        pretrained_path="baudm/parseq-base",
        vocab="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        max_length=7, frozen=False,
    ).to(device)

    # Minimal dummy loader
    dummy_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 34, 62))
    dummy_loader = torch.utils.data.DataLoader(dummy_ds, batch_size=2)

    trainer = ProgressiveTrainer(
        generator=generator, ocr=ocr,
        train_loader=dummy_loader, val_loader=dummy_loader,
        config=config, device=device,
    )

    hr = torch.randn(8, 3, 68, 124).to(device)
    lr = torch.randn(8, 3, 34, 62).to(device)

    result = trainer._simulate_test_pipeline(hr, lr, degrade_ratio=0.5)
    assert result.shape == hr.shape
    same = sum(1 for i in range(8) if torch.allclose(result[i], hr[i]))
    print(f"  Input: hr={hr.shape}, lr={lr.shape}")
    print(f"  Result: {result.shape}, unchanged: {same}/8 (~4 expected)")


# ============================================================
# Test 9: _apply_ocr_augmentation
# ============================================================
@test("_apply_ocr_augmentation")
def test_ocr_augmentation():
    from src.training.progressive_trainer import ProgressiveTrainer
    from src.models.generator import Generator
    from src.ocr.ocr_model import OCRModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("configs/lp_asrn.yaml", "r") as f:
        config = yaml.safe_load(f)

    generator = Generator(
        in_channels=3, out_channels=3, num_features=64,
        num_blocks=4, num_layers_per_block=2, upscale_factor=2,
        use_enhanced_attention=False, use_deformable=False,
        use_autoencoder=True,
    ).to(device)

    ocr = OCRModel(
        pretrained_path="baudm/parseq-base",
        vocab="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        max_length=7, frozen=False,
    ).to(device)

    dummy_ds = torch.utils.data.TensorDataset(torch.randn(4, 3, 34, 62))
    dummy_loader = torch.utils.data.DataLoader(dummy_ds, batch_size=2)

    trainer = ProgressiveTrainer(
        generator=generator, ocr=ocr,
        train_loader=dummy_loader, val_loader=dummy_loader,
        config=config, device=device,
    )

    images = torch.randn(4, 3, 68, 124).to(device)
    augmented = trainer._apply_ocr_augmentation(images)
    assert augmented.shape == images.shape
    print(f"  Input: {images.shape} -> Augmented: {augmented.shape}")


# ============================================================
# Test 10: Full Stage 0 training step (synthetic data)
# ============================================================
@test("Stage 0 full training step (synthetic)")
def test_stage0_step():
    from src.data.lp_dataset import create_dataloaders
    from src.models.generator import Generator
    from src.ocr.ocr_model import OCRModel
    from src.training.progressive_trainer import ProgressiveTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("configs/lp_asrn.yaml", "r") as f:
        config = yaml.safe_load(f)

    tmp_dir = create_synthetic_dataset(num_tracks=3)
    try:
        result = create_dataloaders(
            root_dir=tmp_dir, batch_size=4, num_workers=0,
            image_size=(34, 62), distributed=False,
            ocr_pretrain_mode=True, aspect_ratio_augment=True,
            test_aspect_range=(0.25, 0.45),
            test_resolution_augment=True, test_resolution_prob=0.7,
            jpeg_augment=True, jpeg_quality_range=(60, 95),
            no_crop_prob=0.3,
            test_like_val=True, test_like_val_fraction=0.5,
        )
        train_loader, val_loader, _, test_like_val_loader = result

        generator = Generator(
            in_channels=3, out_channels=3, num_features=64,
            num_blocks=4, num_layers_per_block=2, upscale_factor=2,
            use_enhanced_attention=False, use_deformable=False, use_autoencoder=True,
        ).to(device)

        ocr = OCRModel(
            pretrained_path="baudm/parseq-base",
            vocab="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            max_length=7, frozen=False,
        ).to(device)

        trainer = ProgressiveTrainer(
            generator=generator, ocr=ocr,
            train_loader=train_loader, val_loader=val_loader,
            config=config, device=device,
            test_like_val_loader=test_like_val_loader,
        )

        # Simulate Stage 0 training step
        trainer.ocr.train()
        ocr_unwrapped = trainer._unwrap_model(trainer.ocr)
        optimizer = torch.optim.AdamW(trainer.ocr.parameters(), lr=0.0005, weight_decay=0.01)

        batch = next(iter(train_loader))
        hr = batch["hr"].to(device)
        lr = batch["lr"].to(device)
        gt = batch["plate_text"]

        # The key new feature: _simulate_test_pipeline
        hr = trainer._simulate_test_pipeline(hr, lr, degrade_ratio=0.4)
        hr = trainer._apply_ocr_augmentation(hr)

        loss = ocr_unwrapped.forward_train(hr, gt, label_smoothing=0.1)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainer.ocr.parameters(), max_norm=1.0)
        optimizer.step()

        print(f"  Stage 0 step OK: loss={loss.item():.4f}")
        print(f"  Batch: LR={batch['lr'].shape}, HR={batch['hr'].shape}")
        print(f"  Texts: {gt[:4]}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# Test 11: Generator + OCR pipeline (Stage 1+ flow)
# ============================================================
@test("Generator+OCR pipeline (Stage 1+ flow)")
def test_generator_ocr_pipeline():
    from src.models.generator import Generator
    from src.ocr.ocr_model import OCRModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    generator = Generator(
        in_channels=3, out_channels=3, num_features=64,
        num_blocks=4, num_layers_per_block=2, upscale_factor=2,
        use_enhanced_attention=False, use_deformable=False, use_autoencoder=True,
    ).to(device)

    ocr = OCRModel(
        pretrained_path="baudm/parseq-base",
        vocab="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        max_length=7, frozen=True,
    ).to(device)

    # Synthetic batch
    lr = torch.randn(4, 3, 34, 62).to(device)
    hr = torch.randn(4, 3, 68, 124).to(device)

    # Forward: LR -> SR -> OCR
    sr = generator(lr)
    assert sr.shape == hr.shape, f"SR shape {sr.shape} != HR shape {hr.shape}"

    with torch.no_grad():
        preds = ocr.predict(sr, beam_width=1)
    print(f"  LR: {lr.shape} -> SR: {sr.shape}")
    print(f"  Predictions: {preds}")

    # L1 + backward
    l1_loss = torch.nn.L1Loss()(sr, hr)
    l1_loss.backward()
    print(f"  L1 loss: {l1_loss.item():.4f}, backward OK")


# ============================================================
# Test 12: Test-like validation loop
# ============================================================
@test("Test-like validation loop")
def test_test_like_validation():
    from src.data.lp_dataset import create_dataloaders
    from src.models.generator import Generator
    from src.ocr.ocr_model import OCRModel
    from src.training.progressive_trainer import ProgressiveTrainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open("configs/lp_asrn.yaml", "r") as f:
        config = yaml.safe_load(f)

    tmp_dir = create_synthetic_dataset(num_tracks=3)
    try:
        result = create_dataloaders(
            root_dir=tmp_dir, batch_size=4, num_workers=0,
            image_size=(34, 62), distributed=False,
            jpeg_augment=True, jpeg_quality_range=(60, 95), no_crop_prob=0.3,
            test_like_val=True, test_like_val_fraction=0.5,
        )
        train_loader, val_loader, _, test_like_val_loader = result

        generator = Generator(
            in_channels=3, out_channels=3, num_features=64,
            num_blocks=4, num_layers_per_block=2, upscale_factor=2,
            use_enhanced_attention=False, use_deformable=False, use_autoencoder=True,
        ).to(device)

        ocr = OCRModel(
            pretrained_path="baudm/parseq-base",
            vocab="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            max_length=7, frozen=True,
        ).to(device)

        trainer = ProgressiveTrainer(
            generator=generator, ocr=ocr,
            train_loader=train_loader, val_loader=val_loader,
            config=config, device=device,
            test_like_val_loader=test_like_val_loader,
        )

        metrics = trainer.validate_test_like(beam_width=1)
        print(f"  Metrics: {metrics}")
        assert "test_like_word_acc" in metrics
        assert "test_like_char_acc" in metrics
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ============================================================
# Test 13: LCOFL loss computation
# ============================================================
@test("LCOFL loss computation")
def test_lcofl_loss():
    from src.losses.lcofl import LCOFL
    from src.ocr.ocr_model import OCRModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lcofl = LCOFL(
        vocab="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        lambda_layout=0.5,
        lambda_ssim=0.2,
    )

    ocr = OCRModel(
        pretrained_path="baudm/parseq-base",
        vocab="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        max_length=7, frozen=True,
    ).to(device)

    sr = torch.randn(4, 3, 68, 124, requires_grad=True).to(device)
    hr = torch.randn(4, 3, 68, 124).to(device)
    gt_texts = ["ABC1234", "DEF5678", "GHI9012", "JKL3456"]
    layouts = ["Brazilian", "Mercosur", "Brazilian", "Mercosur"]

    # Get proper logits and predictions from OCR (with gradient tracking)
    pred_logits = ocr(sr, return_logits=True)
    with torch.no_grad():
        pred_texts = ocr.predict(sr, beam_width=1)

    loss, lcofl_info = lcofl(sr, hr, pred_logits, gt_texts, pred_texts)
    loss.backward()
    print(f"  LCOFL loss: {loss.item():.4f}")
    print(f"  LCOFL info keys: {list(lcofl_info.keys())}")
    # SR gradient may be None if LCOFL doesn't directly use sr pixels
    # The key is that loss.backward() succeeded without errors
    print(f"  SR grad: {'exists' if sr.grad is not None else 'None (LCOFL uses logits path)'}")


# ============================================================
# Test 14: train_progressive.py import & config wiring
# ============================================================
@test("train_progressive.py config wiring")
def test_train_script_config():
    """Verify the train script correctly passes all new params."""
    script_path = Path("scripts/train_progressive.py")
    assert script_path.exists()

    source = script_path.read_text()

    # Check single-GPU path passes new params
    assert "jpeg_augment" in source, "Single-GPU path missing jpeg_augment"
    assert "no_crop_prob" in source, "Single-GPU path missing no_crop_prob"
    assert "test_like_val" in source, "Single-GPU path missing test_like_val"
    assert "test_like_val_fraction" in source, "Single-GPU path missing test_like_val_fraction"
    assert "test_like_val_loader" in source, "Missing test_like_val_loader in trainer creation"

    # Count occurrences to verify both DDP and single-GPU paths
    jpeg_count = source.count("jpeg_augment")
    assert jpeg_count >= 2, f"jpeg_augment only appears {jpeg_count} times (need DDP + single-GPU)"

    test_like_count = source.count("test_like_val_loader=test_like_val_loader")
    assert test_like_count >= 2, f"test_like_val_loader passed to trainer only {test_like_count} times (need DDP + single-GPU)"

    print(f"  jpeg_augment occurrences: {jpeg_count}")
    print(f"  test_like_val_loader wired: {test_like_count} places")
    print(f"  All new params properly wired in train script")


# ============================================================
# Run all tests
# ============================================================
def main():
    print("=" * 60)
    print("LP-ASRN SMOKE TEST")
    print(f"Python: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, 'total_memory', getattr(props, 'total_mem', 0))
        print(f"GPU memory: {mem / 1024**3:.1f} GB")
    print("=" * 60)

    # Tests ordered: fast first, GPU-heavy last
    test_config()
    test_jpeg_compression()
    test_train_script_config()
    test_dataset_creation()
    test_dataset_creation_no_testlike()
    test_generator()
    test_ocr()
    test_parseq_forward_train()
    test_simulate_test_pipeline()
    test_ocr_augmentation()
    test_stage0_step()
    test_generator_ocr_pipeline()
    test_test_like_validation()
    test_lcofl_loss()

    # Summary
    print(f"\n{'='*60}")
    print(f"SMOKE TEST SUMMARY")
    print(f"{'='*60}")
    print(f"  Passed: {len(passed)}/{len(passed) + len(failed)}")
    for name in passed:
        print(f"    + {name}")

    if failed:
        print(f"\n  Failed: {len(failed)}/{len(passed) + len(failed)}")
        for name, error in failed:
            print(f"    X {name}: {error}")
        sys.exit(1)
    else:
        print(f"\n  All tests passed! Pipeline is ready for deployment.")
        sys.exit(0)


if __name__ == "__main__":
    main()
