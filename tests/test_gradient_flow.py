"""
Test: Verify LCOFL gradient flow from OCR through to generator.

This test ensures that when LCOFL loss is computed, gradients
propagate through the frozen OCR model back to the generator.
Previously, torch.no_grad() was blocking this critical gradient path.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import pytest


def test_lcofl_gradients_reach_generator():
    """
    Verify that LCOFL loss produces non-zero gradients in generator parameters.

    The key test: if OCR forward pass is wrapped in torch.no_grad(),
    LCOFL loss has no gradient path to the generator. After the fix,
    OCR params are frozen (requires_grad=False) but gradients flow
    THROUGH the OCR back to the generator.
    """
    from src.models.generator import Generator
    from src.ocr.ocr_model import OCRModel
    from src.losses.lcofl import LCOFL

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Create a small generator
    gen = Generator(
        in_channels=3,
        out_channels=3,
        num_features=16,  # Small for speed
        num_blocks=2,
        upscale_factor=2,
        use_enhanced_attention=False,
        use_deformable=False,
    ).to(device)

    # Create OCR model (initially unfrozen so we can freeze manually)
    ocr = OCRModel(
        use_parseq=False,  # Use SimpleCRNN
        vocab="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        max_length=7,
        frozen=False,  # We freeze manually below
    ).to(device)

    # Freeze OCR (as done in LCOFL stage)
    for param in ocr.parameters():
        param.requires_grad = False

    # Create LCOFL loss
    lcofl = LCOFL(
        lambda_layout=0.5,
        lambda_ssim=0.2,
    )

    # Forward pass
    lr_input = torch.randn(2, 3, 34, 62, device=device)
    sr_output = gen(lr_input)

    # OCR forward WITH gradients flowing through (the fix)
    pred_logits = ocr(sr_output, return_logits=True)

    # Compute LCOFL loss
    gt_texts = ["ABC1234", "XYZ5678"]
    loss, info = lcofl(
        sr_output, sr_output.detach(),  # Use sr as both for simplicity
        pred_logits, gt_texts
    )

    # Backward pass
    loss.backward()

    # The critical assertion: generator parameters must have non-zero gradients
    has_grad = False
    for name, param in gen.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, (
        "LCOFL gradients did NOT reach the generator! "
        "The OCR forward pass may still be wrapped in torch.no_grad()."
    )
    print("[PASS] LCOFL gradients successfully flow through OCR to generator")


def test_ocr_params_remain_frozen():
    """
    Verify that OCR parameters don't accumulate gradients even though
    gradients flow through the network.
    """
    from src.models.generator import Generator
    from src.ocr.ocr_model import OCRModel
    from src.losses.lcofl import LCOFL

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gen = Generator(
        in_channels=3, out_channels=3, num_features=16,
        num_blocks=2, upscale_factor=2, use_enhanced_attention=False,
        use_deformable=False,
    ).to(device)

    ocr = OCRModel(
        use_parseq=False,
        vocab="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        max_length=7,
        frozen=False,
    ).to(device)

    # Freeze OCR
    for param in ocr.parameters():
        param.requires_grad = False

    lcofl = LCOFL(lambda_layout=0.5, lambda_ssim=0.2)

    lr_input = torch.randn(2, 3, 34, 62, device=device)
    sr_output = gen(lr_input)
    pred_logits = ocr(sr_output, return_logits=True)
    gt_texts = ["ABC1234", "XYZ5678"]
    loss, _ = lcofl(sr_output, sr_output.detach(), pred_logits, gt_texts)
    loss.backward()

    # OCR params should have NO gradients (frozen)
    for name, param in ocr.named_parameters():
        assert param.grad is None, (
            f"OCR param '{name}' has gradients -- it should be frozen!"
        )

    print("[PASS] OCR parameters remain frozen (no gradient accumulation)")


def test_perceptual_loss_vgg19():
    """Verify VGG19 perceptual loss computes and produces gradients."""
    from src.losses.basic import PerceptualLoss

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    percep = PerceptualLoss().to(device)

    # Create test images (requires grad for one to check gradient flow)
    pred = torch.randn(2, 3, 68, 124, device=device, requires_grad=True)
    target = torch.randn(2, 3, 68, 124, device=device)

    loss = percep(pred, target)
    assert loss.item() > 0, "Perceptual loss should be non-zero for different images"

    loss.backward()
    assert pred.grad is not None, "Perceptual loss should produce gradients"
    assert pred.grad.abs().sum() > 0, "Gradients should be non-zero"

    # VGG params should remain frozen
    for param in percep.features.parameters():
        assert not param.requires_grad, "VGG parameters should be frozen"

    print("[PASS] VGG19 perceptual loss works correctly")


def test_gaussian_ssim():
    """Verify upgraded Gaussian SSIM computes correctly."""
    from src.losses.lcofl import ssim

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Identical images should have SSIM ~ 1.0
    x = torch.randn(2, 3, 68, 124, device=device)
    # Convert to [-1, 1] range (ssim expects this)
    x = torch.tanh(x)
    ssim_val = ssim(x, x)
    assert abs(ssim_val.item() - 1.0) < 0.01, (
        f"SSIM of identical images should be ~1.0, got {ssim_val.item():.4f}"
    )

    # Very different images should have low SSIM
    y = torch.randn(2, 3, 68, 124, device=device)
    y = torch.tanh(y)
    ssim_val2 = ssim(x, y)
    assert ssim_val2.item() < 0.5, (
        f"SSIM of random images should be low, got {ssim_val2.item():.4f}"
    )

    print(f"[PASS] Gaussian SSIM: identical={ssim_val.item():.4f}, random={ssim_val2.item():.4f}")


if __name__ == "__main__":
    print("Running gradient flow and loss tests...\n")
    test_gaussian_ssim()
    test_lcofl_gradients_reach_generator()
    test_ocr_params_remain_frozen()
    test_perceptual_loss_vgg19()
    print("\nAll tests passed!")
