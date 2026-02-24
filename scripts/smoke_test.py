"""Smoke test all changes to trainer, config, and dataset."""
import sys
sys.path.insert(0, '.')
import yaml
import torch


def main():
    # 1. Config loading
    with open('configs/lp_asrn.yaml') as f:
        config = yaml.safe_load(f)

    stage2 = config['progressive_training']['stage2']
    print(f"Stage2 loss_components: {stage2.get('loss_components')}")
    print(f"Stage2 psnr_floor: {stage2.get('psnr_floor')}")
    print(f"lambda_lcofl: {config['loss']['lambda_lcofl']}")
    print(f"val_split: {config['data']['val_split']}")
    print(f"lambda_ssim: {config['loss'].get('lambda_ssim', 'missing')}")
    assert config['loss']['lambda_lcofl'] == 0.5, "lambda_lcofl should be 0.5"
    assert config['data']['val_split'] == 0.10, "val_split should be 0.10"
    assert 'ssim' in stage2.get('loss_components', []), "ssim should be in stage2 loss_components"
    assert stage2.get('psnr_floor') == 12.5, "psnr_floor should be 12.5"
    print("  [OK] Config\n")

    # 2. StageConfig
    from src.training.progressive_trainer import StageConfig
    sc = StageConfig(
        name="test", epochs=10, lr=0.001,
        loss_components=['l1', 'lcofl', 'ssim'],
        freeze_ocr=True,
        psnr_floor=12.5
    )
    assert sc.psnr_floor == 12.5
    sc2 = StageConfig(name="test2", epochs=5, lr=0.01, loss_components=['l1'], freeze_ocr=False)
    assert sc2.psnr_floor == 0.0, "Default psnr_floor should be 0.0"
    print("  [OK] StageConfig\n")

    # 3. Dataset loads both PNG and JPG
    from src.data.lp_dataset import LicensePlateDataset
    ds = LicensePlateDataset(root_dir='data/train', image_size=(34, 62), augment=False)
    jpg_count = sum(1 for s in ds.samples if s['lr_path'].endswith('.jpg'))
    png_count = sum(1 for s in ds.samples if s['lr_path'].endswith('.png'))
    print(f"Dataset: {len(ds)} samples ({png_count} PNG + {jpg_count} JPG)")
    assert jpg_count > 0, "Should have JPG samples"
    assert png_count > 0, "Should have PNG samples"
    assert len(ds) == 100000, f"Expected 100000, got {len(ds)}"
    print("  [OK] Dataset loads PNG + JPG\n")

    # 4. Load a JPG sample
    sample = next(s for s in ds.samples if s['lr_path'].endswith('.jpg'))
    idx = ds.samples.index(sample)
    item = ds[idx]
    print(f"JPG sample: lr={item['lr'].shape}, hr={item['hr'].shape}, text={item['plate_text']}")
    assert item['lr'].shape[0] == 3, "LR should have 3 channels"
    assert item['hr'].shape[0] == 3, "HR should have 3 channels"
    print("  [OK] JPG sample loads correctly\n")

    # 5. Check save_checkpoint signature accepts save_path
    import inspect
    from src.training.progressive_trainer import ProgressiveTrainer
    sig = inspect.signature(ProgressiveTrainer.save_checkpoint)
    assert 'save_path' in sig.parameters, "save_checkpoint should accept save_path"
    print("  [OK] save_checkpoint has save_path parameter\n")

    print("=" * 50)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 50)


if __name__ == '__main__':
    main()
