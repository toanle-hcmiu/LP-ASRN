"""Compare predictions from warmup vs LCOFL checkpoints on test-public."""
import sys
sys.path.insert(0, '.')

import torch
import yaml
import argparse
from scripts.inference import (
    load_models, load_and_preprocess, predict_with_confidence,
    aggregate_track_predictions, discover_tracks, _detect_architecture_from_checkpoint
)
from src.models.generator import Generator


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    with open('configs/lp_asrn.yaml') as f:
        config = yaml.safe_load(f)

    # Load LCOFL best checkpoint
    args = argparse.Namespace(
        checkpoint='outputs/run_20260223_142529/best.pth',
        config='configs/lp_asrn.yaml',
        device=str(device),
        beam_width=5,
        ocr_only=False,
        no_sr=False,
    )
    generator_lcofl, ocr = load_models(args, config, device)

    # Load warmup checkpoint into separate generator
    warmup_ckpt = torch.load(
        'outputs/run_20260223_142529/stage_warmup_epoch_76.pth',
        map_location=device
    )
    gen_state = warmup_ckpt['generator_state_dict']
    detected = _detect_architecture_from_checkpoint(gen_state)
    gen_warmup = Generator(
        in_channels=3, out_channels=3,
        num_features=detected.get('num_features', 64),
        num_blocks=detected.get('num_blocks', 12),
        num_layers_per_block=3,
        upscale_factor=detected.get('upscale_factor', 2),
        use_enhanced_attention=detected.get('use_enhanced_attention', True),
        use_deformable=detected.get('use_deformable', True),
        use_character_attention=detected.get('use_character_attention', False),
    ).to(device)
    gen_warmup.load_state_dict(gen_state)
    gen_warmup.eval()
    print(f"Warmup checkpoint: epoch {warmup_ckpt.get('epoch')}, word_acc={warmup_ckpt.get('best_word_acc')}")

    # Discover test tracks
    tracks = discover_tracks('data/test-public')[:30]
    print(f"Testing on {len(tracks)} tracks\n")

    warmup_preds = {}
    lcofl_preds = {}

    for tid, paths in tracks:
        tensors = [load_and_preprocess(p, lr_size=(34, 62)) for p in paths]
        batch = torch.stack(tensors).to(device)

        with torch.no_grad():
            # Warmup SR
            sr_w = gen_warmup(batch)
            logits_w = ocr(sr_w, return_logits=True)
            preds_w = predict_with_confidence(ocr, logits_w, beam_width=5)
            text_w, conf_w = aggregate_track_predictions(preds_w)
            warmup_preds[tid] = (text_w, conf_w)

            # LCOFL SR
            sr_l = generator_lcofl(batch)
            logits_l = ocr(sr_l, return_logits=True)
            preds_l = predict_with_confidence(ocr, logits_l, beam_width=5)
            text_l, conf_l = aggregate_track_predictions(preds_l)
            lcofl_preds[tid] = (text_l, conf_l)

    # Print comparison
    header = f"{'Track':<15} | {'Warmup SR':<20} | {'LCOFL SR':<20} | Match"
    print(header)
    print("-" * len(header))

    agree = 0
    for tid, _ in tracks:
        tw, cw = warmup_preds[tid]
        tl, cl = lcofl_preds[tid]
        match = "YES" if tw == tl else ""
        print(f"{tid:<15} | {tw:<12} ({cw:.2f})  | {tl:<12} ({cl:.2f})  | {match}")
        if tw == tl:
            agree += 1

    print(f"\nAgreement: {agree}/{len(tracks)}")
    print(f"Warmup avg confidence: {sum(c for _, c in warmup_preds.values())/len(warmup_preds):.3f}")
    print(f"LCOFL avg confidence:  {sum(c for _, c in lcofl_preds.values())/len(lcofl_preds):.3f}")


if __name__ == '__main__':
    main()
