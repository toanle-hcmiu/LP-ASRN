"""Compare test-blind and test-public datasets for distribution mismatch."""
import os
import sys
from PIL import Image
import numpy as np
from collections import Counter

def analyze_dataset(data_dir, name):
    tracks = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"Total tracks: {len(tracks)}")
    print(f"Track ID range: {tracks[0]} - {tracks[-1]}")
    
    # Count files per track (sample first 50)
    files_per_track = []
    for t in tracks[:50]:
        tp = os.path.join(data_dir, t)
        files = os.listdir(tp)
        files_per_track.append(len(files))
    fpct = Counter(files_per_track)
    print(f"Files per track (first 50): {dict(fpct)}")
    
    # Extensions
    exts = Counter()
    for t in tracks[:50]:
        tp = os.path.join(data_dir, t)
        for f in os.listdir(tp):
            exts[os.path.splitext(f)[1].lower()] += 1
    print(f"File extensions (first 50 tracks): {dict(exts)}")
    
    # Image dimensions - sample 200 tracks evenly
    widths, heights, aspects, filesizes = [], [], [], []
    step = max(1, len(tracks) // 200)
    for t in tracks[::step][:200]:
        tp = os.path.join(data_dir, t)
        for f in sorted(os.listdir(tp))[:1]:  # just first image per track
            fp = os.path.join(tp, f)
            try:
                img = Image.open(fp)
                w, h = img.size
                widths.append(w)
                heights.append(h)
                aspects.append(w / h)
                filesizes.append(os.path.getsize(fp))
            except Exception:
                pass
    
    widths = np.array(widths)
    heights = np.array(heights)
    aspects = np.array(aspects)
    filesizes = np.array(filesizes)
    
    print(f"\nImage dimensions (sampled {len(widths)} tracks, lr-001.jpg each):")
    print(f"  Width:  min={widths.min()}, max={widths.max()}, mean={widths.mean():.1f}, median={np.median(widths):.0f}, std={widths.std():.1f}")
    print(f"  Height: min={heights.min()}, max={heights.max()}, mean={heights.mean():.1f}, median={np.median(heights):.0f}, std={heights.std():.1f}")
    print(f"  Aspect: min={aspects.min():.3f}, max={aspects.max():.3f}, mean={aspects.mean():.3f}, median={np.median(aspects):.3f}, std={aspects.std():.3f}")
    print(f"  Filesize: min={filesizes.min()}, max={filesizes.max()}, mean={filesizes.mean():.0f}, median={np.median(filesizes):.0f}")
    
    # Width histogram
    w_bins = [0, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 200]
    w_hist = np.histogram(widths, bins=w_bins)[0]
    print(f"\n  Width distribution:")
    for i in range(len(w_bins) - 1):
        pct = w_hist[i] / len(widths) * 100
        bar = "#" * int(pct / 2)
        print(f"    {w_bins[i]:3d}-{w_bins[i+1]:3d}: {w_hist[i]:4d} ({pct:5.1f}%) {bar}")
    
    # Height histogram
    h_bins = [0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80]
    h_hist = np.histogram(heights, bins=h_bins)[0]
    print(f"\n  Height distribution:")
    for i in range(len(h_bins) - 1):
        pct = h_hist[i] / len(heights) * 100
        bar = "#" * int(pct / 2)
        print(f"    {h_bins[i]:3d}-{h_bins[i+1]:3d}: {h_hist[i]:4d} ({pct:5.1f}%) {bar}")
    
    # Aspect ratio histogram
    a_bins = [0, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.8, 1.0, 2.0]
    a_hist = np.histogram(aspects, bins=a_bins)[0]
    print(f"\n  Aspect ratio (W/H) distribution:")
    for i in range(len(a_bins) - 1):
        pct = a_hist[i] / len(aspects) * 100
        bar = "#" * int(pct / 2)
        print(f"    {a_bins[i]:.2f}-{a_bins[i+1]:.2f}: {a_hist[i]:4d} ({pct:5.1f}%) {bar}")
    
    return set(tracks), widths, heights, aspects, filesizes


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    blind_dir = os.path.join(base, "data", "test-blind")
    pub_dir = os.path.join(base, "data", "test-public")
    
    blind_tracks, bw, bh, ba, bfs = analyze_dataset(blind_dir, "TEST-BLIND")
    pub_tracks, pw, ph, pa, pfs = analyze_dataset(pub_dir, "TEST-PUBLIC")
    
    # Overlap analysis
    print(f"\n{'='*60}")
    print(f"  OVERLAP ANALYSIS")
    print(f"{'='*60}")
    overlap = blind_tracks & pub_tracks
    blind_only = blind_tracks - pub_tracks
    pub_only = pub_tracks - blind_tracks
    print(f"test-blind tracks:  {len(blind_tracks)}")
    print(f"test-public tracks: {len(pub_tracks)}")
    print(f"Shared tracks:      {len(overlap)}")
    print(f"Blind-only tracks:  {len(blind_only)}")
    print(f"Public-only tracks: {len(pub_only)}")
    print(f"test-public is subset of test-blind: {pub_tracks.issubset(blind_tracks)}")
    print(f"Overlap %: {len(overlap)/len(blind_tracks)*100:.1f}% of blind, {len(overlap)/len(pub_tracks)*100:.1f}% of public")
    
    # Distribution comparison
    print(f"\n{'='*60}")
    print(f"  DISTRIBUTION COMPARISON")
    print(f"{'='*60}")
    print(f"Mean width  - blind: {bw.mean():.1f}, public: {pw.mean():.1f}, diff: {abs(bw.mean()-pw.mean()):.1f}")
    print(f"Mean height - blind: {bh.mean():.1f}, public: {ph.mean():.1f}, diff: {abs(bh.mean()-ph.mean()):.1f}")
    print(f"Mean aspect - blind: {ba.mean():.3f}, public: {pa.mean():.3f}, diff: {abs(ba.mean()-pa.mean()):.3f}")
    print(f"Mean fsize  - blind: {bfs.mean():.0f}, public: {pfs.mean():.0f}, diff: {abs(bfs.mean()-pfs.mean()):.0f}")
    
    # Percentile comparison
    print(f"\nPercentile comparison:")
    for p in [5, 25, 50, 75, 95]:
        print(f"  P{p:2d} width:  blind={np.percentile(bw,p):.0f}, public={np.percentile(pw,p):.0f}")
    print()
    for p in [5, 25, 50, 75, 95]:
        print(f"  P{p:2d} height: blind={np.percentile(bh,p):.0f}, public={np.percentile(ph,p):.0f}")
    print()
    for p in [5, 25, 50, 75, 95]:
        print(f"  P{p:2d} aspect: blind={np.percentile(ba,p):.3f}, public={np.percentile(pa,p):.3f}")
    
    # Kolmogorov-Smirnov test
    print(f"\n{'='*60}")
    print(f"  KOLMOGOROV-SMIRNOV TESTS (p>0.05 = similar)")
    print(f"{'='*60}")
    from scipy.stats import ks_2samp
    for name, bdata, pdata in [("width", bw, pw), ("height", bh, ph), ("aspect", ba, pa), ("filesize", bfs, pfs)]:
        stat, pval = ks_2samp(bdata, pdata)
        verdict = "SIMILAR" if pval > 0.05 else "DIFFERENT"
        print(f"  {name:10s}: KS={stat:.4f}, p={pval:.4f}  -> {verdict}")
    
    # Also compare blind-only vs shared tracks (do they differ?)
    if len(blind_only) > 50:
        print(f"\n{'='*60}")
        print(f"  BLIND-ONLY vs SHARED TRACKS")
        print(f"{'='*60}")
        # Get dimensions of blind-only tracks
        bow, boh, boa = [], [], []
        step = max(1, len(blind_only) // 100)
        blind_only_sorted = sorted(list(blind_only))
        for t in blind_only_sorted[::step][:100]:
            tp = os.path.join(blind_dir, t)
            fp = os.path.join(tp, "lr-001.jpg")
            if os.path.exists(fp):
                try:
                    img = Image.open(fp)
                    w, h = img.size
                    bow.append(w)
                    boh.append(h)
                    boa.append(w/h)
                except:
                    pass
        
        sow, soh, soa = [], [], []
        shared_sorted = sorted(list(overlap))
        step = max(1, len(shared_sorted) // 100)
        for t in shared_sorted[::step][:100]:
            tp = os.path.join(blind_dir, t)
            fp = os.path.join(tp, "lr-001.jpg")
            if os.path.exists(fp):
                try:
                    img = Image.open(fp)
                    w, h = img.size
                    sow.append(w)
                    soh.append(h)
                    soa.append(w/h)
                except:
                    pass
        
        bow, boh, boa = np.array(bow), np.array(boh), np.array(boa)
        sow, soh, soa = np.array(sow), np.array(soh), np.array(soa)
        
        print(f"Blind-only ({len(bow)} sampled): width={bow.mean():.1f}, height={boh.mean():.1f}, aspect={boa.mean():.3f}")
        print(f"Shared     ({len(sow)} sampled): width={sow.mean():.1f}, height={soh.mean():.1f}, aspect={soa.mean():.3f}")
        
        for name, bd, sd in [("width", bow, sow), ("height", boh, soh), ("aspect", boa, soa)]:
            stat, pval = ks_2samp(bd, sd)
            verdict = "SIMILAR" if pval > 0.05 else "DIFFERENT"
            print(f"  {name:10s}: KS={stat:.4f}, p={pval:.4f}  -> {verdict}")
    
    print(f"\n{'='*60}")
    print(f"  CONCLUSION")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
