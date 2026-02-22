#!/usr/bin/env python3
"""
Analyze differences between training and test data distributions.

This helps identify why validation accuracy (79%) doesn't match test accuracy (41%).
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image
import json
from collections import defaultdict
import argparse

def get_image_stats(image_path):
    """Get statistics for a single image."""
    try:
        img = Image.open(image_path).convert('RGB')
        arr = np.array(img)
        h, w = arr.shape[:2]
        aspect = h / w if w > 0 else 0

        # Brightness and contrast
        brightness = arr.mean() / 255.0
        std = arr.std() / 255.0

        # Sharpness (using Laplacian variance)
        gray = np.mean(arr, axis=2)
        laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        from scipy.ndimage import correlate
        sharpness = correlate(gray, laplacian).var() if gray.size > 0 else 0

        # Blur score (edge detection)
        from scipy.ndimage import sobel
        gx = sobel(gray)
        gy = sobel(gray, axis=0)
        edge_strength = np.sqrt(gx**2 + gy**2).mean()

        return {
            'height': h,
            'width': w,
            'aspect_ratio': aspect,
            'brightness': brightness,
            'contrast': std,
            'sharpness': sharpness,
            'edge_strength': edge_strength,
            'size': h * w
        }
    except Exception as e:
        return None

def analyze_dataset(root_dir, name, max_samples=None):
    """Analyze a dataset directory."""
    print(f"\n{'='*60}")
    print(f"Analyzing {name}")
    print(f"{'='*60}")

    root = Path(root_dir)

    # Find all images
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(root.rglob(ext)))

    if max_samples:
        image_files = image_files[:max_samples]

    print(f"Found {len(image_files)} images")

    if not image_files:
        print("No images found!")
        return {}

    # Collect statistics
    stats = {
        'heights': [],
        'widths': [],
        'aspect_ratios': [],
        'brightness': [],
        'contrast': [],
        'sharpness': [],
        'edge_strength': [],
        'sizes': [],
        'unique_sizes': defaultdict(int),
    }

    for i, img_path in enumerate(image_files):
        if (i + 1) % 1000 == 0:
            print(f"  Processing {i+1}/{len(image_files)}...")

        s = get_image_stats(img_path)
        if s:
            stats['heights'].append(s['height'])
            stats['widths'].append(s['width'])
            stats['aspect_ratios'].append(s['aspect_ratio'])
            stats['brightness'].append(s['brightness'])
            stats['contrast'].append(s['contrast'])
            stats['sharpness'].append(s['sharpness'])
            stats['edge_strength'].append(s['edge_strength'])
            stats['sizes'].append(s['size'])
            stats['unique_sizes'][(s['height'], s['width'])] += 1

    # Print summary statistics
    print(f"\nImage Dimensions:")
    print(f"  Height: {np.mean(stats['heights']):.1f} ± {np.std(stats['heights']):.1f} (range: {min(stats['heights'])} - {max(stats['heights'])})")
    print(f"  Width:  {np.mean(stats['widths']):.1f} ± {np.std(stats['widths']):.1f} (range: {min(stats['widths'])} - {max(stats['widths'])})")
    print(f"  Aspect Ratio: {np.mean(stats['aspect_ratios']):.3f} ± {np.std(stats['aspect_ratios']):.3f} (range: {min(stats['aspect_ratios']):.3f} - {max(stats['aspect_ratios']):.3f})")
    print(f"  Size (pixels): {np.mean(stats['sizes']):.0f} ± {np.std(stats['sizes']):.0f}")

    print(f"\nImage Quality:")
    print(f"  Brightness: {np.mean(stats['brightness']):.3f} ± {np.std(stats['brightness']):.3f} (0=black, 1=white)")
    print(f"  Contrast: {np.mean(stats['contrast']):.3f} ± {np.std(stats['contrast']):.3f}")
    print(f"  Sharpness: {np.mean(stats['sharpness']):.1f} ± {np.std(stats['sharpness']):.1f}")
    print(f"  Edge Strength: {np.mean(stats['edge_strength']):.1f} ± {np.std(stats['edge_strength']):.1f}")

    print(f"\nUnique Sizes (top 10):")
    top_sizes = sorted(stats['unique_sizes'].items(), key=lambda x: x[1], reverse=True)[:10]
    for (h, w), count in top_sizes:
        print(f"  {h}×{w}: {count} images ({count/len(image_files)*100:.1f}%)")
    print(f"  Total unique sizes: {len(stats['unique_sizes'])}")

    return stats


def compare_distributions(train_stats, test_stats):
    """Compare train and test distributions and report significant differences."""
    print(f"\n{'='*60}")
    print("DISTRIBUTION COMPARISON")
    print(f"{'='*60}")

    metrics = [
        ('height', 'heights', 'pixels'),
        ('width', 'widths', 'pixels'),
        ('aspect_ratio', 'aspect_ratios', ''),
        ('brightness', 'brightness', ''),
        ('contrast', 'contrast', ''),
        ('sharpness', 'sharpness', ''),
        ('edge_strength', 'edge_strength', ''),
    ]

    print(f"\n{'Metric':<20} {'Train Mean':<15} {'Test Mean':<15} {'Diff':<15} {'% Change':<10}")
    print("-" * 80)

    for name, key, unit in metrics:
        train_mean = np.mean(train_stats[key])
        test_mean = np.mean(test_stats[key])
        diff = test_mean - train_mean
        pct_change = (diff / train_mean) * 100 if train_mean != 0 else 0

        # Calculate Cohen's d (effect size)
        train_std = np.std(train_stats[key])
        test_std = np.std(test_stats[key])
        pooled_std = np.sqrt((train_std**2 + test_std**2) / 2)
        cohens_d = diff / pooled_std if pooled_std > 0 else 0

        # Flag significant differences
        flag = " ⚠️" if abs(cohens_d) > 0.5 else ""

        print(f"{name:<20} {train_mean:<15.3f} {test_mean:<15.3f} {diff:<+15.3f} {pct_change:>+6.1f}%{flag}")

    # Check size overlap
    train_sizes = set(train_stats['unique_sizes'].keys())
    test_sizes = set(test_stats['unique_sizes'].keys())

    common_sizes = train_sizes & test_sizes
    train_only = train_sizes - test_sizes
    test_only = test_sizes - train_sizes

    print(f"\nSize Overlap:")
    print(f"  Common sizes: {len(common_sizes)}")
    print(f"  Train-only: {len(train_only)}")
    print(f"  Test-only: {len(test_only)}")

    if test_only:
        print(f"\n  Test-exclusive sizes (first 10):")
        for i, (h, w) in enumerate(sorted(test_only)[:10]):
            count = test_stats['unique_sizes'][(h, w)]
            print(f"    {h}×{w}: {count} images")


def main():
    parser = argparse.ArgumentParser(description='Analyze train/test data mismatch')
    parser.add_argument('--train-dir', default='data/train', help='Training data directory')
    parser.add_argument('--test-dir', default='data/test', help='Test data directory')
    parser.add_argument('--max-samples', type=int, default=None, help='Max samples to analyze per dataset')

    args = parser.parse_args()

    # Analyze training data
    train_stats = analyze_dataset(args.train_dir, "TRAINING DATA", args.max_samples)

    # Analyze test data
    test_stats = analyze_dataset(args.test_dir, "TEST DATA", args.max_samples)

    # Compare
    if train_stats and test_stats:
        compare_distributions(train_stats, test_stats)


if __name__ == "__main__":
    main()
