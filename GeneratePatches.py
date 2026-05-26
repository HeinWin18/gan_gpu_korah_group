#!/usr/bin/env python
# coding: utf-8
"""
Generate patches from large TIFF with quality filtering.
Drops patches where b1=0 or NoData exceeds threshold.
Supports both systematic (grid) and random sampling modes.
"""

import numpy as np
from requests import patch
import rasterio
from rasterio.windows import Window
import os
import argparse
from tqdm import tqdm


# def generate_filtered_patches(
#         input_tiff,
#         output_dir,
#         patch_size=16,
#         num_patches=None,
#         stride=None,
#         max_invalid_percent=10,
#         b1_band_index=-1,
#         random_sampling=False,
#         seed=42
# ):
#     """
#     Generate patches and filter based on data quality.

#     Args:
#         input_tiff: Path to large TIFF file
#         output_dir: Where to save patches
#         patch_size: Size of patches (H=W)
#         num_patches: Number of patches to generate (if None, uses systematic grid)
#         stride: Step size for grid sampling (default: patch_size for non-overlapping)
#         max_invalid_percent: Maximum % of invalid pixels allowed (default: 10)
#         b1_band_index: Which band is b1 (default: -1 = last)
#         random_sampling: If True, randomly sample patches (default: False)
#         seed: Random seed for reproducibility (default: 42)
#     """

#     os.makedirs(output_dir, exist_ok=True)

#     print("\n" + "=" * 70)
#     print("PATCH GENERATION WITH QUALITY FILTERING")
#     print("=" * 70)

#     with rasterio.open(input_tiff) as src:
#         width, height = src.width, src.height
#         n_bands = src.count
#         nodata = src.nodata

#         print(f"\nInput: {input_tiff}")
#         print(f"  Size: {width} x {height} pixels")
#         print(f"  Bands: {n_bands}")
#         print(f"  NoData: {nodata}")

#         print(f"\nSettings:")
#         print(f"  Patch size: {patch_size}x{patch_size}")

#         if num_patches is not None:
#             print(f"  Target patches: {num_patches}")
#             print(f"  Sampling mode: {'Random' if random_sampling else 'Grid + random if needed'}")
#             random_sampling = True  # Force random if num_patches specified
#         else:
#             if stride is None:
#                 stride = patch_size
#             print(f"  Stride: {stride}")
#             print(f"  Sampling mode: Systematic grid")

#         print(f"  Max invalid: {max_invalid_percent}%")
#         print(f"  b1 band: {b1_band_index if b1_band_index >= 0 else f'last ({n_bands})'}")

#         saved = 0
#         dropped = 0
#         attempts = 0

#         if random_sampling or num_patches is not None:
#             # RANDOM SAMPLING MODE
#             rng = np.random.default_rng(seed)
#             max_attempts = (num_patches * 100) if num_patches else 10000  # Prevent infinite loop

#             target = num_patches if num_patches else max_attempts

#             print(f"\nGenerating {target} patches via random sampling...")
#             pbar = tqdm(total=target, desc="Generating patches")

#             while saved < target and attempts < max_attempts:
#                 # Random location
#                 top = int(rng.integers(0, height - patch_size))
#                 left = int(rng.integers(0, width - patch_size))

#                 window = Window(left, top, patch_size, patch_size)
#                 patch = src.read(window=window)  # (C, H, W)
#                 patch = np.moveaxis(patch, 0, -1)  # (H, W, C)

#                 attempts += 1

#                 # Check quality
#                 is_valid = check_patch_quality(
#                     patch,
#                     max_invalid_percent,
#                     b1_band_index,
#                     nodata
#                 )

#                 if is_valid:
#                     # Save patch
#                     filename = f"patch_{saved:05d}.tif"
#                     filepath = os.path.join(output_dir, filename)

#                     # Save with rasterio
#                     patch_out = np.moveaxis(patch, -1, 0)  # Back to (C, H, W)

#                     profile = src.profile.copy()
#                     profile.update({
#                         'height': patch_size,
#                         'width': patch_size,
#                         'transform': src.window_transform(window)
#                     })

#                     with rasterio.open(filepath, 'w', **profile) as dst:
#                         dst.write(patch_out)

#                     saved += 1
#                     pbar.update(1)
#                 else:
#                     dropped += 1

#                 pbar.set_postfix({
#                     'saved': saved,
#                     'dropped': dropped,
#                     'success%': f'{saved / attempts * 100:.1f}' if attempts > 0 else '0.0'
#                 })

#             pbar.close()

#             if saved < target:
#                 print(f"\n⚠️  Warning: Only generated {saved}/{target} patches after {attempts} attempts")
#                 print(f"   Try lowering --max_invalid threshold or using larger input image")

#         else:
#             # SYSTEMATIC GRID MODE
#             if stride is None:
#                 stride = patch_size

#             # Calculate grid
#             n_x = (width - patch_size) // stride + 1
#             n_y = (height - patch_size) // stride + 1
#             total = n_x * n_y

#             print(f"\nTotal possible patches: {total:,} ({n_x} x {n_y})")
#             print("\nGenerating patches...")
#             pbar = tqdm(total=total)

#             for i in range(n_y):
#                 for j in range(n_x):
#                     # Read patch
#                     col = j * stride
#                     row = i * stride
#                     window = Window(col, row, patch_size, patch_size)
#                     patch = src.read(window=window)  # (C, H, W)
#                     patch = np.moveaxis(patch, 0, -1)  # (H, W, C)

#                     attempts += 1

#                     # Skip if wrong size (edge cases)
#                     if patch.shape[:2] != (patch_size, patch_size):
#                         pbar.update(1)
#                         continue

#                     # Check quality
#                     is_valid = check_patch_quality(
#                         patch,
#                         max_invalid_percent,
#                         b1_band_index,
#                         nodata
#                     )

#                     if is_valid:
#                         # Save patch
#                         filename = f"patch_{i:05d}_{j:05d}.tif"
#                         filepath = os.path.join(output_dir, filename)

#                         # Save with rasterio
#                         patch_out = np.moveaxis(patch, -1, 0)  # Back to (C, H, W)

#                         profile = src.profile.copy()
#                         profile.update({
#                             'height': patch_size,
#                             'width': patch_size,
#                             'transform': src.window_transform(window)
#                         })

#                         with rasterio.open(filepath, 'w', **profile) as dst:
#                             dst.write(patch_out)

#                         saved += 1
#                     else:
#                         dropped += 1

#                     pbar.update(1)
#                     pbar.set_postfix({
#                         'saved': saved,
#                         'dropped': dropped,
#                         'keep%': f'{saved / (saved + dropped) * 100:.1f}'
#                     })

#             pbar.close()

#     print("\n" + "=" * 70)
#     print("RESULTS")
#     print("=" * 70)
#     print(f"Total attempts: {attempts:,}")
#     print(f"Saved: {saved:,} ({saved / attempts * 100:.1f}%)")
#     print(f"Dropped: {dropped:,} ({dropped / attempts * 100:.1f}%)")
#     print(f"Output: {output_dir}")
#     print("=" * 70 + "\n")

#     return saved, dropped

def generate_filtered_patches(
        input_tiff,
        output_dir,
        patch_size=16,
        num_patches=None,
        stride=None,
        max_invalid_percent=10,
        b1_band_index=-1,
        random_sampling=False,
        seed=42
):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("PATCH GENERATION WITH QUALITY FILTERING new technique")
    print("=" * 70)

    with rasterio.open(input_tiff) as src:
        width, height = src.width, src.height
        n_bands = src.count
        nodata = src.nodata

        print(f"\nInput: {input_tiff}")
        print(f"  Size: {width} x {height} pixels")
        print(f"  Bands: {n_bands}")
        print(f"  NoData: {nodata}")

        print(f"\nSettings:")
        print(f"  Patch size: {patch_size}x{patch_size}")

        if num_patches is not None:
            print(f"  Target patches: {num_patches}")
            print(f"  Sampling mode: Random (valid-location pre-scan)")
            random_sampling = True
        else:
            if stride is None:
                stride = patch_size
            print(f"  Stride: {stride}")
            print(f"  Sampling mode: Systematic grid")

        print(f"  Max invalid: {max_invalid_percent}%")
        print(f"  b1 band: {b1_band_index if b1_band_index >= 0 else f'last ({n_bands})'}")

        saved = 0
        dropped = 0
        attempts = 0

        if random_sampling or num_patches is not None:
            # ── PRE-SCAN: find valid candidate locations ──────────────────
            print("\nScanning raster for valid regions (this may take a moment)...")
            b1_full = src.read(n_bands)  # read last band (b1)

            # Build a boolean mask: True = valid pixel
            valid_mask = ~np.isnan(b1_full)
            if nodata is not None:
                valid_mask &= (b1_full != nodata)

            # For each possible top-left corner, check what fraction of the
            # patch would be valid using a sliding window sum
            from scipy.ndimage import uniform_filter
            valid_float = valid_mask.astype(np.float32)
            patch_valid_frac = uniform_filter(valid_float, size=patch_size, mode='constant')

            threshold = 1.0 - (max_invalid_percent / 100.0)

            # Restrict to corners where a full patch fits inside the raster
            margin = patch_size // 2
            interior = patch_valid_frac[margin: height - patch_size,
                                        margin: width  - patch_size]
            local_rows, local_cols = np.where(interior >= threshold)

            # Offset back to actual raster coordinates
            candidate_rows = local_rows + margin
            candidate_cols = local_cols + margin

            print(f"Found {len(candidate_rows):,} valid candidate locations")

            if len(candidate_rows) == 0:
                print("⚠️  No valid locations found! Try increasing --max_invalid.")
                return 0, 0
            # ─────────────────────────────────────────────────────────────

            rng = np.random.default_rng(seed)
            target = num_patches if num_patches else 10000
            max_attempts = target * 10  # safety cap (much less needed now)

            print(f"\nGenerating {target} patches via random sampling...")
            pbar = tqdm(total=target, desc="Generating patches")

            while saved < target and attempts < max_attempts:
                idx = int(rng.integers(0, len(candidate_rows)))
                top  = int(candidate_rows[idx])
                left = int(candidate_cols[idx])

                window = Window(left, top, patch_size, patch_size)
                patch = src.read(window=window)        # (C, H, W)
                patch = np.moveaxis(patch, 0, -1)      # (H, W, C)

                attempts += 1

                is_valid = check_patch_quality(
                    patch,
                    max_invalid_percent,
                    b1_band_index,
                    nodata
                )

                if is_valid:
                    filename = f"patch_{saved:05d}.tif"
                    filepath = os.path.join(output_dir, filename)

                    patch_out = np.moveaxis(patch, -1, 0)

                    profile = src.profile.copy()
                    profile.update({
                        'height': patch_size,
                        'width':  patch_size,
                        'transform': src.window_transform(window)
                    })

                    with rasterio.open(filepath, 'w', **profile) as dst:
                        dst.write(patch_out)

                    saved += 1
                    pbar.update(1)
                else:
                    dropped += 1

                pbar.set_postfix({
                    'saved':    saved,
                    'dropped':  dropped,
                    'success%': f'{saved / attempts * 100:.1f}' if attempts > 0 else '0.0'
                })

            pbar.close()

            if saved < target:
                print(f"\n⚠️  Warning: Only generated {saved}/{target} patches after {attempts} attempts")
                print(f"   Try increasing --max_invalid or using a larger input raster")

        else:
            # SYSTEMATIC GRID MODE (unchanged)
            if stride is None:
                stride = patch_size

            n_x = (width  - patch_size) // stride + 1
            n_y = (height - patch_size) // stride + 1
            total = n_x * n_y

            print(f"\nTotal possible patches: {total:,} ({n_x} x {n_y})")
            print("\nGenerating patches...")
            pbar = tqdm(total=total)

            for i in range(n_y):
                for j in range(n_x):
                    col = j * stride
                    row = i * stride
                    window = Window(col, row, patch_size, patch_size)
                    patch = src.read(window=window)
                    patch = np.moveaxis(patch, 0, -1)

                    attempts += 1

                    if patch.shape[:2] != (patch_size, patch_size):
                        pbar.update(1)
                        continue

                    is_valid = check_patch_quality(
                        patch,
                        max_invalid_percent,
                        b1_band_index,
                        nodata
                    )

                    if is_valid:
                        filename = f"patch_{i:05d}_{j:05d}.tif"
                        filepath = os.path.join(output_dir, filename)

                        patch_out = np.moveaxis(patch, -1, 0)

                        profile = src.profile.copy()
                        profile.update({
                            'height': patch_size,
                            'width':  patch_size,
                            'transform': src.window_transform(window)
                        })

                        with rasterio.open(filepath, 'w', **profile) as dst:
                            dst.write(patch_out)

                        saved += 1
                    else:
                        dropped += 1

                    pbar.update(1)
                    pbar.set_postfix({
                        'saved':   saved,
                        'dropped': dropped,
                        'keep%':   f'{saved / (saved + dropped) * 100:.1f}'
                    })

            pbar.close()

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Total attempts: {attempts:,}")
    print(f"Saved:   {saved:,} ({saved / attempts * 100:.1f}%)" if attempts > 0 else "Saved: 0")
    print(f"Dropped: {dropped:,} ({dropped / attempts * 100:.1f}%)" if attempts > 0 else "Dropped: 0")
    print(f"Output:  {output_dir}")
    print("=" * 70 + "\n")

    return saved, dropped


# def check_patch_quality(patch, max_invalid_percent, b1_band_index, nodata):
#     """
#     Check if patch has enough valid data.

#     Invalid pixels are:
#     - b1 = 0 (absent/invalid)
#     - NoData values
#     - NaN values

#     Returns True if valid, False if should be dropped.
#     """

#     b1 = patch[..., b1_band_index]
#     total_pixels = b1.size

#     # Count invalid pixels
#     invalid = 0 #training we uncomment and testing we comment

#     # 1. b1 = 0
#     invalid += np.sum(b1 == 0)

#     # 2. NaN
#     invalid += np.sum(np.isnan(b1))

#     # 3. NoData
#     # if nodata is not None:
#     #     invalid += np.sum(b1 == nodata)

#     # Calculate percentage
#     invalid_percent = (invalid / total_pixels) * 100
#     print(f"Total invalid pixels: {invalid:,}")

#     # Keep if below threshold
#     return invalid_percent <= max_invalid_percent

def check_patch_quality(patch, max_invalid_percent, b1_band_index, nodata):
    """
    Check if patch has enough valid data.
    Invalid pixels are NaN or NoData only.
    b1 = 0 means no crop presence — valid for training.
    """
    b1 = patch[..., b1_band_index]
    total_pixels = b1.size

    nan_count = np.sum(np.isnan(b1))
    nodata_count = np.sum(b1 == nodata) if nodata is not None else 0
    invalid = nan_count + nodata_count

    invalid_percent = (invalid / total_pixels) * 100

    # ADD THIS
    if invalid_percent > max_invalid_percent:
        print(f"[DROPPED] nan={nan_count}, nodata={nodata_count}, invalid%={invalid_percent:.1f}")

    return invalid_percent <= max_invalid_percent

def verify_patches(output_dir):
    """Quick verification of generated patches."""
    import glob

    patches = glob.glob(os.path.join(output_dir, "*.tif"))

    if not patches:
        print("No patches found!")
        return

    print(f"\n{'=' * 70}")
    print("VERIFICATION")
    print(f"{'=' * 70}")
    print(f"\nTotal patches: {len(patches)}")

    # Check 5 random samples
    samples = np.random.choice(patches, min(5, len(patches)), replace=False)

    print("\nChecking random samples:")
    for i, path in enumerate(samples, 1):
        with rasterio.open(path) as src:
            data = src.read()
            data = np.moveaxis(data, 0, -1)
            b1 = data[..., -1]

            valid = np.sum((b1 != 0) & ~np.isnan(b1))
            valid_pct = (valid / b1.size) * 100

            print(f"\n{i}. {os.path.basename(path)}")
            print(f"   Shape: {data.shape}")
            print(f"   Valid pixels: {valid}/{b1.size} ({valid_pct:.1f}%)")
            if valid > 0:
                print(f"   b1 range: [{b1[~np.isnan(b1)].min():.2f}, {b1[~np.isnan(b1)].max():.2f}]")

    print(f"\n{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Generate quality-filtered patches from large TIFF',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate exactly 1000 patches via random sampling
  python generate_patches.py -i large.tif -o patches/ --patch_size 16 --num_patches 1000

  # Systematic grid (all possible patches)
  python generate_patches.py -i large.tif -o patches/ --patch_size 16 --stride 16

  # Random sampling with stricter quality
  python generate_patches.py -i large.tif -o patches/ --num_patches 500 --max_invalid 5

  # Grid with overlap
  python generate_patches.py -i large.tif -o patches/ --patch_size 16 --stride 8
        """
    )

    parser.add_argument('-i', '--input', required=True,
                        help='Input TIFF file')
    parser.add_argument('-o', '--output', required=True,
                        help='Output directory')
    parser.add_argument('--patch_size', type=int, default=4, #16
                        help='Patch size (default: 16)')
    parser.add_argument('--num_patches', type=int, default=300, #change threshold here
                        help='Number of patches to generate (enables random sampling)')
    parser.add_argument('--stride', type=int, default=None,
                        help='Stride for grid sampling (default: patch_size, ignored if --num_patches used)')
    parser.add_argument('--max_invalid', type=float, default=50, #20
                        help='Max invalid pixel %% (default: 10)')
    parser.add_argument('--b1_band', type=int, default=-1,
                        help='b1 band index, -1=last (default: -1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify patches after generation')

    args = parser.parse_args()

    # Generate
    saved, dropped = generate_filtered_patches(
        args.input,
        args.output,
        args.patch_size,
        args.num_patches,
        args.stride,
        args.max_invalid,
        args.b1_band,
        seed=args.seed
    )

    # Verify
    if args.verify:
        verify_patches(args.output)

    # Warnings
    if saved == 0:
        print("⚠️  No patches saved! Try:")
        print("   - Increase --max_invalid")
        print("   - Check --b1_band index")
        print("   - Verify input TIFF has valid data")
    elif args.num_patches and saved < args.num_patches:
        print(f"⚠️  Only {saved}/{args.num_patches} patches generated!")
        print("   - Try lowering --max_invalid threshold")
        print("   - Or use larger input image")
    elif saved < 50 and args.num_patches is None:
        print(f"⚠️  Only {saved} patches saved. Consider:")
        print("   - Increasing --max_invalid threshold")
        print("   - Using smaller --stride for more overlap")


if __name__ == '__main__':
    main()