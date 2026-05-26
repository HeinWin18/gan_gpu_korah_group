import rasterio
from rasterio.windows import Window
import os
import numpy as np


# === Settings ===
input_tif = "presence_1_points_only.tif"
patch_size = 32
output_dir = "train_patches"
os.makedirs(output_dir, exist_ok=True)
num_patches = 300
seed = 42

# === Open raster ===
# with rasterio.open(input_tif) as src:
#     width, height = src.width, src.height
#     count = src.count
#     profile = src.profile.copy()
#
#     # Loop over grid of patches
#     patch_id = 0
#     for top in range(0, height, patch_size):
#         for left in range(0, width, patch_size):
#             # Make sure window doesn't exceed bounds
#             win_width = min(patch_size, width - left)
#             win_height = min(patch_size, height - top)
#             window = Window(left, top, win_width, win_height)
#
#             # Read the windowed patch
#             patch = src.read(window=window)
#
#             # Skip empty (all-nodata) patches
#             if np.isnan(patch).all():
#                 continue
#
#             # Update profile for the patch
#             patch_profile = profile.copy()
#             patch_profile.update({
#                 "height": win_height,
#                 "width": win_width,
#                 "transform": rasterio.windows.transform(window, src.transform)
#             })
#
#             # Write to new TIFF
#             out_path = os.path.join(output_dir, f"patch_{patch_id:05d}.tif")
#             with rasterio.open(out_path, "w", **patch_profile) as dst:
#                 dst.write(patch)
#
#             patch_id += 1






os.makedirs(output_dir, exist_ok=True)
rng = np.random.default_rng(seed)

with rasterio.open(input_tif) as src:
    h, w = src.height, src.width
    profile = src.profile.copy()

    patch_id = 0
    while patch_id < num_patches:
        # pick random top-left corner
        top = int(rng.integers(0, h - patch_size))
        left = int(rng.integers(0, w - patch_size))

        window = Window(left, top, patch_size, patch_size)
        patch = src.read(window=window, masked=True)

        # skip fully empty/nodata patches
        if np.ma.getmaskarray(patch).all():
            continue

        patch_profile = profile.copy()
        patch_profile.update({
            "height": patch_size,
            "width": patch_size,
            "transform": rasterio.windows.transform(window, src.transform)
        })

        out_path = os.path.join(output_dir, f"patch_{patch_id:05d}.tif")
        with rasterio.open(out_path, "w", **patch_profile) as dst:
            dst.write(patch.filled(src.nodata if src.nodata else 0))

        patch_id += 1
