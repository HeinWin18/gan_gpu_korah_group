import numpy as np
from scipy.linalg import sqrtm
from tensorflow.keras.models import load_model
import os
import rasterio
import matplotlib.pyplot as plt


def load_real_patches(folder, num=None):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".tif")])
    if num:
        files = files[:num]

    patches = []
    for f in files:
        with rasterio.open(os.path.join(folder, f)) as src:
            patch = src.read()  # shape (C, H, W)
            patch = np.transpose(patch, (1, 2, 0))  # → (H, W, C)
            patches.append(patch)

    return np.array(patches)  # shape (N, H, W, C)


def load_fake_patches(folder, num=None):
    files = sorted([f for f in os.listdir(folder) if f.endswith(".npy")])
    if num:
        files = files[:num]
    return np.array([np.load(os.path.join(folder, f)) for f in files])

def compute_fid(real_patches, fake_patches):
    print("Flattening patches...")
    real_flat = real_patches.reshape(len(real_patches), -1).astype(np.float32)
    fake_flat = fake_patches.reshape(len(fake_patches), -1).astype(np.float32)

    print("Computing means...")
    mu_real = np.mean(real_flat, axis=0)
    mu_fake = np.mean(fake_flat, axis=0)

    print("Computing covariances...")  # likely slow here
    sigma_real = np.cov(real_flat, rowvar=False)
    sigma_fake = np.cov(fake_flat, rowvar=False)

    # Add small regularization to avoid numerical issues
    eps = 1e-6
    sigma_real += np.eye(sigma_real.shape[0]) * eps
    sigma_fake += np.eye(sigma_fake.shape[0]) * eps

    print("Computing matrix square root...")  # or here
    diff = mu_real - mu_fake
    covmean = sqrtm(sigma_real @ sigma_fake)

    print("Finalizing...")
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
    return fid

# Load patches
print("Loading patches...")
real_patches = load_real_patches("Grape/TrainPatches-Cali")
real_patches = real_patches[:, :, :, :-1]
real_patches_clean = real_patches.copy()

for i in range(real_patches.shape[-1]):
    band = real_patches_clean[:, :, :, i]
    band_mean = np.nanmean(band)
    band[np.isnan(band)] = band_mean

print("NaNs remaining:", np.isnan(real_patches_clean).any())
print("Real patches:", real_patches_clean.shape)  # should still be (400, 4, 4, 19)

fake_patches = load_fake_patches("Grape/FakePatches")

print(f"Fake patches: {fake_patches.shape}")

print("Real min/max:", real_patches_clean.min(), real_patches_clean.max())
print("Fake min/max:", fake_patches.min(), fake_patches.max())



bands = []
real_means, fake_means = [], []
real_stds, fake_stds = [], []
real_mins, fake_mins = [], []
real_maxs, fake_maxs = [], []

# for i in range(real_patches_clean.shape[-1]):
#     real_band = real_patches_clean[:, :, :, i].flatten()
#     fake_band = fake_patches[:, :, :, i].flatten()
#
#     bands.append(f'B{i + 1}')
#     real_means.append(real_band.mean())
#     fake_means.append(fake_band.mean())
#     real_stds.append(real_band.std())
#     fake_stds.append(fake_band.std())
#
#     print(
#         f"Band {i + 1:2d} | Real mean: {real_band.mean():8.2f}  Fake mean: {fake_band.mean():8.2f} | Real std: {real_band.std():8.2f}  Fake std: {fake_band.std():8.2f}")
#
# x = np.arange(len(bands))
# width = 0.35
#
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
#
# ax1.bar(x - width / 2, real_means, width, label='Real', color='steelblue')
# ax1.bar(x + width / 2, fake_means, width, label='Fake', color='coral')
# ax1.set_title('Mean per Band — Real vs Fake')
# ax1.set_xticks(x)
# ax1.set_xticklabels(bands)
# ax1.legend()
# ax1.set_ylabel('Mean value')
#
# ax2.bar(x - width / 2, real_stds, width, label='Real', color='steelblue')
# ax2.bar(x + width / 2, fake_stds, width, label='Fake', color='coral')
# ax2.set_title('Std per Band — Real vs Fake')
# ax2.set_xticks(x)
# ax2.set_xticklabels(bands)
# ax2.legend()
# ax2.set_ylabel('Std value')
#
# plt.tight_layout()
# plt.savefig('band_comparison.png', dpi=150)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np

bands = []
real_means, fake_means = [], []
real_stds, fake_stds = [], []
real_mins, fake_mins = [], []
real_maxs, fake_maxs = [], []

for i in range(real_patches_clean.shape[-1]):
    real_band = real_patches_clean[:, :, :, i].flatten()
    fake_band = fake_patches[:, :, :, i].flatten()

    bands.append(f'B{i + 1}')
    real_means.append(real_band.mean())
    fake_means.append(fake_band.mean())
    real_stds.append(real_band.std())
    fake_stds.append(fake_band.std())
    real_mins.append(real_band.min())
    fake_mins.append(fake_band.min())
    real_maxs.append(real_band.max())
    fake_maxs.append(fake_band.max())

    print(f"Band {i + 1:2d} | "
          f"Real mean: {real_band.mean():8.2f}  Fake mean: {fake_band.mean():8.2f} | "
          f"Real std: {real_band.std():8.2f}  Fake std: {fake_band.std():8.2f} | "
          f"Real min: {real_band.min():8.2f}  Fake min: {fake_band.min():8.2f} | "
          f"Real max: {real_band.max():8.2f}  Fake max: {fake_band.max():8.2f}")

x = np.arange(len(bands))
width = 0.35

fig, axes = plt.subplots(2, 1, figsize=(14, 12))

for ax, real, fake, title, ylabel in zip(
    axes,
    [real_means, real_stds],
    [fake_means, fake_stds],
    ['Mean', 'Std'],
    ['Mean value', 'Std value']
):
    ax.bar(x - width/2, real, width, label='Real', color='steelblue')
    ax.bar(x + width/2, fake, width, label='Fake', color='coral')
    ax.set_title(f'{title} per Band — Real vs Fake')
    ax.set_xticks(x)
    ax.set_xticklabels(bands)
    ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_yscale('log')  # log scale

plt.tight_layout(pad=3.0)
plt.savefig('band_comparison.png', dpi=150)
plt.show()


# # Compute FID
# print("Computing FID...")
# fid = compute_fid(real_patches_clean, fake_patches)
# print(f"FID Score: {fid:.4f}")