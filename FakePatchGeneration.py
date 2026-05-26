import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os

os.makedirs("Grape/FakePatches", exist_ok=True)



# Load saved stats from training
mean = np.load("train_mean.npy")  # shape (1,1,1,C)
std = np.load("train_std.npy")  # shape (1,1,1,C)

# Load generator
generator = load_model("Grape/savedModels/generator_20260420_143916.h5", compile=False)
print("Input shape:", generator.input_shape)
print("Output shape:", generator.output_shape)


latent_dim = generator.input_shape[-1]

print(f"Latent dim: {latent_dim}")
print(f"Output shape: {generator.output_shape}")
print(f"Bands: {mean.shape[-1]}")

# Generate
num_images = 400

for i in range(num_images):
    z = np.random.randn(1, latent_dim)
    fake = generator.predict(z, verbose=0)  # z-score space, shape (1,H,W,C)
    fake_denorm = (fake * std) + mean        # denormalize to physical units
    np.save(f"Grape/FakePatches/{i}.npy", fake_denorm[0])

    if (i + 1) % 50 == 0:
        print(f"Generated {i + 1}/{num_images}")

print("Done!")