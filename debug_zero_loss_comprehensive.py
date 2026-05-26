#!/usr/bin/env python
# coding: utf-8
"""
Comprehensive Zero Loss Debugger
Tests every component to find why loss is exactly zero.
"""

import numpy as np
import tensorflow as tf
import argparse
import sys
import os

# Add project path
sys.path.insert(0, '.')


def test_data_loading(args):
    """Test 1: Data Loading"""
    print("\n" + "=" * 70)
    print("TEST 1: DATA LOADING")
    print("=" * 70)

    try:
        import load
        X_train, _ = load.load_raster_data(
            args.datapath,
            None,
            args.imgsize,
            mode='train',
            scale_mode='zscore'
        )

        print(f"✓ Data loaded: {X_train.shape}")
        print(f"  dtype: {X_train.dtype}")
        print(f"  range: [{X_train.min():.4f}, {X_train.max():.4f}]")
        print(f"  mean: {X_train.mean():.4f}")
        print(f"  std: {X_train.std():.4f}")

        # Check for issues
        if np.isnan(X_train).any():
            print(f"  ⚠️  Contains {np.isnan(X_train).sum()} NaNs")
        if np.isinf(X_train).any():
            print(f"  ⚠️  Contains {np.isinf(X_train).sum()} Infs")
        if X_train.min() == X_train.max():
            print(f"  ⚠️  All values are constant!")
            return None

        return X_train

    except Exception as e:
        print(f"✗ Data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_mask_generation(X_train, args):
    """Test 2: Mask Generation"""
    print("\n" + "=" * 70)
    print("TEST 2: MASK GENERATION")
    print("=" * 70)

    try:
        from patchGan import build_keep_mask_from_b1

        # Take a small batch
        x_batch = tf.constant(X_train[:4], dtype=tf.float32)

        print(f"Batch shape: {x_batch.shape}")
        print(f"Last channel (b1) range: [{x_batch[..., -1].numpy().min():.4f}, {x_batch[..., -1].numpy().max():.4f}]")

        # Generate mask
        mask = build_keep_mask_from_b1(x_batch)

        print(f"\nMask generated:")
        print(f"  Shape: {mask.shape}")
        print(f"  Sum: {tf.reduce_sum(mask).numpy()}")
        print(f"  Max: {tf.reduce_max(mask).numpy()}")
        print(f"  Min: {tf.reduce_min(mask).numpy()}")
        print(f"  Valid pixels: {tf.reduce_sum(mask).numpy()} / {mask.size}")

        if tf.reduce_sum(mask).numpy() == 0:
            print("\n  ❌ PROBLEM: Mask is all zeros!")

            # Debug why
            b1 = x_batch[..., -1].numpy()
            print(f"\n  Debugging b1 band:")
            print(f"    Unique values: {np.unique(b1)[:20]}")
            print(f"    Values == 1.0: {np.sum(b1 == 1.0)}")
            print(f"    Values > 0: {np.sum(b1 > 0)}")
            print(f"    Values > 0.5: {np.sum(b1 > 0.5)}")

            return mask, False
        else:
            print(f"  ✓ Mask has {tf.reduce_sum(mask).numpy():.0f} valid pixels")
            return mask, True

    except Exception as e:
        print(f"✗ Mask generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_model_forward_pass(args):
    """Test 3: Model Forward Pass"""
    print("\n" + "=" * 70)
    print("TEST 3: MODEL FORWARD PASS")
    print("=" * 70)

    try:
        from model import generator_model, patchgan_discriminator_model

        # Build models
        print("Building generator...")
        g = generator_model(args.zdims, args.imgsize, args.channels)
        print(f"  ✓ Generator built: {g.input_shape} → {g.output_shape}")

        print("\nBuilding discriminator...")
        d = patchgan_discriminator_model(args.imgsize, args.channels)
        print(f"  ✓ Discriminator built: {d.input_shape} → {d.output_shape}")

        # Test forward pass
        z_test = tf.random.normal([4, args.zdims])
        x_fake = g(z_test, training=False)

        print(f"\n✓ Generator forward pass:")
        print(f"  Input: {z_test.shape}")
        print(f"  Output: {x_fake.shape}")
        print(f"  Range: [{tf.reduce_min(x_fake):.4f}, {tf.reduce_max(x_fake):.4f}]")
        print(f"  Mean: {tf.reduce_mean(x_fake):.4f}")

        # Check for NaN/Inf
        if tf.reduce_any(tf.math.is_nan(x_fake)):
            print("  ⚠️  Generator output contains NaN!")
            return None, None, False
        if tf.reduce_any(tf.math.is_inf(x_fake)):
            print("  ⚠️  Generator output contains Inf!")
            return None, None, False

        # Test discriminator
        x_real = tf.random.normal([4, args.imgsize, args.imgsize, args.channels])
        d_out = d(x_real, training=False)

        print(f"\n✓ Discriminator forward pass:")
        print(f"  Input: {x_real.shape}")
        print(f"  Output: {d_out.shape}")
        print(f"  Range: [{tf.reduce_min(d_out):.4f}, {tf.reduce_max(d_out):.4f}]")

        # Check discriminator output
        if tf.reduce_any(tf.math.is_nan(d_out)):
            print("  ⚠️  Discriminator output contains NaN!")
            return None, None, False

        return g, d, True

    except Exception as e:
        print(f"✗ Model building/forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, False


def test_loss_computation(X_train, mask, g, d, args, mask_is_valid):
    """Test 4: Loss Computation"""
    print("\n" + "=" * 70)
    print("TEST 4: LOSS COMPUTATION")
    print("=" * 70)

    try:
        from patchGan import masked_disc_loss, masked_adv_loss_for_generator, masked_pixel_l1_loss

        # Prepare batch
        x_batch = tf.constant(X_train[:4], dtype=tf.float32)
        z = tf.random.normal([4, args.zdims])

        print(f"Test batch shape: {x_batch.shape}")

        # Generate fake
        x_fake = g(z, training=False)
        print(f"Generated fake: {x_fake.shape}")

        # Test with mask
        if mask is not None and mask_is_valid:
            print(f"\nUsing actual mask (sum={tf.reduce_sum(mask).numpy():.0f})")
            test_mask = mask
        else:
            print(f"\nMask invalid, creating all-ones mask for testing")
            test_mask = tf.ones((4, args.imgsize, args.imgsize, 1), dtype=tf.float32)

        # Compute D loss
        print("\nComputing discriminator loss...")
        d_loss = masked_disc_loss(d, x_batch, x_fake, test_mask)
        print(f"  D loss: {d_loss.numpy():.6f}")

        if d_loss.numpy() == 0.0:
            print("  ❌ D loss is exactly zero!")

            # Debug components
            logits_real = d(x_batch, training=False)
            logits_fake = d(x_fake, training=False)

            print(f"\n  Debugging D loss components:")
            print(f"    logits_real range: [{tf.reduce_min(logits_real):.4f}, {tf.reduce_max(logits_real):.4f}]")
            print(f"    logits_fake range: [{tf.reduce_min(logits_fake):.4f}, {tf.reduce_max(logits_fake):.4f}]")

            loss_real_map = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(logits_real),
                logits=logits_real
            )
            loss_fake_map = tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(logits_fake),
                logits=logits_fake
            )

            print(f"    loss_real_map: {tf.reduce_mean(loss_real_map):.6f}")
            print(f"    loss_fake_map: {tf.reduce_mean(loss_fake_map):.6f}")
            print(f"    test_mask sum: {tf.reduce_sum(test_mask).numpy():.0f}")

            masked_loss_real = tf.reduce_sum(loss_real_map * test_mask)
            masked_loss_fake = tf.reduce_sum(loss_fake_map * test_mask)
            denom = tf.reduce_sum(test_mask) + 1e-8

            print(f"    masked_loss_real: {masked_loss_real.numpy():.6f}")
            print(f"    masked_loss_fake: {masked_loss_fake.numpy():.6f}")
            print(f"    denominator: {denom.numpy():.6f}")
            print(f"    final: {((masked_loss_real + masked_loss_fake) / denom / 2).numpy():.6f}")

        # Compute G adv loss
        print("\nComputing generator adversarial loss...")
        g_adv = masked_adv_loss_for_generator(d, x_fake, test_mask)
        print(f"  G adv loss: {g_adv.numpy():.6f}")

        # Compute L1 loss
        print("\nComputing L1 pixel loss...")
        l1_loss = masked_pixel_l1_loss(x_batch, x_fake, test_mask)
        print(f"  L1 loss: {l1_loss.numpy():.6f}")

        # Combined G loss
        g_total = g_adv + 100.0 * l1_loss
        print(f"\nCombined G loss (adv + 100*L1): {g_total.numpy():.6f}")

        # Summary
        print("\n" + "-" * 70)
        if d_loss.numpy() == 0.0 or g_total.numpy() == 0.0:
            print("❌ ZERO LOSS CONFIRMED")
            return False
        else:
            print("✓ Losses are non-zero")
            return True

    except Exception as e:
        print(f"✗ Loss computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_step(X_train, args):
    """Test 5: Full Training Step"""
    print("\n" + "=" * 70)
    print("TEST 5: FULL TRAINING STEP")
    print("=" * 70)

    try:
        from dcgan import DCGAN

        # Initialize DCGAN
        print("Initializing DCGAN...")
        dcgan = DCGAN(args)
        print("  ✓ DCGAN initialized")

        # Take a batch
        x_batch = X_train[:args.batchsize]
        print(f"\nTraining batch shape: {x_batch.shape}")

        # Test WITH masking
        print("\n--- Testing WITH masking ---")
        d_loss, g_loss, adv_loss, l1_loss = dcgan.train(x_batch, l1_weight=100.0, use_masking=True)

        print(f"D loss: {float(d_loss):.6f}")
        print(f"G loss: {float(g_loss):.6f}")
        print(f"Adv loss: {float(adv_loss):.6f}")
        print(f"L1 loss: {float(l1_loss):.6f}")

        if float(d_loss) == 0.0 or float(g_loss) == 0.0:
            print("\n❌ ZERO LOSS WITH MASKING")

            # Test WITHOUT masking
            print("\n--- Testing WITHOUT masking ---")
            d_loss2, g_loss2, adv_loss2, l1_loss2 = dcgan.train(x_batch, l1_weight=100.0, use_masking=False)

            print(f"D loss: {float(d_loss2):.6f}")
            print(f"G loss: {float(g_loss2):.6f}")
            print(f"Adv loss: {float(adv_loss2):.6f}")
            print(f"L1 loss: {float(l1_loss2):.6f}")

            if float(d_loss2) == 0.0 or float(g_loss2) == 0.0:
                print("\n❌ ZERO LOSS EVEN WITHOUT MASKING!")
                print("   This indicates a problem beyond masking.")
                return False
            else:
                print("\n✓ NON-ZERO LOSS WITHOUT MASKING")
                print("   ➜ SOLUTION: Use use_masking=False")
                return True
        else:
            print("\n✓ NON-ZERO LOSSES - Training is working!")
            return True

    except Exception as e:
        print(f"✗ Training step failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description='Comprehensive zero loss debugger')
    parser.add_argument('--datapath', '-d', required=True)
    parser.add_argument('--imgsize', type=int, default=8)
    parser.add_argument('--channels', type=int, default=20)
    parser.add_argument('--zdims', type=int, default=100)
    parser.add_argument('--batchsize', type=int, default=4)

    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("COMPREHENSIVE ZERO LOSS DEBUGGER")
    print("=" * 70)
    print(f"Data path: {args.datapath}")
    print(f"Image size: {args.imgsize}x{args.imgsize}")
    print(f"Channels: {args.channels}")
    print("=" * 70)

    # Run all tests
    X_train = test_data_loading(args)
    if X_train is None:
        print("\n❌ Cannot proceed - data loading failed")
        return

    mask, mask_valid = test_mask_generation(X_train, args)

    g, d, models_ok = test_model_forward_pass(args)
    if not models_ok:
        print("\n❌ Cannot proceed - model building/forward pass failed")
        return

    losses_ok = test_loss_computation(X_train, mask, g, d, args, mask_valid)

    training_ok = test_training_step(X_train, args)

    # Final summary
    print("\n" + "=" * 70)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 70)

    if training_ok:
        print("\n✓ Training step produces non-zero losses")
        print("\nYour system is working correctly!")
        if not mask_valid:
            print("Note: Mask is all zeros, recommend use_masking=False")
    else:
        print("\n❌ Zero loss issue confirmed")
        print("\nMost likely causes:")
        if mask is not None and not mask_valid:
            print("  1. Mask is all zeros (MOST LIKELY)")
            print("     Fix: use_masking=False or fix masking condition")
        print("  2. Data normalization issue")
        print("  3. Model initialization problem")
        print("  4. Loss function bug")
        print("\nRecommendation: Review the detailed output above for specific issues")

    print("=" * 70 + "\n")


if __name__ == '__main__':
    main()