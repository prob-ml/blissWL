"""Generate weak lensing convergence maps using sbi_lens.

This script generates convergence maps using the log-normal forward model
from sbi_lens with LSST Y10 survey settings. Uses JAX vmap for efficient
batched generation on GPU.

Usage:
    python generate_maps.py                          # Use default config
    python generate_maps.py num_maps=5000            # Override num_maps
    python generate_maps.py output_dir=/custom/path  # Override output path
    python generate_maps.py batch_size=32            # Adjust batch size for GPU memory
"""

import time
from functools import partial
from pathlib import Path

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import torch
from jax import random, vmap
from numpyro.handlers import condition, seed, trace
from omegaconf import DictConfig
from tqdm import tqdm

from sbi_lens.simulator.LogNormal_field import lensingLogNormal

from maps_to_cosmology.prior import LambdaCDMPrior


def combine_batches(output_dir: Path) -> None:
    """Combine all batch files into a single combined_batches.pt file.

    Handles variable batch sizes (due to NaN filtering).

    Args:
        output_dir: Directory containing batch_*.pt files
    """
    batch_files = sorted(output_dir.glob("batch_*.pt"))
    if not batch_files:
        print(f"No batch_*.pt files found in {output_dir}")
        return

    print(f"\nCombining {len(batch_files)} batch files...")

    # First pass: count total samples (batches may have different sizes due to NaN filtering)
    total_samples = 0
    for batch_path in tqdm(batch_files, desc="Counting samples"):
        batch = torch.load(batch_path, weights_only=True)
        total_samples += batch["maps"].shape[0]
        # Get shapes and dtypes from first batch
        if batch_path == batch_files[0]:
            map_shape = batch["maps"].shape[1:]
            param_shape = batch["params"].shape[1:]
            maps_dtype = batch["maps"].dtype
            params_dtype = batch["params"].dtype
        del batch

    print(f"Total samples across all batches: {total_samples}")

    # Pre-allocate tensors
    combined_maps = torch.empty((total_samples, *map_shape), dtype=maps_dtype)
    combined_params = torch.empty((total_samples, *param_shape), dtype=params_dtype)

    # Second pass: fill in data
    current_idx = 0
    for batch_path in tqdm(batch_files, desc="Loading batches"):
        batch = torch.load(batch_path, weights_only=True)
        batch_size = batch["maps"].shape[0]
        combined_maps[current_idx : current_idx + batch_size] = batch["maps"]
        combined_params[current_idx : current_idx + batch_size] = batch["params"]
        current_idx += batch_size
        del batch

    print(f"Combined maps shape: {combined_maps.shape}")
    print(f"Combined params shape: {combined_params.shape}")

    # Save combined file
    combined_path = output_dir / "combined_batches.pt"
    torch.save({"maps": combined_maps, "params": combined_params}, combined_path)

    combined_size_mb = combined_path.stat().st_size / (1024 * 1024)
    print(f"Saved combined_batches.pt ({combined_size_mb:.1f} MB)")


def sample_single(model, key, params_dict):
    """Sample a single convergence map with fixed cosmological parameters.

    Args:
        model: Partial function of lensingLogNormal with fixed survey parameters
        key: JAX PRNG key
        params_dict: Dict of cosmological parameter values to condition on

    Returns:
        Convergence map (kappa)
    """
    model_trace = trace(seed(condition(model, data=params_dict), key)).get_trace()
    return model_trace["y"]["value"]


def create_batched_sampler(model, param_names):
    """Create a vmapped sampler for efficient batch generation.

    Args:
        model: Partial function of lensingLogNormal with fixed survey parameters
        param_names: List of cosmological parameter names

    Returns:
        Function that takes batched keys and params and returns batched maps
    """

    def _sample(key, params_dict):
        return sample_single(model, key, params_dict)

    # vmap over both key and each parameter value
    return vmap(_sample, in_axes=(0, {name: 0 for name in param_names}))


@hydra.main(
    version_base=None, config_path="configs", config_name="generate_lsst_y10_lognormal"
)
def main(cfg: DictConfig) -> None:
    """Main function for generating convergence maps."""
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Handle combine_only mode
    if cfg.get("combine_only", False):
        print("Combine-only mode: skipping generation, combining existing batches")
        combine_batches(output_dir)
        return

    # Print device info
    devices = jax.devices()
    print(f"JAX devices: {devices}")
    print(f"Using device: {devices[0]}")

    print(f"\nGenerating {cfg.num_maps} convergence maps")
    print(f"Output directory: {cfg.output_dir}")
    print(f"Random seed: {cfg.seed}")
    print(f"Map resolution: {cfg.N}x{cfg.N}")
    print(f"Redshift bins: {cfg.nbins}")
    print(f"Batch size: {cfg.batch_size}")

    # Create prior and save parameter names
    prior = LambdaCDMPrior(cfg.prior)
    param_names_path = output_dir / "param_names.txt"
    with open(param_names_path, "w") as f:
        f.write(",".join(prior.param_names) + "\n")

    # Check for existing progress
    existing_batches = sorted(output_dir.glob("batch_*.pt"))
    start_batch = len(existing_batches)

    if start_batch > 0:
        print(f"\nFound existing progress: {start_batch} batches")
        print(f"Resuming from batch {start_batch}...")

    # Set up the model with simulation parameters
    model = partial(
        lensingLogNormal,
        N=cfg.N,
        map_size=cfg.map_size,
        gal_per_arcmin2=cfg.gals_per_arcmin2,
        sigma_e=cfg.sigma_e,
        nbins=cfg.nbins,
        a=cfg.a,
        b=cfg.b,
        z0=cfg.z0,
        model_type=cfg.model_type,
        lognormal_shifts="LSSTY10",
        with_noise=cfg.with_noise,
    )

    # Create batched sampler
    batched_sampler = create_batched_sampler(model, prior.param_names)

    # Initialize random key and advance to correct position
    key = random.PRNGKey(cfg.seed)
    for _ in range(start_batch):
        key, *_ = random.split(
            key, cfg.batch_size + 2
        )  # +2 for prior key and batch keys

    # Calculate number of batches
    num_batches = (cfg.num_maps + cfg.batch_size - 1) // cfg.batch_size
    remaining_batches = num_batches - start_batch

    if remaining_batches <= 0:
        print(
            f"\nAlready have {start_batch * cfg.batch_size} or more maps. Nothing to do."
        )
        return

    # Generate maps in batches
    print(f"\nGenerating {remaining_batches} batches...")
    if start_batch == 0:
        print("(First batch includes JIT compilation and will be slower)")
    start_time = time.time()
    total_dropped = 0

    for batch_idx in tqdm(range(start_batch, num_batches), desc="Generating batches"):
        # Sample cosmological parameters from prior
        key, prior_key = random.split(key)
        params_batch = prior.sample_batch(prior_key, cfg.batch_size)

        # Generate batch of keys for map generation
        key, *batch_keys = random.split(key, cfg.batch_size + 1)
        batch_keys = jnp.stack(batch_keys)

        # Generate batch of maps with fixed cosmological parameters
        maps_batch = batched_sampler(batch_keys, params_batch)

        # Wait for computation to complete (for accurate timing)
        jax.block_until_ready(maps_batch)

        # Stack params into array [batch_size, num_params]
        params_array = jnp.stack(
            [params_batch[name] for name in prior.param_names], axis=1
        )

        # Convert to numpy for NaN checking
        maps_np = np.array(maps_batch)
        params_np = np.array(params_array)

        # Filter out samples with NaN values in maps or params
        maps_has_nan = np.isnan(maps_np).any(axis=(1, 2, 3))
        params_has_nan = np.isnan(params_np).any(axis=1)
        valid_mask = ~(maps_has_nan | params_has_nan)

        num_dropped = (~valid_mask).sum()
        if num_dropped > 0:
            total_dropped += num_dropped
            maps_np = maps_np[valid_mask]
            params_np = params_np[valid_mask]

        # Skip saving if all samples were dropped
        if len(maps_np) == 0:
            continue

        # Convert to torch tensors and save
        batch_data = {
            "maps": torch.from_numpy(maps_np),
            "params": torch.from_numpy(params_np),
        }
        batch_path = output_dir / f"batch_{batch_idx:05d}.pt"
        torch.save(batch_data, batch_path)

    elapsed_time = time.time() - start_time
    maps_attempted = remaining_batches * cfg.batch_size
    maps_generated = maps_attempted - total_dropped
    time_per_map = elapsed_time / maps_generated if maps_generated > 0 else 0

    print("\nGeneration complete!")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Time per map: {time_per_map:.3f}s")
    print(f"Maps per second: {maps_generated / elapsed_time:.2f}")
    if total_dropped > 0:
        print(
            f"Dropped {total_dropped} samples with NaN values ({100 * total_dropped / maps_attempted:.2f}%)"
        )

    # Summary - count actual samples in batch files (may vary due to NaN filtering)
    total_batches = len(list(output_dir.glob("batch_*.pt")))
    print(f"\nSaved {total_batches} batch files to {output_dir}")
    print(f"  - batch_00000.pt ... batch_{total_batches - 1:05d}.pt")
    print(f"  - param_names.txt: {prior.param_names}")

    # Combine all batches into a single file for fast loading
    combine_batches(output_dir)


if __name__ == "__main__":
    main()
