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
    version_base=None, config_path="configs", config_name="generate_lssty10_lognormal"
)
def main(cfg: DictConfig) -> None:
    """Main function for generating convergence maps."""
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

    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

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

        # Convert to torch tensors and save
        batch_data = {
            "maps": torch.from_numpy(np.array(maps_batch)),
            "params": torch.from_numpy(np.array(params_array)),
        }
        batch_path = output_dir / f"batch_{batch_idx:05d}.pt"
        torch.save(batch_data, batch_path)

    elapsed_time = time.time() - start_time
    maps_generated = remaining_batches * cfg.batch_size
    time_per_map = elapsed_time / maps_generated if maps_generated > 0 else 0

    print("\nGeneration complete!")
    print(f"Total time: {elapsed_time:.2f}s")
    print(f"Time per map: {time_per_map:.3f}s")
    print(f"Maps per second: {maps_generated / elapsed_time:.2f}")

    # Summary
    total_batches = len(list(output_dir.glob("batch_*.pt")))
    total_maps = total_batches * cfg.batch_size
    maps_size_mb = total_maps * cfg.N * cfg.N * cfg.nbins * 4 / (1024 * 1024)

    print(f"\nSaved {total_batches} batch files to {output_dir}")
    print(f"  - batch_00000.pt ... batch_{total_batches - 1:05d}.pt")
    print(f"  - Total maps: {total_maps} ({maps_size_mb:.1f} MB)")
    print(f"  - param_names.txt: {prior.param_names}")


if __name__ == "__main__":
    main()
