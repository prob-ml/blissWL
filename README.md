# blissWL

This repository is part of the [Bayesian Light Source Separator (BLISS)](https://github.com/prob-ml/bliss) software family. It implements two forms of neural probabilistic weak lensing inference. The first infers convergence maps from images, and the second infers LambdaCDM parameters from convergence maps.

## Installation

Create a virtual environment and install dependencies:

```bash
uv venv
source .venv/bin/activate
uv sync
```

For development dependencies (Jupyter, linting, etc.):

```bash
uv sync --group dev
```

## Images to maps

*Coming soon.* This module will implement neural posterior estimation (NPE) for inferring mass maps from full-field astronomical images.

## Maps to cosmology

This module implements NPE for inferring cosmological parameters from weak lensing convergence maps.

### Generate convergence maps

Generate synthetic convergence maps using the log-normal forward model from [`sbi_lens`](https://github.com/DifferentiableUniverseInitiative/sbi_lens):

```bash
python -m maps_to_cosmology.generate_maps
```

Override defaults with Hydra syntax:

```bash
python -m maps_to_cosmology.generate_maps num_maps=50000 output_dir=/path/to/output
```

### Train the encoder

Train the NPE encoder network:

```bash
python -m maps_to_cosmology.train
```

Training logs are saved to TensorBoard. View with:

```bash
tensorboard --logdir=maps_to_cosmology/results
```
