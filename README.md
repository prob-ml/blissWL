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

This module implements neural posterior estimation (NPE) for inferring weak lensing shear and convergence from LSST-like images. It supports two applications:

1. **DC2**: Infer tomographic mass maps for the [DC2 Simulated Sky Survey](https://data.lsstdesc.org/doc/dc2_sim_sky_survey)
2. **descwl-shear-sims**: Infer constant shear from images generated with [`descwl-shear-sims`](https://github.com/timwhite0/descwl-shear-sims)

### DC2

#### Generate catalog

```bash
python -u images_to_maps/dc2/generate_catalog.py
```

#### Train MassMapEncoder

```bash
python -m images_to_maps.dc2.train
```

Or with Hydra config override:

```bash
python -m images_to_maps.dc2.train --config-path=. --config-name=config_train_npe
```

#### Notebooks

- **Results**: `images_to_maps/dc2/results/credibleintervals.ipynb`, `posteriormeanmaps.ipynb`
- **Exploratory**: `images_to_maps/dc2/exploratory/dc2imageandmaps.ipynb`, `ellipticity.ipynb`, `galaxyproperties.ipynb`, `twopoint.ipynb`

### descwl-shear-sims

#### Train ScalarShearEncoder

```bash
python -m images_to_maps.descwl.train
```

Or with Hydra config override:

```bash
python -m images_to_maps.descwl.train --config-path=. --config-name=config_train_npe
```

#### Run AnaCal

Configure settings in `images_to_maps/descwl/config_run_anacal.yaml`.

```bash
python -u images_to_maps/descwl/run_anacal.py
```

#### Notebooks

- **Results**: `images_to_maps/descwl/results/compute_npe_credibleintervals.py`, `credibleintervals.ipynb`, `scatterplots.ipynb`
- **Exploratory**: `images_to_maps/descwl/exploratory/images.ipynb`

### View training logs

Training logs are saved to TensorBoard. View with:

```bash
tensorboard --logdir=images_to_maps/dc2/results
```

or for descwl:

```bash
tensorboard --logdir=images_to_maps/descwl/results
```

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
