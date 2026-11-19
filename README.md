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

#### Generate cached data

```bash
python -u images_to_maps/dc2/generate_dc2_cache.py --write-catalogs
```

This reads the DC2 coadds and truth/object tables from `/nfs/turbo/lsa-regier/dc2_nersc`, joins them to CosmoDC2 lensing quantities, and writes overlapping 2048 x 2048 padded crops (images and 8 x 8 tile target maps) to `/nfs/turbo/lsa-regier/dc2_wl`. The crops lie on each tract's pixel grid (27 x 27 crops per tract), and their central 4 x 4 tiles tile the tract with no gaps or overlaps. The tracts come from `surveys.dc2.tracts` in `config_train_npe.yaml`. With `--write-catalogs`, every merged per-tract galaxy catalog the generator builds is also saved to `/nfs/turbo/lsa-regier/dc2_wl_catalog/tract_<tract>.parquet`. This includes neighboring tracts whose galaxies fall in crops near the tract edges. The exploratory notebooks read these catalogs. Existing files are skipped unless `--overwrite` is passed.

#### Train

Mean-field Gaussian NPE (`MassMapEncoder`):

```bash
python -m images_to_maps.dc2.train
```

Flow matching:

```bash
python -m images_to_maps.dc2.train --config-name=config_train_flowmatching
```

Both configs train on the padded crops with the network from the paper (`ch_max=1024`, `initial_downsample=false`, `more_up_layers=true`), which needs a GPU with more than 11 GB of memory. The same configs are used on every server; override paths on the command line if needed (e.g. `paths.dc2_cache=...`). Run training from the repository root, since logs and checkpoints are written to `images_to_maps/results/dc2/{npe,flowmatching}/<version>/` relative to the working directory.

#### Sample (flow matching)

```bash
python -m images_to_maps.dc2.stitch_flowmatching_samples --split all \
    --ckpt images_to_maps/results/dc2/flowmatching/<version>/checkpoints/encoder_epoch=<epoch>.ckpt \
    --output <output_dir> --num-samples 100
```

This draws posterior samples of the shear and convergence maps for each whole tract in `surveys.dc2.tracts`. It integrates a single ODE over the tract, stitching the velocities from each crop's interior at every step, so neighboring crops share one sample. For each tract it writes `tract_<tract>_global_flowmatching_stitched_samples.pt` (samples, posterior mean, true maps, and RA/Dec) and a metrics JSON. Use `--split all`: the sampler needs every crop in a tract, and the test region can be cut out of the tract maps afterwards. Pass `--ckpt` explicitly, because the default checkpoint is from an earlier run. Run `--help` for the other options.

#### Notebooks

- **Results**: `images_to_maps/dc2/results/credibleintervals.ipynb`, `crosscorrelation.ipynb`, `flowmatching.ipynb`, `posteriormeanmaps.ipynb`
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
tensorboard --logdir=images_to_maps/results/dc2
```

or for descwl:

```bash
tensorboard --logdir=images_to_maps/results/descwl
```

## Maps to cosmology

This module implements inference of cosmological parameters from weak lensing convergence maps. Two methods are supported:

1. **NPE** (neural posterior estimation): Infers a full-rank multivariate normal variational distribution over cosmological parameters.
2. **Flow matching**: Approximates the posterior distribution using a velocity field that transports noise to cosmological parameters via ODE integration.

### Generate convergence maps

Generate synthetic convergence maps using the log-normal forward model from [`sbi_lens`](https://github.com/DifferentiableUniverseInitiative/sbi_lens):

```bash
# LSST Y10 settings (default: 256x256, 10° field)
python -m maps_to_cosmology.generate_maps

# DC2 settings (64x40, 0.91° field)
python -m maps_to_cosmology.generate_maps --config-name=generate_dc2_lognormal
```

Override defaults with Hydra syntax:

```bash
python -m maps_to_cosmology.generate_maps num_maps=50000 output_dir=/path/to/output
```

### Train the encoder

Train with NPE:

```bash
# LSST Y10
python -m maps_to_cosmology.train --config-name=train_npe_lsst_y10

# DC2
python -m maps_to_cosmology.train --config-name=train_npe_dc2
```

Train with flow matching:

```bash
# LSST Y10
python -m maps_to_cosmology.train --config-name=train_flowmatching_lsst_y10

# DC2
python -m maps_to_cosmology.train --config-name=train_flowmatching_dc2
```

Training logs are saved to TensorBoard. View with:

```bash
tensorboard --logdir=maps_to_cosmology/results
```
