![](https://raw.githubusercontent.com/nmearl/feadme/refs/heads/main/images/feadme_logo_wide.png)

# Fast Elliptical Accretion Disk Modeling Engine

`feadme` fits elliptical accretion disk models to double-peaked emission line
profiles via Bayesian inference. It implements the disk model from
[Eracleous et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995ApJ...438..610E/abstract)
and uses JAX and NumPyro for JIT-compiled, gradient-based NUTS sampling.

## Features

- JAX-accelerated disk integration with GPU support
- NUTS posterior sampling via NumPyro, with NeuTra available for experiments
- Multiple initialization strategies, including SVI, Pathfinder, MAP, AutoDelta-style MAP, and JAX-LSQ
- JSON-driven model templates with multi-component and shared-parameter support
- Disk radial extent sampled with `radius_ratio`; `outer_radius` is derived as `inner_radius * radius_ratio`

## Installation

```bash
pip install feadme         # CPU
pip install feadme[gpu]    # GPU (CUDA 12)
```

From source with [uv](https://github.com/astral-sh/uv):

```bash
git clone https://github.com/nmearl/feadme.git
cd feadme
uv sync --group dev
```

From source without uv:

```bash
git clone https://github.com/nmearl/feadme.git
cd feadme
pip install -e ".[dev]"
```

## Quickstart

```bash
feadme run \
  --template-path template.json \
  --data-path data.csv \
  --output-path results \
  --sampler nuts \
  --init-method jax-lsq \
  --integrator mixed
```
