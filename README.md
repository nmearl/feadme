![](https://github.com/nmearl/feadme/blob/main/images/feadme_logo_wide.png)

A fast elliptical accretion disk modeling engine built on [JAX](https://github.com/google/jax) and [NumPyro](https://github.com/pyro-ppl/numpyro).

`feadme` implements the elliptical accretion disk model described in
[Eracleous et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995ApJ...438..610E/abstract)
and fits it to double-peaked emission line profiles via Bayesian inference.

## Features

- **Fast**: JAX-accelerated disk integration with JIT compilation and GPU support.
- **Bayesian**: NUTS-based posterior sampling via NumPyro with full uncertainty quantification.
- **Flexible**: JSON-driven model templates supporting multiple disk and line components with shared parameters.
- **Robust initialization**: SVI, Pathfinder, MAP, AutoDelta-style MAP, and JAX-LSQ initializers for difficult multimodal line-profile fits.

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
  --template-path my_template.json \
  --data-path my_data.csv \
  --output-path results \
  --init-method jax-lsq \
  --integrator mixed
```

See the [documentation](https://nmearl.github.io/feadme) for the full CLI reference, data format, and template specification.

## Contributing

Bug reports, feature requests, and pull requests are welcome on the [GitHub repository](https://github.com/nmearl/feadme).
