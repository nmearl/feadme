![](https://raw.githubusercontent.com/nmearl/feadme/refs/heads/main/images/feadme_logo_wide.png)

# Fast Elliptical Accretion Disk Modeling Engine

`feadme` fits elliptical accretion disk models to double-peaked emission line
profiles via Bayesian inference. It implements the disk model from
[Eracleous et al. (1995)](https://ui.adsabs.harvard.edu/abs/1995ApJ...438..610E/abstract)
and uses JAX and NumPyro for JIT-compiled, gradient-based NUTS sampling.

## Features

- JAX-accelerated disk integration with GPU support
- NUTS posterior sampling via NumPyro
- JSON-driven model templates with multi-component and shared-parameter support

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
