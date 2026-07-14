# Usage

## CLI Reference

`feadme` is invoked with the `run` subcommand:

```bash
feadme run --template-path TEMPLATE --data-path DATA [OPTIONS]
```

### Core Options

| Option | Default | Description |
|---|---:|---|
| `--template-path PATH` | required | JSON template file |
| `--data-path PATH` | required | CSV spectrum file |
| `--output-path PATH` | `output` | Directory for `results.nc`, `summary.csv`, and plots |
| `--skip-existing / --no-skip-existing` | `False` | Skip sampling when output already exists |
| `--compute-prior-predictive / --no-compute-prior-predictive` | `False` | Save prior predictive samples for diagnostics |
| `--progress-bar / --no-progress-bar` | `True` | Toggle sampler progress bars |
| `--rebin FLOAT` | `None` | Rebin the loaded spectrum to a velocity resolution in km/s before fitting |
| `--debug-plot / --no-debug-plot` | `False` | Save initializer diagnostic plots |

### Sampler Options

| Option | Default | Description |
|---|---:|---|
| `--sampler [nuts\|neutra]` | `nuts` | Sampler backend |
| `--num-warmup INTEGER` | `1000` | NUTS warmup steps |
| `--num-samples INTEGER` | `1000` | Posterior samples per chain |
| `--num-chains INTEGER` | `1` | Number of MCMC chains |
| `--target-accept-prob FLOAT` | `0.8` | NUTS target acceptance probability |
| `--max-tree-depth INTEGER` | `10` | Maximum NUTS tree depth |
| `--dense-mass / --sparse-mass` | `False` | Use dense or diagonal mass-matrix adaptation |
| `--integrator [mixed\|quad\|split_quad\|trap]` | auto | Disk integrator. Defaults to `quad` on CPU and `mixed` on GPU |

`nuts` is the production sampler. `neutra` is available for experiments with
NeuTra reparameterization, but is not the default production path.

### Initialization Options

The initializer chooses the starting basin passed to NUTS. The default is `svi`,
but for difficult AGN/TDE line-profile fits `jax-lsq` is usually the most
practical general-purpose option.

| Option | Default | Description |
|---|---:|---|
| `--init-method [svi\|pathfinder\|map\|jax-lsq\|delta-map]` | `svi` | Basin-refinement method before NUTS |
| `--init-candidate-distance-threshold FLOAT` | `0.25` | Distance below which initialization candidates are treated as duplicate basins |
| `--lsq-init-candidates INTEGER` | `1` | Structured Astropy-LSQ starts used by SVI/Pathfinder when requested |
| `--lsq-init-maxiter INTEGER` | `2000` | Maximum optimizer iterations for each Astropy-LSQ start |
| `--svi-init-candidates INTEGER` | `1` | Distinct LSQ basins refined independently with SVI |
| `--svi-init-steps INTEGER` | `2000` | SVI optimization steps per candidate |
| `--svi-init-samples INTEGER` | `1000` | Guide samples used to score SVI candidates |
| `--svi-init-max-loss-relative-std FLOAT` | `0.10` | Recent-loss stability threshold for SVI candidate selection |
| `--pathfinder-init-candidates INTEGER` | `8` | Distinct start basins refined with BlackJAX Pathfinder |
| `--pathfinder-start-method [lsq\|structured]` | `lsq` | Use Astropy-LSQ starts or direct structured template starts |
| `--pathfinder-init-samples INTEGER` | `32` | Approximate posterior samples drawn per Pathfinder path |
| `--pathfinder-score-batch-size INTEGER` | `8` | Pathfinder sample scoring batch size |
| `--pathfinder-init-maxiter INTEGER` | `30` | L-BFGS iterations per Pathfinder path |
| `--map-init-candidates INTEGER` | `64` | Independent starts for batched MAP initialization |
| `--map-start-method [prior\|structured]` | `prior` | MAP start generation method |
| `--map-selection-score [likelihood\|posterior]` | `likelihood` | Final MAP candidate ranking score |
| `--map-init-steps INTEGER` | `200` | L-BFGS iterations per MAP start |
| `--delta-map-init-candidates INTEGER` | `8` | Independent AutoDelta/BFGS starts |
| `--delta-map-start-method [prior\|structured]` | `structured` | AutoDelta/BFGS start generation method |
| `--delta-map-init-maxiter INTEGER` | `300` | BFGS iterations per AutoDelta/BFGS start |
| `--delta-map-selection-score [likelihood\|posterior\|penalized-posterior]` | `penalized-posterior` | Final AutoDelta/BFGS candidate ranking score |
| `--jax-lsq-init-candidates INTEGER` | `64` | Independent starts for JAX-LSQ initialization |
| `--jax-lsq-start-method [prior\|structured]` | `structured` | JAX-LSQ start generation method |
| `--jax-lsq-init-steps INTEGER` | `500` | Levenberg-Marquardt iterations per JAX-LSQ start |
| `--jax-lsq-batch-size INTEGER` | `4` | Number of starts optimized in one vmapped JAX-LSQ batch |
| `--jax-lsq-selection-score [likelihood\|posterior\|penalized-posterior]` | `penalized-posterior` | Final JAX-LSQ candidate ranking score |

### Example

```bash
feadme run \
  --template-path template.json \
  --data-path data.csv \
  --output-path results \
  --sampler nuts \
  --num-warmup 2000 \
  --num-samples 2000 \
  --num-chains 2 \
  --target-accept-prob 0.9 \
  --max-tree-depth 10 \
  --dense-mass \
  --integrator mixed \
  --init-method jax-lsq \
  --jax-lsq-init-candidates 64 \
  --jax-lsq-init-steps 500 \
  --jax-lsq-batch-size 4 \
  --jax-lsq-selection-score penalized-posterior
```

---

## Data Format

Input data must be a CSV file with three columns. Column names are ignored.

| Column | Units |
|---|---|
| Wavelength | Angstroms |
| Flux | User-defined flux density or continuum-subtracted flux unit |
| Flux uncertainty | Same flux unit as `Flux` |

The template mask is applied after loading the data. If `--rebin` is supplied,
the spectrum is rebinned before fitting.

---

## Template Format

The model is specified as a JSON file describing all components and priors.

### Top-Level Fields

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Target or run identifier |
| `redshift` | Yes | Redshift parameter specification |
| `disk_profiles` | Yes | List of elliptical disk emission components |
| `line_profiles` | Yes | List of Gaussian or Lorentzian line components |
| `mask` | Yes | Wavelength intervals to include in the fit |
| `obs_date` | No | Observation date |
| `log_frac_noise` | No | Log fractional model jitter. Total variance is `flux_err^2 + total_flux^2 * exp(2 * log_frac_noise)` |

### Parameter Specification

Free and fixed parameters share the same schema:

```json
{
  "distribution": "normal",
  "value": 0.0,
  "fixed": false,
  "shared": null,
  "low": -1.0,
  "high": 1.0,
  "loc": 0.0,
  "scale": 0.25,
  "circular": false
}
```

| Field | Description |
|---|---|
| `distribution` | Prior family. Current templates use `uniform`, `log_uniform`, `normal`, `log_normal`, and `beta` where supported |
| `value` | Fixed value or initializer/reference value |
| `fixed` | `true` locks the parameter at `value` |
| `shared` | Name of another profile whose same-named parameter this profile shares |
| `low` / `high` | Hard parameter bounds |
| `loc` / `scale` | Center and width for normal/log-normal priors |
| `alpha` / `beta` | Shape parameters for `beta` priors |
| `circular` | `true` for angular parameters such as `apocenter` |

### Disk Profile Parameters

Each disk profile has a `name`, a fixed rest-frame `center`, and parameter
specifications for the disk model. `center` is metadata, not a sampled
parameter. Velocity shifts are represented by `offset`.

| Parameter | Units | Description |
|---|---|---|
| `inner_radius` | `R_g` | Inner disk radius |
| `radius_ratio` | - | Radial extent `outer_radius / inner_radius` |
| `inclination` | radians | Disk inclination, from face-on (`0`) to edge-on (`pi/2`) |
| `sigma` | km/s | Local velocity broadening |
| `q` | - | Emissivity power-law index |
| `eccentricity` | - | Orbital eccentricity |
| `apocenter` | radians | Apocenter orientation angle |
| `area` | flux x wavelength | Integrated disk flux |
| `offset` | km/s | Velocity offset from `center` |
| `baseline` | flux | Additive continuum baseline |

`outer_radius` is not specified in templates. It is derived deterministically as

```text
outer_radius = inner_radius * radius_ratio
```

and is included in summaries and posterior outputs for convenience.

### Line Profile Parameters

Each line profile has a `name`, a fixed rest-frame `center`, and a `shape`
(`gaussian` or `lorentzian`; default is `gaussian`).

| Parameter | Units | Description |
|---|---|---|
| `area` | flux x wavelength | Integrated line flux |
| `vel_width` | km/s | Velocity width parameter |
| `offset` | km/s | Velocity offset from `center` |

### Mask

The `mask` field is a list of wavelength windows to include in the fit. Pixels
outside all windows are excluded.

```json
"mask": [
  { "lower_limit": 6400.0, "upper_limit": 6800.0 }
]
```

### Minimal Example

```json
{
  "name": "my_target",
  "redshift": {
    "distribution": "normal",
    "value": 0.074,
    "fixed": false,
    "low": 0.064,
    "high": 0.084,
    "loc": 0.074,
    "scale": 0.005,
    "circular": false
  },
  "disk_profiles": [
    {
      "name": "halpha_disk",
      "center": 6562.819,
      "inner_radius": { "distribution": "log_uniform", "low": 100.0, "high": 10000.0, "value": 1000.0, "fixed": false },
      "radius_ratio": { "distribution": "log_normal", "low": 1.2, "high": 22.0, "loc": 10.0, "scale": 6.0, "value": 10.0, "fixed": false },
      "inclination": { "distribution": "uniform", "low": 0.0, "high": 1.5708, "value": 0.7854, "fixed": false },
      "sigma": { "distribution": "log_normal", "low": 200.0, "high": 3000.0, "loc": 600.0, "scale": 320.0, "value": 600.0, "fixed": false },
      "q": { "distribution": "normal", "low": 1.0, "high": 4.0, "loc": 2.5, "scale": 1.0, "value": 2.5, "fixed": false },
      "eccentricity": { "distribution": "normal", "low": 0.0, "high": 0.95, "loc": 0.2, "scale": 0.15, "value": 0.2, "fixed": false },
      "apocenter": { "distribution": "uniform", "low": 0.0, "high": 6.2832, "value": 3.1416, "circular": true, "fixed": false },
      "area": { "distribution": "log_normal", "low": 0.05, "high": 500.0, "loc": 10.0, "scale": 5.33, "value": 10.0, "fixed": false },
      "offset": { "distribution": "normal", "low": -10000.0, "high": 10000.0, "loc": 0.0, "scale": 600.0, "value": 0.0, "fixed": true },
      "baseline": { "distribution": "uniform", "low": 0.0, "high": 0.05, "value": 0.0, "fixed": true }
    }
  ],
  "line_profiles": [
    {
      "name": "halpha_narrow",
      "center": 6562.819,
      "shape": "gaussian",
      "area": { "distribution": "log_normal", "low": 0.01, "high": 100.0, "loc": 3.0, "scale": 1.60, "value": 3.0, "fixed": false },
      "vel_width": { "distribution": "uniform", "low": 70.0, "high": 500.0, "value": 200.0, "fixed": false },
      "offset": { "distribution": "normal", "low": -10000.0, "high": 10000.0, "value": 0.0, "fixed": true }
    }
  ],
  "log_frac_noise": {
    "distribution": "uniform",
    "fixed": false,
    "low": -10.0,
    "high": 1.0,
    "value": 0.0
  },
  "mask": [
    { "lower_limit": 6400.0, "upper_limit": 6800.0 }
  ]
}
```

For a fuller starting point, see `src/feadme/templates/default.json`.

---

## Outputs

Each run writes, at minimum:

| File | Description |
|---|---|
| `results.nc` | ArviZ `InferenceData` with posterior samples and diagnostics |
| `summary.csv` | Parameter summaries, uncertainties, ESS, and R-hat values |
| `model_fit.png` | Median model reconstruction and component decomposition |
| `corner.png` | Posterior corner plot when enough dynamic range is available |
| `initialization_candidates.csv` | Candidate table for multi-candidate initializers when applicable |

---

## Parallelization

JAX parallelizes at the device level. On a single CPU, chains are vectorized
rather than run in parallel unless multiple CPU devices are exposed. To simulate
multiple CPU devices:

```bash
export FORCE_DEVICE_COUNT=4
```

On GPU, chains run across available accelerator resources automatically.
