# Usage

## CLI Reference

`feadme` is invoked via the `run` subcommand:

```bash
feadme run --template-path TEMPLATE --data-path DATA [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--template-path PATH` | *(required)* | Path to the JSON template file |
| `--data-path PATH` | *(required)* | Path to the CSV data file |
| `--output-path PATH` | `./output` | Directory to save results and plots |
| `--sampler [nuts\|neutra]` | `nuts` | Sampler to use |
| `--num-warmup INTEGER` | | Number of warmup steps |
| `--num-samples INTEGER` | | Number of posterior samples |
| `--num-chains INTEGER` | | Number of independent MCMC chains |
| `--target-accept-prob FLOAT` | | Target acceptance probability |
| `--max-tree-depth INTEGER` | | Maximum NUTS leapfrog tree depth |
| `--dense-mass / --sparse-mass` | | Mass matrix type for adaptation |
| `--integrator [mixed\|quad\|split_quad\|trap]` | auto | Disk integrator family |
| `--lsq-init-candidates INTEGER` | `1` | Number of LSQ starting basins before sampling |
| `--rebin FLOAT` | | Rebin spectrum to this velocity resolution (km/s) |
| `--skip-existing` | | Skip run if results already exist at output path |
| `--compute-prior-predictive` | | Compute prior predictive samples for diagnostics |
| `--progress-bar / --no-progress-bar` | | Toggle sampling progress bar |

### Example

```bash
feadme run \
    --template-path my_template.json \
    --data-path my_spectrum.csv \
    --output-path results/ \
    --sampler nuts \
    --num-warmup 2000 \
    --num-samples 2000 \
    --num-chains 2 \
    --integrator mixed
```

---

## Data Format

Input data must be a CSV file with three columns (column names are ignored):

| Column | Units |
|---|---|
| Wavelength | Ångströms |
| Flux | mJy |
| Flux uncertainty | mJy |

---

## Template Format

The model is specified as a JSON file describing all components and their
parameter priors.

### Top-Level Fields

| Field | Required | Description |
|---|---|---|
| `name` | Yes | Identifier for the target or run |
| `redshift` | Yes | Redshift parameter specification |
| `disk_profiles` | Yes | List of disk emission components |
| `line_profiles` | Yes | List of narrow/broad line components |
| `mask` | Yes | Wavelength intervals to include in the fit |
| `obs_date` | No | Observation date (float) |
| `log_frac_noise` | No | Fractional noise floor parameter |

### Parameter Specification

Every free parameter in the template uses the same structure:

```json
{
  "distribution": "uniform",
  "value": null,
  "fixed": false,
  "shared": null,
  "low": 0.0,
  "high": 1.0,
  "loc": 0.5,
  "scale": 0.25,
  "circular": false
}
```

| Field | Description |
|---|---|
| `distribution` | `"uniform"`, `"log_uniform"`, `"normal"`, or `"log_normal"` |
| `value` | Initial/fixed value; `null` to sample |
| `fixed` | `true` to lock the parameter at `value` |
| `shared` | Name of another **profile** whose same-named parameter this links to |
| `low` / `high` | Hard prior bounds |
| `loc` / `scale` | Center and width for normal-family distributions |
| `circular` | `true` for angular parameters (e.g. `apocenter`) |

### Disk Profile Parameters

Each entry in `disk_profiles` must have a `name` (string) and a `center`
(float, in observed-frame Ångströms), plus parameter specifications for:

| Parameter | Units | Description |
|---|---|---|
| `inner_radius` | R_g | Inner disk radius |
| `outer_radius` | R_g | Outer disk radius |
| `inclination` | radians | Disk inclination (0 = face-on, π/2 = edge-on) |
| `sigma` | km/s | Per-annulus velocity broadening |
| `q` | — | Emissivity power-law index |
| `eccentricity` | — | Orbital eccentricity (0–1) |
| `apocenter` | radians | Apocenter orientation angle (circular parameter) |
| `area` | — | Integrated disk flux |
| `offset` | km/s | Systemic velocity offset |
| `baseline` | — | Additive continuum baseline |

### Line Profile Parameters

Each entry in `line_profiles` must have a `name`, a `center` (float), and
optionally a `shape` (`"gaussian"` or `"lorentzian"`, default `"gaussian"`),
plus parameter specifications for:

| Parameter | Units | Description |
|---|---|---|
| `area` | — | Integrated line flux |
| `vel_width` | km/s | Line velocity dispersion (σ) |
| `offset` | km/s | Velocity offset from line center |

### Mask

The `mask` field is a list of wavelength windows to include in the fit.
Pixels outside all windows are excluded.

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
      "center": 6562.8,
      "inner_radius": { "distribution": "log_uniform", "low": 100, "high": 5000, "fixed": false },
      "outer_radius": { "distribution": "log_uniform", "low": 500, "high": 50000, "fixed": false },
      "inclination": { "distribution": "uniform", "low": 0.0, "high": 1.57, "fixed": false },
      "sigma":       { "distribution": "log_uniform", "low": 200, "high": 3000, "fixed": false },
      "q":           { "distribution": "normal", "low": 1.0, "high": 4.0, "loc": 2.5, "scale": 1.0, "fixed": false },
      "eccentricity":{ "distribution": "normal", "low": 0.0, "high": 1.0, "loc": 0.35, "scale": 0.35, "fixed": false },
      "apocenter":   { "distribution": "uniform", "low": 0.0, "high": 6.283, "circular": true, "fixed": false },
      "area":        { "distribution": "uniform", "low": 0.0, "high": 50.0, "fixed": false },
      "offset":      { "distribution": "normal",  "low": -5000, "high": 5000, "loc": 0.0, "scale": 500, "fixed": true },
      "baseline":    { "distribution": "uniform", "low": 0.0, "high": 0.05, "fixed": true }
    }
  ],
  "line_profiles": [
    {
      "name": "halpha_narrow",
      "center": 6562.8,
      "shape": "gaussian",
      "area":      { "distribution": "uniform", "low": 0.0, "high": 30.0, "fixed": false },
      "vel_width": { "distribution": "normal",  "low": 70,  "high": 500,  "loc": 150, "scale": 25, "fixed": false },
      "offset":    { "distribution": "normal",  "low": -500, "high": 500, "loc": 0.0, "scale": 100, "fixed": true }
    }
  ],
  "mask": [
    { "lower_limit": 6400.0, "upper_limit": 6800.0 }
  ]
}
```

---

## Parallelization

JAX parallelizes at the device level. On a single CPU, chains are vectorized
rather than run in parallel. To simulate multiple CPU devices:

```bash
export FORCE_DEVICE_COUNT=4
```

This allows multiple MCMC chains to run in parallel on a single CPU machine.
On GPU, chains run in parallel across available GPU resources automatically.
