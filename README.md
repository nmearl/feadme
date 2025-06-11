# FEADME

A fast elliptical accretion disk modeling engine written with NumPyro and Jax.

`feadme` is a highly efficient and flexible framework for modeling elliptical 
accretion disks. It leverages the power of Jax and NumPyro to provide fast 
computations and easy-to-use interfaces for astrophysical modeling of 
double-peaked emission line features in spectra.

## Features

- **Fast Computation**: Utilizes Jax for automatic differentiation and GPU acceleration.
- **Flexible Modeling**: Leverages the No-U-Turn gradient-based sampler to 
  provide optimized convergence and exploration of parameter space.
- **Probabilistic Programming**: Built on NumPyro for Bayesian inference and probabilistic modeling.
- **User-Friendly**: Easy-to-use API for defining model templates.
- **Extensible**: Designed to be easily extended for custom sampler 
  implementations.

## Installation

The installation of `feadme` is platform-dependent as the installation requires
libraries specific to the platform resources. Refer to the table below for
installation instructions based on available resources of your platform.

| Platform | Installation Command |
|----------|----------------------|
| CPU      | `pip install feadme` |
| GPU      | `pip install feadme[gpu]` |
| TPU      | `pip install feadme[tpu]` |

## Usage

`feadme` provides a simple interface for generating disk fits and follows the
structure below:

```bash
Usage: feadme [OPTIONS] TEMPLATE_PATH [DATA_PATH]

Options:
  --output-path PATH     Directory to save output files and plots. Defaults to
                         './output'.
  --sampler-type [nuts]  Type of NumPyro sampler to use.
  --num-warmup INTEGER   Number of warmup steps for the MCMC sampler.
  --num-samples INTEGER  Number of samples to draw from the posterior
                         distribution.
  --num-chains INTEGER   Number of MCMC chains to run.
  --progress-bar         Display a progress bar during sampling.
  --pre-fit              Run a pre-fit using the least-squares model fitter
                         before sampling.
  --help                 Show this message and exit.
```

As an example, to run a model fit using the `nuts` (currently the only 
available) sampler, you can use the following command:

```bash
feadme my_template.json my_data.csv --sampler-type nuts --num-warmup 1000 
--num-samples 5000 --num-chains 2
```

## Data Format

The data parser is currently designed to handle CSV files with three defined 
column. The column names do not matter.

- **First column**: Wavelength values in `Angstrom`s.
- **Second column**: Flux values in `mJy`.
- **Third column**: Uncertainties in `mJy`.

## Template Format

To use `feadme`, you must provide a JSON configuration file that defines the model components and their parameter priors. This file serves as the input "template" describing the physical and statistical structure of your emission line model.

### Top-Level Fields

The JSON file must include the following keys:

* **`name`**: A descriptive name for the model or target.
* **`redshift`**: The redshift of the source.
* **`disk_profiles`**: A list of disk-like emission components.
* **`line_profiles`**: A list of Gaussian line components.
* **`mask`**: A list of wavelength intervals to exclude from the fit.

### Defining Disk and Line Profiles

Each entry in `disk_profiles` or `line_profiles` represents a parameterized 
model component, and the definitions of these parameters are used to construct 
the prior distributions for the sampling. All model parameters (e.g. `center`, 
`sigma`, `inclination`, `amplitude`, etc.) must follow this structure:

```json
{
  "name": <string>, // unique name for the parameter
  "distribution": <string>, // distribution type: "uniform", "log_uniform", "normal", or "log_normal"
  "value": <null | float>, // null for sampling, float for fixed value
  "fixed": <true | false>, // true if the parameter is fixed at "value"
  "shared": <null | string>, // name of another **profile** that contains the same **parameter name** to link to
  "low": <float>, // lower bound
  "high": <float>, // upper bound
  "loc": <float>, // center of the prior (used in normal distributions)
  "scale": <float>,  // width of the prior (used in normal distributions)
  "circular": <true | false> // true if the parameter is circular (e.g., angles)
}
```

* Set `"fixed": true` to lock the parameter at `"value"`.
* Use `"shared"` to link this parameter to one from another component.
* Set `"circular": true` for angular variables (e.g., azimuthal angles).

Each `disk_profile` must define these physical parameters:

* `center`: the central wavelength of the emission line (in `Angstrom`s), 
* `inner_radius`: the inner radius of the disk (in gravitational radii units `R_g`),
* `delta_radius`: the radial width of the disk (in `R_g`),
* `inclination`: the inclination angle of the disk (in radians),
* `sigma`: the broadening component of the disk (in `km/s`),
* `q`: the emissivity profile power-law index,
* `eccentricity`: the eccentricity of the disk,
* `apocenter`: the apocenter of the disk (in radians),
* `scale`: the scale factor for the disk profile (unitless),
* `offset`: the offset of the disk profile (in `mJy`).

Each `line_profile` must define parameters for:

* `center`: the central wavelength of the emission line (in `Angstrom`s),
* `amplitude`: the amplitude of the line profile (in `mJy`),
* `vel_width`: the velocity width of the line profile (in `km/s`),
* [Optionally] `shape`: the shape of the line profile, ("gaussian" [default], "lorentzian")

### Notes

* Parameter names must be unique within each component.
* Units must be consistent (e.g., wavelengths in Angstroms, velocities in km/s).
* You may define relationships between parameters using `shared` references, where supported.
* The `shared` field requires the name of another **profile** that contains the same **parameter name** to link to.

### Example

```json
{
  "name": "ZTF18aahiqst",
  "redshift": 0.074675,
  "disk_profiles": [
    {
      "name": "halpha_disk",
      "center": {
        "name": "center", 
        "distribution": "uniform", 
        "low": 6557.8, 
        "high": 6567.8, 
        "loc": 6562.8, 
        "scale": 5.0, 
        "fixed": false, 
        "value": null, 
        "circular": false
      }
    }
    // additional disk profiles would be defined here
  ],
  "line_profiles": [
    {
      "name": "halpha_narrow",
      "shape": "gaussian",
      "center": { 
        "name": "center", 
        "distribution": "uniform", 
        "low": 6560.8, 
        "high": 6564.8, 
        "loc": 6562.8, 
        "scale": 2.0, 
        "shared": "halpha_disk", // will use the `center` parameter from the `halpha_disk` profile
        "fixed": false, 
        "value": null, 
        "circular": false
      }
    }
    // additional line profiles would be defined here
  ],
  "mask": [
    { 
      "lower_limit": 6400.0, 
      "upper_limit": 6800.0
    }
    // additional mask intervals can be defined here
  ]
}
```

## Contributing

We welcome contributions to `feadme`. If you find a bug, have a feature 
request, or want to contribute code, please open an issue or pull request on 
the GitHub repository.

