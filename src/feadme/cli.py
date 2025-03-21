import json
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import numpy as np
from astropy.table import Table

from .compose import disk_model
from .parser import Template, Parameter
from .samplers import (
    nuts_with_adaptation,
    nuts_with_adaptation_multi,
    # initialize_to_nuts,
)
from .samplers import NUTSSampler

finfo = np.finfo(float)


@click.command()
@click.argument(
    "data-file",
    type=click.Path(exists=True),
    required=True,
    # description="Path to the data file. Data files should have three columns: "
    # "wavelengths (in Angstrom), fluxes, and flux uncertainties.",
)
@click.argument(
    "template-file",
    type=click.Path(exists=True),
    required=True,
    # description="Path to the template file.",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="output",
    help="Directory to which the output files and plots will be saved. Defaults to current directory.",
)
@click.option(
    "--label",
    type=str,
    help="Optional label for the object. Overrides the name given in the template file.",
)
@click.option(
    "--num_warmup",
    type=int,
    default=1000,
    help="Number of warmup steps for the MCMC sampler.",
)
@click.option(
    "--num_samples",
    type=int,
    default=2000,
    help="Number of samples to draw from the posterior.",
)
@click.option(
    "--num_chains",
    type=int,
    default=jax.local_device_count(),
    help="Number of chains to run in parallel.",
)
def run(
    data_file: str,
    template_file: str,
    output_dir: str = None,
    label: str = None,
    num_warmup: int = 2000,
    num_samples: int = 2000,
    num_chains: int = jax.local_device_count(),
):
    # Load the data
    data = Table.read(data_file, format="ascii.csv", names=("wave", "flux", "flux_err"))

    # Load the template
    with open(template_file, "r") as f:
        loaded_data = json.load(f)
        template = Template(**loaded_data)

    label = label or template.name

    wave = (data["wave"] / (1 + template.redshift)).value
    flux = data["flux"].value
    flux_err = data["flux_err"].value

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    nuts_sampler = NUTSSampler(
        disk_model,
        template,
        wave,
        flux,
        flux_err,
        output_dir,
        label,
        num_warmup,
        num_samples,
        num_chains,
    )

    nuts_sampler.sample()

    print(nuts_sampler._mcmc.get_samples().keys())
    print(nuts_sampler.posterior_samples.keys())
    print(nuts_sampler.posterior_samples_transformed.keys())

    nuts_sampler.write_results()
    nuts_sampler.plot_results()
