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
    "template-file",
    type=click.Path(exists=True),
    required=True,
    # help="Path to the template file, or a directory containing template "
    #      "files.",
)
@click.argument(
    "data-file",
    type=click.Path(exists=True),
    required=False,
    # help="Overrides the data file given in the template. Path to the data "
    #      "file. Data files should have three columns: wavelengths "
    #      "(in Angstrom), fluxes, and flux uncertainties in (in mJy).",
)
@click.option(
    "--output-dir",
    type=click.Path(),
    default="output",
    help="Directory to which the output files and plots will be saved. "
         "Defaults to current directory.",
)
@click.option(
    "--label",
    type=str,
    help="Optional label for the object. Overrides the name given in the "
         "template file.",
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
    template_file: str,
    data_file: str = None,
    output_dir: str = None,
    label: str = None,
    num_warmup: int = 2000,
    num_samples: int = 2000,
    num_chains: int = jax.local_device_count(),
):
    template_file = Path(template_file)

    if not template_file.is_dir():
        template_files = [template_file]
    else:
        template_files = sorted(template_file.glob("*.json"))

    for template_path in template_files:
        if "14li" in str(template_path):
            # Skip the 14li template for now
            continue

        # Load the template
        with open(template_path, "r") as f:
            loaded_data = json.load(f)
            template = Template(**loaded_data)

        if data_file is None:
            print(f"Using data file from template: {template.data_path}")
            local_data_file = template.data_path
        else:
            local_data_file = data_file

        if not Path(local_data_file).exists():
            print(f"Warning: Data file {local_data_file} does not exist.")
            continue

        data = Table.read(local_data_file, format="ascii.csv",
                          names=("wave", "flux", "flux_err"))

        local_label = label or template.name
        base_name = template_path.stem

        print(f"Fitting model for {local_label}")

        local_output_dir = Path(output_dir) / base_name

        if not local_output_dir.exists():
            local_output_dir.mkdir(parents=True)

        wave = (data["wave"] / (1 + template.redshift)).value
        flux = data["flux"].value
        flux_err = data["flux_err"].value

        if not Path(local_output_dir).exists():
            Path(local_output_dir).mkdir(parents=True)

        nuts_sampler = NUTSSampler(
            disk_model,
            template,
            wave,
            flux,
            flux_err,
            local_output_dir,
            local_label,
            num_warmup,
            num_samples,
            num_chains,
        )

        if not nuts_sampler.converged:
            nuts_sampler.sample()
            nuts_sampler.write_results()
            nuts_sampler.plot_results()
        else:
            print(f"{label} is already converged. Skipping sampling.")
