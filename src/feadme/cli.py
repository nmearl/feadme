import json
from pathlib import Path

import click
import jax
import numpy as np
from astropy.table import Table
from loguru import logger
from numpyro.infer import init_to_median

from .compose import disk_model
from .parser import Template
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
    "--override-data-dir",
    type=click.Path(),
    help="Overrides the data directory read from template file.",
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
    "--num-warmup",
    type=int,
    default=1000,
    help="Number of warmup steps for the MCMC sampler.",
)
@click.option(
    "--num-samples",
    type=int,
    default=2000,
    help="Number of samples to draw from the posterior.",
)
@click.option(
    "--num-chains",
    type=int,
    default=jax.local_device_count(),
    help="Number of chains to run in parallel.",
)
@click.option(
    "--no-progress-bar",
    is_flag=True,
    default=False,
    help="Display a progress bar during sampling.",
)
def run(
    template_file: str,
    data_file: str = None,
    override_data_dir: str = None,
    output_dir: str = None,
    label: str = None,
    num_warmup: int = 2000,
    num_samples: int = 2000,
    num_chains: int = jax.local_device_count(),
    no_progress_bar: bool = True,
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

        # if "ZTF" not in str(template_path):
        #     continue

        if "ZTF18aacrkse" not in str(template_path):
            continue

        # Load the template
        with open(template_path, "r") as f:
            loaded_data = json.load(f)
            template = Template(**loaded_data)

        local_label = label or template.name
        base_name = template_path.stem

        logger.info(f"Starting sampling for `{local_label}`.")

        if data_file is None:
            local_data_file = template.data_path
        else:
            local_data_file = data_file

        if override_data_dir is not None:
            local_data_file = Path(override_data_dir) / Path(local_data_file).name

        if not Path(local_data_file).exists():
            logger.warning(f"Data file {local_data_file} does not exist.")
            continue

        logger.info(f"Reading data file from template: `{local_data_file}`")

        data = Table.read(
            local_data_file, format="ascii.csv", names=("wave", "flux", "flux_err")
        )

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
            progress_bar=not no_progress_bar,
        )

        if not nuts_sampler.converged:
            nuts_sampler.sample(init_strategy=init_to_median(num_samples=1000))
            nuts_sampler.write_results()
            nuts_sampler.plot_results()
        else:
            logger.info(f"{local_label} is already converged. Skipping sampling.")

        jax.clear_caches()
