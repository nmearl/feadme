import json
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import numpy as np
from astropy.table import Table

from feadme.compose import disk_model
from feadme.parser import Template, Parameter
from .samplers import (
    nuts_with_adaptation,
    nuts_with_adaptation_multi,
    initialize_to_nuts,
)

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

    mask = (wave > 6350) & (wave < 6800)
    wave = wave[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]

    profile_ref = {}

    for prof in template.disk_profiles + template.line_profiles:
        profile_ref.setdefault(prof.name, {})

        for field in prof.model_fields:
            field_ref = getattr(prof, field)

            if isinstance(field_ref, Parameter):
                profile_ref[prof.name][field] = {}
                profile_ref[prof.name][field].setdefault(
                    "distribution", field_ref.distribution.name
                )
                profile_ref[prof.name][field].setdefault("fixed", field_ref.fixed)
                profile_ref[prof.name][field].setdefault("shared", field_ref.shared)
                profile_ref[prof.name][field].setdefault("low", field_ref.low)
                profile_ref[prof.name][field].setdefault("high", field_ref.high)
                profile_ref[prof.name][field].setdefault("loc", field_ref.loc)
                profile_ref[prof.name][field].setdefault("scale", field_ref.scale)

    # Construct masks
    masks = {}

    for prof in template.disk_profiles:
        mask = [
            np.bitwise_and(wave > m.lower_limit, wave < m.upper_limit)
            for m in prof.mask
        ]

        if len(mask) > 1:
            mask = np.bitwise_or(*mask)
        else:
            mask = mask[0]

        masks[prof.name] = {"mask": jnp.asarray(mask), "wave": jnp.asarray(wave[mask])}

    # full_disk_model = partial(
    #     disk_model,
    #     template=template,
    #     masks=masks,
    # )

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    initialize_to_nuts(
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
