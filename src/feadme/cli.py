import json
import pickle
from functools import partial
from pathlib import Path

import click
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
from astropy.table import Table
from numpyro.infer import MCMC, NUTS, init_to_feasible, init_to_sample, init_to_uniform

from feadme.compose import disk_model
from feadme.parser import Template, Parameter
from .plotting import plot_corner, plot_fit

finfo = np.finfo(float)


def output_results(mcmc, label, output_dir):
    posterior_samples = mcmc.get_samples()
    param_dict = {}

    def summarize_posterior(samples):
        summary = {}
        for param, values in samples.items():
            median = jnp.median(values)
            lower_bound = jnp.percentile(values, 16)
            upper_bound = jnp.percentile(values, 86)
            summary[param] = {"median": median, "16%": lower_bound, "84%": upper_bound}
        return summary

    posterior_summary = summarize_posterior(posterior_samples)

    for k, v in posterior_summary.items():
        param_dict.setdefault("label", []).append(label)
        param_dict.setdefault("param", []).append(k)
        param_dict.setdefault("value", []).append(v["median"])
        param_dict.setdefault("err_lo", []).append(v["median"] - v["16%"])
        param_dict.setdefault("err_hi", []).append(v["84%"] - v["median"])

    Table(param_dict).write(
        f"{output_dir}/disk_param_results.csv", format="ascii.csv", overwrite=True
    )


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
    data = Table.read(
        data_file, format="ascii.basic", names=("wave", "flux", "flux_err")
    )

    # Load the template
    with open(template_file, "r") as f:
        loaded_data = json.load(f)
        template = Template(**loaded_data)

    label = label or template.name

    wave = data["wave"] / (1 + template.redshift)
    flux = data["flux"]
    flux_err = data["flux_err"]

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

    full_disk_model = partial(
        disk_model,
        template=template,
        # parameters={
        #     prof.name: {
        #         k: v for k, v in profile_ref[prof.name].items() if v["shared"] is None
        #     }
        #     for prof in template.disk_profiles + template.line_profiles
        # },
        # shared_parameters={
        #     prof.name: {
        #         k: v
        #         for k, v in profile_ref[prof.name].items()
        #         if v["shared"] is not None
        #     }
        #     for prof in template.disk_profiles + template.line_profiles
        # },
        masks=masks,
    )

    nuts_kernel = NUTS(
        full_disk_model,
        init_strategy=init_to_uniform(),
        # find_heuristic_step_size=True,
        # dense_mass=True,
        # max_tree_depth=(20, 10),
        # adapt_step_size=True,
        # target_accept_prob=0.9,
    )
    rng_key = jax.random.PRNGKey(0)

    if not Path(output_dir).exists():
        Path(output_dir).mkdir(parents=True)

    if Path(f"{output_dir}/{label}.pkl").exists():
        with open(f"{output_dir}/{label}.pkl", "rb") as f:
            mcmc = pickle.load(f)
    else:
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            chain_method=(
                "vectorized" if jax.local_device_count() == 1 else "parallel"
            ),
        )

        with numpyro.validation_enabled():
            mcmc.run(
                rng_key,
                wave=jnp.asarray(wave),
                flux=jnp.asarray(flux),
                flux_err=jnp.asarray(flux_err),
            )

        with open(f"{output_dir}/{label}.pkl", "wb") as f:
            pickle.dump(mcmc, f)

    mcmc.print_summary()

    output_results(mcmc, label, output_dir)
    plot_fit(mcmc, full_disk_model, wave, flux, flux_err, output_dir, rng_key)
    plot_corner(mcmc, output_dir)
