from pathlib import Path

import loguru
from astropy.table import Table
from astropy.time import Time
import arviz as az
import click
from functools import wraps
import numpy as np
import astropy.constants as const

from .core.parser import Template, Config, Data, Mask
from .sampling.initializers import (
    DefaultInitializer,
    SVIInitializer,
    LSQInitializer,
)
from .core.integrators import (
    trap_jax_integrate,
    quad_jax_integrate,
    split_quad_jax_integrate,
    mixed_jax_integrate,
)
from .plotter import Plotter
from .reporter import Reporter
from .sampling.lsq.model import LSQModel
from .sampling.lsq.sampler import LSQSampler
from .sampling.numpyro.model import NumpyroModel
from .sampling.numpyro.nuts.sampler import NUTSSampler
from .sampling.numpyro.neutra.sampler import NeuTraSampler
from .utils import rebin_spectrum_logdv

logger = loguru.logger.opt(colors=True)

C_KMS = const.c.to("km/s").value


def load_data(data_path: str, template: Template, rebin: float | None = None) -> Data:
    """
    Load data from a CSV file and adjust the wavelength based on the
    template's redshift.

    Parameters
    ----------
    data_path : str
        Path to the CSV file containing the data.
    template : Template
        Template object containing the redshift and mask information.

    Returns
    -------
    Data
        A Data object containing the wavelength, flux, flux error, and mask.
    """
    data_tab = Table.read(
        data_path, format="ascii.csv", names=["wave", "flux", "flux_err"]
    )

    wave, flux, flux_err = (
        data_tab["wave"].value,
        data_tab["flux"].value,
        data_tab["flux_err"].value,
    )

    if rebin is not None:
        wave, flux, flux_err, info = rebin_spectrum_logdv(
            wave,
            flux,
            flux_err,
            dv=rebin,
            R=2000.0,
        )

    return Data.create(
        wave=wave,
        flux=flux,
        flux_err=flux_err,
        mask=template.mask,
    )


def perform_sampling(config, model, sampler):
    output_path = Path(config.output_path)

    logger.info(f"Starting sampling for <cyan>{config.template.name}</cyan>")

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: <light-red>{output_path}</light-red>")

    # If a results file already exists, load it instead of running the sampler
    results_exist = (Path(output_path) / "results.nc").exists()

    if results_exist:
        logger.info(
            f"Loading existing results at "
            f"<light-red>{output_path}/results.nc</light-red>."
        )
        idata = az.from_netcdf(
            f"{output_path}/results.nc",
        )
        logger.info(f"Results loaded for <cyan>{config.template.name}</cyan>.")
    else:
        start_time = Time.now()
        idata = sampler(config, model)
        delta_time = (Time.now() - start_time).to_datetime()

        logger.info(
            f"Finished processing <cyan>{config.template.name}</cyan> in "
            f"<green>{delta_time}</green>."
        )

    # Report results and write to disk
    reporter = Reporter(config=config, idata=idata)

    if not results_exist:
        reporter.write_results()
        logger.info(
            f"Results written to <green>{config.output_path}/results.nc</green>."
        )

    logger.info("Displaying sampler results:\n" + reporter.summary.to_markdown())

    # Plot results and save out figures
    plotter = Plotter(config=config, idata=idata, summary=reporter.summary)

    plotter.plot_model_fit()
    plotter.plot_corner()
    plotter.plot_prior_corner()
    plotter.plot_trace()

    logger.info(f"Plots saved to <green>{config.output_path}</green>.")


# Shared options decorator
def common_options(f):
    """Decorator for common CLI options shared across samplers."""

    @click.option(
        "--template-path",
        type=click.Path(exists=True),
        required=True,
        help="Path to the template file.",
    )
    @click.option(
        "--data-path",
        type=click.Path(exists=True),
        required=True,
        help="Path to the data file.",
    )
    @click.option(
        "--output-path",
        type=click.Path(),
        default="output",
        help="Directory to save output files and plots. Defaults to './output'.",
    )
    @click.option(
        "--skip-existing/--no-skip-existing",
        is_flag=True,
        default=False,
        help="Skip sampling if results already exist at the output path.",
    )
    @click.option(
        "--compute-prior-predictive/--no-compute-prior-predictive",
        is_flag=True,
        default=False,
        help="Whether to compute prior predictive samples for diagnostics.",
    )
    @click.option(
        "--progress-bar/--no-progress-bar",
        is_flag=True,
        default=True,
        help="Display a progress bar during sampling.",
    )
    @click.option(
        "--rebin",
        type=float,
        default=None,
        help="Rebin the spectrum to a specified velocity resolution (in km/s).",
    )
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@click.group()
def cli():
    """FEADME disk modeling CLI."""
    pass


@cli.command("nuts")
@common_options
@click.option(
    "--num-warmup",
    type=int,
    default=1000,
    help="Number of warmup steps for the MCMC sampler.",
)
@click.option(
    "--num-samples",
    type=int,
    default=1000,
    help="Number of samples for the MCMC sampler.",
)
@click.option(
    "--num-chains",
    type=int,
    default=1,
    help="Number of MCMC chains to run.",
)
@click.option(
    "--target-accept-prob",
    type=float,
    default=0.8,
    help="Target acceptance probability for the NUTS sampler.",
)
@click.option(
    "--max-tree-depth",
    type=int,
    default=10,
    help="Maximum tree depth for the NUTS sampler.",
)
@click.option(
    "--dense-mass/--sparse-mass",
    is_flag=True,
    default=False,
    help="Use dense mass matrix for the NUTS sampler.",
)
def nuts_cmd(
    template_path: str,
    data_path: str,
    output_path: str,
    skip_existing: bool,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    target_accept_prob: float,
    max_tree_depth: int,
    dense_mass: bool,
    progress_bar: bool,
    rebin: float | None,
):
    """
    Fit to spectral data using the NUTS sampler.
    """
    template = Template.from_json(Path(template_path))

    # Load the data given the template's redshift and mask
    data = load_data(data_path, template, rebin=rebin)

    config = Config(
        template=template,
        data=data,
        output_path=str(output_path),
        template_path=template_path,
        data_path=data_path,
        skip_existing=skip_existing,
    )

    model = NumpyroModel(
        config=config,
        integrator=quad_jax_integrate,
    ).setup()
    # initializer = LSQInitializer()
    initializer = SVIInitializer()

    # model = LSQModel(config).setup()
    # sampler = LSQSampler(estimate_uncertainties=False)

    sampler = NUTSSampler(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        dense_mass=dense_mass,
        progress_bar=progress_bar,
        initializer=initializer,
    )

    # Perform the sampling with the given configuration
    perform_sampling(config, model, sampler)
