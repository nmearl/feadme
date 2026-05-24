from pathlib import Path

import loguru
from astropy.table import Table
from astropy.time import Time
import arviz as az
import click
from functools import wraps
import jax
import numpy as np
import astropy.constants as const

from .core.parser import Template, Config, Data, Mask
from .sampling.initializers import (
    DefaultInitializer,
    PathfinderInitializer,
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

INTEGRATORS = {
    "quad": quad_jax_integrate,
    "mixed": mixed_jax_integrate,
    "trap": trap_jax_integrate,
    "split_quad": split_quad_jax_integrate,
}


def default_integrator_name() -> str:
    return "mixed" if jax.default_backend() == "gpu" else "quad"


def log_grid_debug(template: Template, data: Data) -> None:
    masked_count = int(np.asarray(data.masked_wave).shape[0])
    logger.debug(f"Masked wavelength bins retained: {masked_count}")

    if template.disk_profiles:
        logger.debug(
            f"Disk evaluation for <cyan>{template.name}</cyan>: "
            f"{masked_count} wavelength bins across {len(template.disk_profiles)} disk profile(s) "
            f"(X-grid derived per sample from observed wavelengths)"
        )


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
    start_time = Time.now()

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
        idata = sampler(config, model)

    # Report results and write to disk
    reporter = Reporter(config=config, idata=idata)

    if not results_exist:
        reporter.write_netcdf()
        logger.info(
            f"Results written to <green>{config.output_path}/results.nc</green>."
        )

    reporter.write_summary()
    logger.info(f"Summary written to <green>{config.output_path}/summary.csv</green>.")

    logger.info("Displaying sampler results:\n" + reporter.summary.to_markdown())

    # Plot results and save out figures
    plotter = Plotter(config=config, idata=idata, summary=reporter.summary)

    plotter.plot_model_fit()
    plotter.plot_corner()

    if sampler.compute_prior_predictive:
        plotter.plot_prior_corner()
        plotter.plot_trace()

    logger.info(f"Plots saved to <green>{config.output_path}</green>.")

    delta_time = (Time.now() - start_time).to_datetime()

    logger.info(
        f"Finished processing <cyan>{config.template.name}</cyan> in "
        f"<green>{delta_time}</green>."
    )


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
    @click.option(
        "--debug-plot/--no-debug-plot",
        is_flag=True,
        default=False,
        help="Save diagnostic plots from the initializer (LSQ/SVI fit and corner).",
    )
    @wraps(f)
    def wrapper(*args, **kwargs):
        return f(*args, **kwargs)

    return wrapper


@click.group()
def cli():
    """FEADME disk modeling CLI."""
    pass


@cli.command("run")
@common_options
@click.option(
    "--sampler",
    type=click.Choice(["nuts", "neutra"], case_sensitive=False),
    default="nuts",
    show_default=True,
    help="Sampler to use.",
)
@click.option(
    "--init-method",
    type=click.Choice(["svi", "pathfinder"], case_sensitive=False),
    default="svi",
    show_default=True,
    help="Initializer basin-refinement method to use before NUTS.",
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
@click.option(
    "--integrator",
    type=click.Choice(sorted(INTEGRATORS)),
    default=None,
    help="Disk integrator to use. Defaults to quad on CPU and mixed on GPU.",
)
@click.option(
    "--lsq-init-candidates",
    type=int,
    default=1,
    show_default=True,
    help=(
        "Number of structured LSQ starting basins to try before SVI/NUTS "
        "initialization. Use 1 to recover the old single-start behavior."
    ),
)
@click.option(
    "--svi-init-candidates",
    type=int,
    default=1,
    show_default=True,
    help="Number of distinct LSQ basins to refine independently with SVI before NUTS.",
)
@click.option(
    "--init-candidate-distance-threshold",
    type=float,
    default=0.25,
    show_default=True,
    help="Transformed-parameter distance below which initialization basins are treated as duplicates.",
)
@click.option(
    "--svi-init-steps",
    type=int,
    default=2000,
    show_default=True,
    help="Number of SVI optimization steps for each initialization candidate.",
)
@click.option(
    "--svi-init-samples",
    type=int,
    default=1000,
    show_default=True,
    help="Number of guide samples used to score each SVI initialization candidate.",
)
@click.option(
    "--svi-init-max-loss-relative-std",
    type=float,
    default=0.10,
    show_default=True,
    help="Maximum recent-loss relative std for an SVI candidate to be eligible for selection.",
)
@click.option(
    "--pathfinder-init-candidates",
    type=int,
    default=8,
    show_default=True,
    help="Number of distinct LSQ basins to refine with multipathfinder before NUTS.",
)
@click.option(
    "--pathfinder-init-samples",
    type=int,
    default=32,
    show_default=True,
    help="Number of approximate posterior samples drawn per Pathfinder path.",
)
@click.option(
    "--pathfinder-score-batch-size",
    type=int,
    default=8,
    show_default=True,
    help="Number of Pathfinder samples scored at once to bound initializer memory use.",
)
@click.option(
    "--pathfinder-init-maxiter",
    type=int,
    default=30,
    show_default=True,
    help="Maximum L-BFGS iterations per Pathfinder path.",
)
def run_cmd(
    template_path: str,
    data_path: str,
    output_path: str,
    skip_existing: bool,
    sampler: str,
    init_method: str,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    target_accept_prob: float,
    max_tree_depth: int,
    dense_mass: bool,
    integrator: str | None,
    lsq_init_candidates: int,
    svi_init_candidates: int,
    init_candidate_distance_threshold: float,
    svi_init_steps: int,
    svi_init_samples: int,
    svi_init_max_loss_relative_std: float,
    pathfinder_init_candidates: int,
    pathfinder_init_samples: int,
    pathfinder_score_batch_size: int,
    pathfinder_init_maxiter: int,
    compute_prior_predictive: bool,
    progress_bar: bool,
    rebin: float | None,
    debug_plot: bool,
):
    """
    Fit spectral data using the specified sampler (default: nuts).
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

    integrator_name = integrator or default_integrator_name()
    integrator_fn = INTEGRATORS[integrator_name]
    logger.debug(f"Using disk integrator: <cyan>{integrator_name}</cyan>")

    model = NumpyroModel(
        config=config,
        integrator=integrator_fn,
    ).setup()
    # initializer = LSQInitializer()
    if init_method.lower() == "pathfinder":
        initializer = PathfinderInitializer(
            debug_plot=debug_plot,
            lsq_candidates=max(1, int(lsq_init_candidates)),
            pathfinder_candidates=max(1, int(pathfinder_init_candidates)),
            candidate_distance_threshold=init_candidate_distance_threshold,
            num_samples=max(1, int(pathfinder_init_samples)),
            score_batch_size=max(1, int(pathfinder_score_batch_size)),
            maxiter=max(1, int(pathfinder_init_maxiter)),
        )
    else:
        initializer = SVIInitializer(
            debug_plot=debug_plot,
            lsq_candidates=max(1, int(lsq_init_candidates)),
            svi_candidates=max(1, int(svi_init_candidates)),
            candidate_distance_threshold=init_candidate_distance_threshold,
            max_candidate_loss_relative_std=svi_init_max_loss_relative_std,
            num_steps=max(1, int(svi_init_steps)),
            num_samples=max(1, int(svi_init_samples)),
        )

    # model = LSQModel(config).setup()
    # sampler = LSQSampler(estimate_uncertainties=False)

    sampler_kwargs = dict(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        target_accept_prob=target_accept_prob,
        max_tree_depth=max_tree_depth,
        dense_mass=dense_mass,
        compute_prior_predictive=compute_prior_predictive,
        progress_bar=progress_bar,
        initializer=initializer,
    )
    if sampler.lower() == "neutra":
        sampler_obj = NeuTraSampler(**sampler_kwargs)
    else:
        sampler_obj = NUTSSampler(**sampler_kwargs)

    # Perform the sampling with the given configuration
    perform_sampling(config, model, sampler_obj)
