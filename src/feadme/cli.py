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
    JAXLSQInitializer,
    PathfinderInitializer,
    MAPInitializer,
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
    type=click.Choice(["svi", "pathfinder", "map", "jax-lsq"], case_sensitive=False),
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
    "--lsq-init-maxiter",
    type=int,
    default=2000,
    show_default=True,
    help="Maximum optimizer iterations for each LSQ initialization start.",
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
    help="Number of distinct start basins to refine with multipathfinder before NUTS.",
)
@click.option(
    "--pathfinder-start-method",
    type=click.Choice(["lsq", "structured"], case_sensitive=False),
    default="lsq",
    show_default=True,
    help=(
        "How to generate Pathfinder starting basins. 'lsq' runs the structured "
        "LSQ pre-fit; 'structured' bypasses LSQ and uses structured template "
        "starts directly."
    ),
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
@click.option(
    "--map-init-candidates",
    type=int,
    default=64,
    show_default=True,
    help="Number of independent starts to optimize with batched MAP initialization.",
)
@click.option(
    "--map-start-method",
    type=click.Choice(["prior", "structured"], case_sensitive=False),
    default="prior",
    show_default=True,
    help="How to generate batched MAP starts before gradient optimization.",
)
@click.option(
    "--map-selection-score",
    type=click.Choice(["likelihood", "posterior"], case_sensitive=False),
    default="likelihood",
    show_default=True,
    help=(
        "Score used to select among optimized MAP candidates. The optimizer "
        "still follows the posterior; this controls final basin ranking."
    ),
)
@click.option(
    "--map-init-steps",
    type=int,
    default=200,
    show_default=True,
    help="Number of gradient optimization steps for each MAP initialization start.",
)
@click.option(
    "--map-init-learning-rate",
    type=float,
    default=1e-2,
    show_default=True,
    help="Adam learning rate for batched MAP initialization.",
)
@click.option(
    "--map-init-grad-clip",
    type=float,
    default=10.0,
    show_default=True,
    help="Global gradient-norm clipping threshold for batched MAP initialization.",
)
@click.option(
    "--jax-lsq-init-candidates",
    type=int,
    default=64,
    show_default=True,
    help="Number of independent starts to optimize with batched JAX-LSQ initialization.",
)
@click.option(
    "--jax-lsq-start-method",
    type=click.Choice(["prior", "structured"], case_sensitive=False),
    default="structured",
    show_default=True,
    help="How to generate batched JAX-LSQ starts before gradient optimization.",
)
@click.option(
    "--jax-lsq-init-steps",
    type=int,
    default=500,
    show_default=True,
    help="Number of gradient optimization steps for each JAX-LSQ initialization start.",
)
@click.option(
    "--jax-lsq-init-learning-rate",
    type=float,
    default=3e-3,
    show_default=True,
    help="Adam learning rate for batched JAX-LSQ initialization.",
)
@click.option(
    "--jax-lsq-init-grad-clip",
    type=float,
    default=10.0,
    show_default=True,
    help="Global gradient-norm clipping threshold for batched JAX-LSQ initialization.",
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
    lsq_init_maxiter: int,
    svi_init_candidates: int,
    init_candidate_distance_threshold: float,
    svi_init_steps: int,
    svi_init_samples: int,
    svi_init_max_loss_relative_std: float,
    pathfinder_init_candidates: int,
    pathfinder_start_method: str,
    pathfinder_init_samples: int,
    pathfinder_score_batch_size: int,
    pathfinder_init_maxiter: int,
    map_init_candidates: int,
    map_start_method: str,
    map_selection_score: str,
    map_init_steps: int,
    map_init_learning_rate: float,
    map_init_grad_clip: float,
    jax_lsq_init_candidates: int,
    jax_lsq_start_method: str,
    jax_lsq_init_steps: int,
    jax_lsq_init_learning_rate: float,
    jax_lsq_init_grad_clip: float,
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
            lsq_maxiter=max(1, int(lsq_init_maxiter)),
            pathfinder_candidates=max(1, int(pathfinder_init_candidates)),
            start_method=pathfinder_start_method.lower(),
            candidate_distance_threshold=init_candidate_distance_threshold,
            num_samples=max(1, int(pathfinder_init_samples)),
            score_batch_size=max(1, int(pathfinder_score_batch_size)),
            maxiter=max(1, int(pathfinder_init_maxiter)),
        )
    elif init_method.lower() == "map":
        initializer = MAPInitializer(
            debug_plot=debug_plot,
            candidates=max(1, int(map_init_candidates)),
            start_method=map_start_method.lower(),
            selection_score=map_selection_score.lower(),
            candidate_distance_threshold=init_candidate_distance_threshold,
            num_steps=max(1, int(map_init_steps)),
            learning_rate=float(map_init_learning_rate),
            grad_clip=float(map_init_grad_clip),
        )
    elif init_method.lower() == "jax-lsq":
        initializer = JAXLSQInitializer(
            debug_plot=debug_plot,
            candidates=max(1, int(jax_lsq_init_candidates)),
            start_method=jax_lsq_start_method.lower(),
            selection_score="likelihood",
            candidate_distance_threshold=init_candidate_distance_threshold,
            num_steps=max(1, int(jax_lsq_init_steps)),
            learning_rate=float(jax_lsq_init_learning_rate),
            grad_clip=float(jax_lsq_init_grad_clip),
        )
    else:
        initializer = SVIInitializer(
            debug_plot=debug_plot,
            lsq_candidates=max(1, int(lsq_init_candidates)),
            lsq_maxiter=max(1, int(lsq_init_maxiter)),
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
