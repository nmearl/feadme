from copy import deepcopy
from pathlib import Path

import click
import loguru
from astropy.table import Table
import arviz as az
from astropy.time import Time
import json
import numpy as np

from .compose import construct_model
from .models.lsq import lsq_model_fitter
from .parser import Config, Template, Data, NUTSSamplerSettings, SVISamplerSettings
from .utils import rebin_spectrum
from .samplers.nuts_sampler import NUTSSampler
from .samplers.svi_sampler import SVISampler

logger = loguru.logger.opt(colors=True)


def load_data(data_path: str, template: Template, rebin: bool = False) -> Data:
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

    if rebin:
        wave, flux, flux_err = rebin_spectrum(wave, flux, flux_err, dv=100)

    return Data.create(
        wave=wave,
        flux=flux,
        flux_err=flux_err,
        mask=template.mask,
    )


def run_pre_fit(template: Template, template_dict: dict, data: Data) -> Template:
    """
    Run a pre-fit using the least-squares model fitter to initialize
    the template parameters based on the provided data.

    Parameters
    ----------
    template : Template
        The template object containing the initial parameters.
    template_path : str
        Path to the template JSON file.
    data : Data
        The data object containing the wavelength, flux, and flux error.

    Returns
    -------
    Template
        The updated template object with parameters initialized from the pre-fit.
    """
    starters = lsq_model_fitter(template, data)[0]

    for dprof in template_dict["disk_profiles"] + template_dict["line_profiles"]:
        for dkey, dparam in dprof.items():
            if not isinstance(dparam, dict):
                continue

            dname = f"{dprof['name']}_{dparam['_field_name']}"

            if dname in starters:
                high_lim = dparam["high"]
                low_lim = dparam["low"]

                if "log" in dparam["distribution"].value:
                    scale = (high_lim / low_lim) ** (1 / 6)
                else:
                    scale = (high_lim - low_lim) / 6

                dparam["loc"] = starters[dname]
                dparam["scale"] = scale

                if dparam["distribution"].value in ["log_uniform", "log_half_normal"]:
                    dparam["distribution"] = "log_normal"
                elif dparam["distribution"].value in ["uniform", "half_normal"]:
                    dparam["distribution"] = "normal"

    for dprof in template_dict["disk_profiles"] + template_dict["line_profiles"]:
        for dkey, dparam in deepcopy(dprof).items():
            if dkey.startswith("_"):
                del dprof[dkey]

    return Template.from_dict(template_dict)


def perform_sampling(config: Config):
    """
    Perform MCMC sampling using the specified configuration.

    Parameters
    ----------
    config : Config
        Configuration object containing the template, data, sampler settings,
        and output paths.
    """
    template = config.template
    output_path = config.output_path

    # Start the fitting process
    logger.info(
        f"Starting fit of <cyan>{template.name}</cyan> using the "
        f"<magenta>{config.sampler_settings.sampler_type}</magenta> sampler."
    )

    if config.sampler_settings.sampler_type == "nuts":
        logger.info(
            f"<magenta>Proceeding using the {config.sampler_settings.chain_method}</magenta> method with "
            f"<light-magenta>{config.sampler_settings.num_chains}</light-magenta> chains "
            f"and <light-magenta>{config.sampler_settings.num_samples}</light-magenta> samples."
        )
        logger.info(
            f"Targetting acceptance probability of <light-magenta>{config.sampler_settings.target_accept_prob}</light-magenta> "
            f"with max tree depth of <light-magenta>{config.sampler_settings.max_tree_depth}</light-magenta> and a "
            f"<light-magenta>{'dense' if config.sampler_settings.dense_mass else 'sparse'}</light-magenta> mass matrix."
        )

    # Initialize the sampler with the model and configuration
    prior_model = construct_model(template, auto_reparam=False)
    model = construct_model(template, auto_reparam=False, circ_only=True)

    if config.sampler_settings.sampler_type == "nuts":
        sampler = NUTSSampler(model=model, config=config, prior_model=prior_model)
    elif config.sampler_settings.sampler_type == "svi":
        sampler = SVISampler(model=model, config=config, prior_model=prior_model)
    else:
        raise ValueError(
            f"Unknown sampler type: {config.sampler_settings.sampler_type}"
        )

    # sampler = DynestySampler(model=model, config=config, prior_model=prior_model)
    # sampler = JAXNSSampler(model=model, config=config, prior_model=prior_model)

    results_exist = (Path(output_path) / "results.nc").exists()

    # If a results file already exists, load it instead of running the sampler
    if results_exist:
        delta_time = None

        logger.info(
            f"Loading existing results at "
            f"<light-red>{output_path}/results.nc</light-red>."
        )
        sampler._idata = az.from_netcdf(
            f"{output_path}/results.nc",
        )
    else:
        start_time = Time.now()
        sampler.run()
        delta_time = (Time.now() - start_time).to_datetime()
        logger.info("Sampling completed.")

    logger.info("Displaying sampler results:\n" + sampler.summary.to_markdown())

    try:
        logger.info(
            f"Total divergences: {sampler._get_divergences()[0]} | "
            f"Rate: {sampler._get_divergences()[1]:.4f}%"
        )
    except:
        pass

    if not results_exist:
        sampler.write_results()

    logger.info("Generating plots...")
    sampler.plot_results()

    if delta_time is not None:
        logger.info(
            f"Finished processing <cyan>{template.name}</cyan> in "
            f"<green>{delta_time}</green>."
        )
    else:
        logger.info(f"Results loaded for <cyan>{template.name}</cyan>.")


@click.group()
def cli():
    """FEADME disk modeling CLI."""
    pass


@cli.command("nuts")
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
    "--num-warmup",
    type=int,
    default=1000,
    help="Number of warmup steps for the MCMC sampler.",
)
@click.option(
    "--num-samples",
    type=int,
    default=1000,
    help="Number of warmup steps for the MCMC sampler.",
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
    "--prefit",
    is_flag=True,
    default=False,
    help="Run a pre-fit to initialize parameters.",
)
@click.option(
    "--neutra",
    is_flag=True,
    default=False,
    help="Use Neutra initialization for the NUTS sampler.",
)
@click.option(
    "--progress-bar/--no-progress-bar",
    is_flag=True,
    default=True,
    help="Display a progress bar during sampling.",
)
@click.option(
    "--experimental-prefit/--no-experimental-prefit",
    is_flag=True,
    default=False,
    help="Run an experimental pre-fit to initialize parameters.",
)
def nuts_cmd(
    template_path: str,
    data_path: str,
    output_path: str,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    target_accept_prob: float,
    max_tree_depth: int,
    dense_mass: bool,
    prefit: bool,
    neutra: bool,
    progress_bar: bool,
    experimental_prefit: bool,
):
    """
    Fit to spectral data using the NUTS sampler.
    """
    # Parse the template from JSON
    # with open(template_path, "r") as f:
    #     template_dict = json.load(f)
    #     template_dict["white_noise"]["fixed"] = True
    #     template_dict["redshift"]["fixed"] = True
    #
    # template = Template.from_dict(template_dict)
    template = Template.from_json(Path(template_path))

    # Load the data given the template's redshift and mask
    data = load_data(data_path, template)

    # If pre-fitting is enabled, run the pre-fit to initialize parameters
    if experimental_prefit:
        template = run_pre_fit(template, template.to_dict(), data)

    # Create configuration object
    config = Config(
        template=template,
        data=data,
        sampler_settings=NUTSSamplerSettings(
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            target_accept_prob=target_accept_prob,
            max_tree_depth=max_tree_depth,
            dense_mass=dense_mass,
            prefit=prefit,
            neutra=neutra,
            progress_bar=progress_bar,
        ),
        output_path=output_path,
        template_path=template_path,
        data_path=data_path,
    )

    # Ensure the output directory exists
    output_path = Path(output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: <light-red>{output_path}</light-red>")

    # Perform the sampling with the given configuration
    perform_sampling(config)


@cli.command("svi")
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
    "--num-steps",
    type=int,
    default=25000,
    help="Number of optimization steps for SVI.",
)
@click.option(
    "--num-posterior-samples",
    type=int,
    default=2000,
    help="Number of posterior samples to draw after SVI optimization.",
)
@click.option(
    "--learning-rate",
    type=float,
    default=1e-3,
    help="Learning rate for the SVI optimizer.",
)
@click.option(
    "--decay-rate",
    type=float,
    default=0.1,
    help="Decay rate for the learning rate scheduler.",
)
@click.option(
    "--decay-steps",
    type=int,
    default=20000,
    help="Number of steps before applying learning rate decay.",
)
@click.option(
    "--hidden-factors",
    multiple=True,
    type=int,
    default=[8, 8],
    help="List of hidden layer sizes for the normalizing flow guide.",
)
@click.option(
    "--num-flows",
    type=int,
    default=2,
    help="Number of flows in the normalizing flow guide.",
)
@click.option(
    "--progress-bar/--no-progress-bar",
    is_flag=True,
    default=True,
    help="Display a progress bar during sampling.",
)
@click.option(
    "--experimental-prefit/--no-experimental-prefit",
    is_flag=True,
    default=False,
    help="Run an experimental pre-fit to initialize parameters.",
)
def svi_cmd(
    template_path: str,
    data_path: str,
    output_path: str,
    num_steps: int,
    num_posterior_samples: int,
    learning_rate: float,
    decay_rate: float,
    decay_steps: int,
    hidden_factors: list[int],
    num_flows: int,
    progress_bar: bool,
    experimental_prefit: bool,
):
    """
    Fit to spectral data using Stochastic Variational Inference (SVI).
    """
    # Parse the template from JSON
    # with open(template_path, "r") as f:
    #     template_dict = json.load(f)
    #     template_dict["white_noise"]["fixed"] = True
    #     template_dict["redshift"]["fixed"] = True
    #
    # template = Template.from_dict(template_dict)
    template = Template.from_json(Path(template_path))

    # Load the data given the template's redshift and mask
    data = load_data(data_path, template)

    # If pre-fitting is enabled, run the pre-fit to initialize parameters
    if experimental_prefit:
        template = run_pre_fit(template, template.to_dict(), data)

    # Create configuration object
    config = Config(
        template=template,
        data=data,
        sampler_settings=SVISamplerSettings(
            num_steps=num_steps,
            num_posterior_samples=num_posterior_samples,
            learning_rate=learning_rate,
            decay_rate=decay_rate,
            decay_steps=decay_steps,
            hidden_factors=hidden_factors,
            num_flows=num_flows,
            progress_bar=progress_bar,
        ),
        output_path=output_path,
        template_path=template_path,
        data_path=data_path,
    )

    # Ensure the output directory exists
    output_path = Path(output_path)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory: <light-red>{output_path}</light-red>")

    # Perform the sampling with the given configuration
    perform_sampling(config)
