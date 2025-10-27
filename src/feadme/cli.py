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
from .parser import Config, Template, Data, Sampler
from .samplers.nuts_sampler import NUTSSampler
from .utils import rebin_spectrum

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


def run_pre_fit(template: Template, template_path: str, data: Data) -> Template:
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
    with open(Path(template_path), "r") as f:
        template_dict = json.load(f)

    starters, _, _, _ = lsq_model_fitter(template, data)

    for dprof in template_dict["disk_profiles"] + template_dict["line_profiles"]:
        for _, dparam in dprof.items():
            if not isinstance(dparam, dict):
                continue

            dname = f"{dprof['name']}_{dparam['name']}"

            if dname in starters:
                high_lim = dparam["high"]
                low_lim = dparam["low"]

                if "log" in dparam["distribution"]:
                    scale = (
                        10 ** ((np.log10(high_lim) - np.log10(low_lim)) / 6)
                    ).item()
                else:
                    scale = (high_lim - low_lim) / 6

                dparam["loc"] = starters[dname][0].item()
                dparam["scale"] = scale

                if dparam["distribution"] in ["log_uniform", "log_half_normal"]:
                    dparam["distribution"] = "log_normal"
                elif dparam["distribution"] in ["uniform", "half_normal"]:
                    dparam["distribution"] = "normal"

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
        f"Starting fit of <cyan>{template.name}</cyan> using "
        f"<magenta>{config.sampler.chain_method}</magenta> method with "
        f"<light-magenta>{config.sampler.num_chains}</light-magenta> chains "
        f"and <light-magenta>{config.sampler.num_samples}</light-magenta> samples."
    )
    logger.info(
        f"Targetting acceptance probability of <light-magenta>{config.sampler.target_accept_prob}</light-magenta> "
        f"with max tree depth of <light-magenta>{config.sampler.max_tree_depth}</light-magenta> and a "
        f"<light-magenta>{'dense' if config.sampler.dense_mass else 'sparse'}</light-magenta> mass matrix."
    )

    # Initialize the sampler with the model and configuration
    model = construct_model(template, auto_reparam=True)
    sampler = NUTSSampler(model=model, config=config, prior_model=None)

    # If a results file already exists, load it instead of running the sampler
    if (Path(output_path) / "results.nc").exists():
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


@click.command()
@click.argument(
    "template-path",
    type=click.Path(exists=True),
    required=True,
    # help="Path to the template file.",
)
@click.argument(
    "data-path",
    type=click.Path(exists=True),
    required=True,
    # help="Path to the data file.",
)
@click.option(
    "--output-path",
    type=click.Path(),
    default="output",
    help="Directory to save output files and plots. Defaults to './output'.",
)
@click.option(
    "--sampler-type",
    type=click.Choice(["nuts"], case_sensitive=False),
    default="nuts",
    help="Type of NumPyro sampler to use.",
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
    help="Number of samples to draw from the posterior distribution.",
)
@click.option(
    "--num-chains",
    type=int,
    default=1,
    help="Number of MCMC chains to run.",
)
@click.option(
    "--progress-bar/--no-progress-bar",
    is_flag=True,
    default=True,
    help="Display a progress bar during sampling.",
)
@click.option(
    "--use-prefit/--no-prefit",
    is_flag=True,
    default=True,
    help="Perform LSQ pre-fit to initialize parameters.",
)
@click.option(
    "--use-neutra/--no-neutra",
    is_flag=True,
    default=False,
    help="Perform preprocessing with NeuTra.",
)
def cli(
    template_path: str,
    data_path: str,
    output_path: str,
    sampler_type: str,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    progress_bar: bool,
    use_prefit: bool = True,
    use_neutra: bool = False,
):
    """
    Command-line interface for the `feadme` package. Fits a template to
    spectral data using Jax and NumPyro.

    Parameters
    ----------
    template_path : str
        Path to the template JSON file.
    data_path : str
        Path to the data CSV file containing wavelength, flux, and flux error.
    output_path : str
        Directory to save output files and plots. Defaults to './output'.
    sampler_type : str
        Type of NumPyro sampler to use (currently only 'nuts' is supported).
    num_warmup : int
        Number of warmup steps for the MCMC sampler. Defaults to 1000.
    num_samples : int
        Number of samples to draw from the posterior distribution. Defaults to 1000.
    num_chains : int
        Number of MCMC chains to run. Defaults to 1.
    progress_bar : bool
        Display a progress bar during sampling. Defaults to True.
    auto_reparam : bool
        Automatically create reparameterization configuration for parameters.
    use_neutra : bool
        Perform NeuTra preprocessing to improve sampling.
    """
    # Parse the template from JSON
    template = Template.from_json(Path(template_path))

    # Run pre-fit to initialize parameters
    if use_prefit:
        logger.info("Running pre-fit to initialize parameters...")
        template = run_pre_fit(template, template_path, load_data(data_path, template))

    # Load the data given the template's redshift and mask
    data = load_data(data_path, template)

    # Create configuration object
    config = Config(
        template=template,
        data=data,
        sampler=Sampler(
            sampler_type=sampler_type,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            progress_bar=progress_bar,
            use_prefit=use_prefit,
            use_neutra=use_neutra,
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
