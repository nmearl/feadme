from pathlib import Path

import click
import loguru
from astropy.table import Table
import arviz as az
import time
import json
import numpy as np

from .compose import create_optimized_model
from .models.lsq import lsq_model_fitter
from .parser import Config, Template, Data, Sampler
from .samplers.nuts_sampler import NUTSSampler


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
    required=False,
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
    "--progress-bar",
    is_flag=True,
    default=True,
    help="Display a progress bar during sampling.",
)
@click.option(
    "--pre-fit",
    is_flag=True,
    default=False,
    help="Run a pre-fit using the least-squares model fitter before sampling.",
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
    pre_fit: bool = False,
):
    loguru.logger.info("Starting FEADME CLI...")

    # Load template
    # if Path(template_path).is_dir():
    #     template_files = list(Path(template_path).glob("*.json"))
    #     if not template_files:
    #         raise ValueError(f"No template files found in directory {template_path}.")
    #     templates = [
    #         Template(**json.loads(file.read_text())) for file in template_files
    #     ]
    # else:
    #     templates = [Template(**json.loads(Path(template_path).read_text()))]

    template = Template.from_json(Path(template_path))

    # Load data
    data_tab = Table.read(
        data_path, format="ascii.csv", names=["wave", "flux", "flux_err"]
    )

    data = Data.create(
        wave=data_tab["wave"] / (1 + template.redshift),
        flux=data_tab["flux"],
        flux_err=data_tab["flux_err"],
        mask=template.mask,
    )

    # If a pre-fit is requested, run the least-squares model fitter and
    # update the template parameters
    if pre_fit:
        with open(Path(template_path), "r") as f:
            template_dict = json.load(f)

        starters = lsq_model_fitter(template, data, show_plot=False)

        for dprof in template_dict["disk_profiles"] + template_dict["line_profiles"]:
            for _, dparam in dprof.items():
                if not isinstance(dparam, dict):
                    print(f"Skipping non-dict parameter: {type(dparam)}")
                    continue

                dname = f"{dprof['name']}_{dparam['name']}"

                if dname in starters:
                    dparam["loc"] = starters[dname][0].item()
                    dparam["scale"] = (dparam["high"] - dparam["low"]) / np.sqrt(
                        2 * np.pi
                    )

                    if "log" in dparam["distribution"]:
                        dparam["scale"] = 10 ** (
                            (np.log10(dparam["high"]) - np.log10(dparam["low"]))
                            / np.sqrt(2 * np.pi)
                        )

                    if dparam["distribution"] == "log_uniform":
                        dparam["distribution"] = "log_normal"
                    elif dparam["distribution"] == "uniform":
                        dparam["distribution"] = "normal"

        template = Template.from_dict(template_dict)

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
        ),
        output_path=output_path,
        template_path=template_path,
        data_path=data_path,
    )

    loguru.logger.info(
        f"Starting fit of `{template.name}` using method "
        f"`{config.sampler.chain_method}` with `{config.sampler.num_chains}` "
        f"chains and `{config.sampler.num_samples}` samples."
    )

    model = create_optimized_model(template)
    sampler = NUTSSampler(model=model, config=config)

    if (Path(output_path) / "results.nc").exists():
        loguru.logger.info(
            f"Sampler results already exist at "
            f"`{output_path}/sampler_results.nc`. Loading existing results."
        )
        sampler._idata = az.from_netcdf(
            f"{output_path}/results.nc",
        )
    else:
        start_time = time.time()
        sampler.run()
        run_time = time.time() - start_time
        loguru.logger.info(
            f"Sampling completed for `{template.name}` in `{run_time:.2f}s`."
        )

    loguru.logger.info("\n" + sampler.summary.to_markdown())
    sampler.write_results()
    sampler.plot_results()
