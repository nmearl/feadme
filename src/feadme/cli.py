from pathlib import Path

import click
import loguru
from astropy.table import Table
import arviz as az
import time

from .compose import create_optimized_model
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
def cli(
    template_path: str,
    data_path: str,
    output_path: str,
    sampler_type: str,
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    progress_bar: bool,
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

    # Create config
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
            f"Sampling completed for `{template.name}` in `{run_time}s`."
        )

    loguru.logger.info(sampler.summary)
    sampler.write_results()
    sampler.plot_results()
