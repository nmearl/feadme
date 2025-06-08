import json
from pathlib import Path
from typing import Optional

import click
import jax
import numpy as np
from astropy.table import Table
from loguru import logger
from numpyro.infer import init_to_median

from .compose import disk_model
from .parser import Template
from .samplers import create_sampler
from .models.lsq import lsq_model_fitter

finfo = np.finfo(float)


def load_template(path: Path) -> Template:
    """Load and parse a JSON template file."""
    with path.open("r") as f:
        data = json.load(f)
    return Template(**data)


def load_data(data_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read an ASCII CSV with columns (wave, flux, flux_err)."""
    tbl = Table.read(data_path, format="ascii.csv", names=("wave", "flux", "flux_err"))
    return (
        np.asarray(tbl["wave"], float),
        np.asarray(tbl["flux"], float),
        np.asarray(tbl["flux_err"], float),
    )


def fit_initial_priors(
    template: Template,
    wave: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    use_quad: bool,
) -> None:
    """
    Run the least‐squares fitter to get starter values, then
    adjust each profile parameter’s loc/scale/distribution.
    """
    starters = lsq_model_fitter(template, wave, flux, flux_err, use_quad=use_quad)

    for profile in template.disk_profiles + template.line_profiles:
        for param in profile._independent():
            key = f"{profile.name}_{param.name}"
            loc_val = starters.get(key, (param.loc,))[0]
            param.loc = loc_val

            # default linear‐space scale
            linear_scale = (param.high - param.low) / np.sqrt(2 * np.pi)
            if "log" in param.distribution:
                param.scale = 10 ** (
                    (np.log10(param.high) - np.log10(param.low)) / np.sqrt(2 * np.pi)
                )
            else:
                param.scale = linear_scale

            # canonicalize distributions
            if param.distribution == "log_uniform":
                param.distribution = "log_normal"
            elif param.distribution == "uniform":
                param.distribution = "normal"


def run_sampling_for_template(
    template_path: Path,
    data_file: Path,
    output_dir: Path,
    label: Optional[str],
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    show_progress: bool,
    use_quad: bool,
) -> None:
    """
    Process one template: load it, load data, fit priors, run NUTS
    and write the .nc file.
    """
    tpl = load_template(template_path)
    obj_label = label or tpl.name
    out_dir = output_dir / obj_label
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / f"{obj_label}.nc"
    if out_file.exists():
        logger.info(f"[{obj_label}] output already exists at {out_file}, skipping.")
        return

    wave, flux, flux_err = load_data(data_file)
    logger.info(f"[{obj_label}] loaded data ({len(wave)} points).")

    fit_initial_priors(tpl, wave, flux, flux_err, use_quad)

    sampler = create_sampler(
        sampler_type="nuts",
        model=disk_model,
        template=tpl,
        wave=wave,
        flux=flux,
        flux_err=flux_err,
        output_dir=str(out_dir),
        label=obj_label,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=show_progress,
    )

    if not sampler.check_convergence():
        sampler.sample(init_strategy=init_to_median(num_samples=1000))
        sampler.write_run()
        logger.success(f"[{obj_label}] sampling complete, results written.")
    else:
        logger.info(f"[{obj_label}] already converged, skipping sampling.")

    # If you find you still need to clear JAX caches to keep memory usage down,
    # you can uncomment the next line:
    # jax.clear_caches()


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.argument("template_path", type=click.Path(exists=True, path_type=Path))
@click.argument(
    "data_file", type=click.Path(exists=True, path_type=Path), required=False
)
@click.option(
    "--override-data-dir",
    "override_dir",
    type=click.Path(path_type=Path),
    help="If set, use this directory for data files instead of what's in the template.",
)
@click.option(
    "--pattern",
    type=str,
    default=None,
    help="Only process templates whose filename contains this substring.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default="output",
    show_default=True,
    help="Base directory to save all outputs.",
)
@click.option(
    "--label",
    type=str,
    default=None,
    help="Override the label used for naming outputs (defaults to template.name).",
)
@click.option(
    "--num-warmup",
    type=int,
    default=1000,
    show_default=True,
    help="Number of warmup steps for NUTS.",
)
@click.option(
    "--num-samples",
    type=int,
    default=2000,
    show_default=True,
    help="Number of posterior samples to draw.",
)
@click.option(
    "--num-chains",
    type=int,
    default=lambda: jax.local_device_count(),
    show_default="jax.local_device_count()",
    help="How many chains to run in parallel.",
)
@click.option(
    "--progress/--no-progress",
    default=True,
    show_default=True,
    help="Show a progress bar during sampling.",
)
@click.option(
    "--use-quad/--no-use-quad",
    default=False,
    show_default=True,
    help="Use quadrature rules in the least‐squares fit.",
)
def run(
    template_path: Path,
    data_file: Optional[Path],
    override_dir: Optional[Path],
    pattern: Optional[str],
    output_dir: Path,
    label: Optional[str],
    num_warmup: int,
    num_samples: int,
    num_chains: int,
    progress: bool,
    use_quad: bool,
):
    # Resolve data directory
    if override_dir:
        base_data_dir = override_dir
    else:
        base_data_dir = None

    # Collect all templates
    if template_path.is_dir():
        all_tpls = sorted(template_path.glob("*.json"))
    else:
        all_tpls = [template_path]

    for tpl_file in all_tpls:
        if pattern and pattern not in tpl_file.name:
            logger.debug(
                f"Skipping {tpl_file.name} (does not match pattern `{pattern}`)."
            )
            continue

        # decide data file for this template
        if data_file:
            df = data_file
        else:
            tpl = load_template(tpl_file)
            df = Path(tpl.data_path)
            if base_data_dir:
                df = base_data_dir / df.name

        try:
            run_sampling_for_template(
                tpl_file,
                df,
                output_dir,
                label,
                num_warmup,
                num_samples,
                num_chains,
                show_progress=progress,
                use_quad=use_quad,
            )
        except Exception:
            logger.exception(
                f"Error processing template {tpl_file.name}, continuing to next one."
            )
