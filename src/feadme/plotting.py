from pathlib import Path

import arviz as az
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from arviz import InferenceData
import corner

from .compose import evaluate_model
from .parser import Template


def plot_hdi(
    idata: InferenceData,
    wave: jnp.ndarray | np.ndarray,
    flux: jnp.ndarray | np.ndarray,
    flux_err: jnp.ndarray | np.ndarray,
    output_path: str | Path,
    hdi_prob: float = 0.9,
):
    """
    Plot the highest density interval (HDI) of the posterior predictive distribution.

    Parameters
    ----------
    idata : InferenceData
        The inference data containing the posterior predictive samples.
    wave : jnp.ndarray or np.ndarray
        The wavelength array.
    flux : jnp.ndarray or np.ndarray
        The observed flux values.
    flux_err : jnp.ndarray or np.ndarray
        The observed flux errors.
    output_path : str or Path
        The path where the plot will be saved.
    hdi_prob : float, optional
        The probability for the HDI, by default 0.9.
    """
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")

    # Plot observed data
    ax.errorbar(wave, flux, yerr=flux_err, fmt="o", alpha=0.6, label="Observed")

    # Plot posterior predictive HDI
    az.plot_hdi(
        x=wave,
        y=idata.posterior_predictive["total_flux"],
        hdi_prob=hdi_prob,
        ax=ax,
        color="lightblue",
        fill_kwargs={"alpha": 0.5},
    )

    # Optionally add posterior predictive mean
    mean_flux = (
        idata.posterior_predictive["total_flux"].mean(dim=("chain", "draw")).values
    )
    ax.plot(wave, mean_flux, label="Posterior Predictive Mean")

    fig.savefig(f"{output_path}/hdi_plot.png")
    plt.close(fig)


def plot_model_fit(
    idata: InferenceData,
    summary: pd.DataFrame,
    template: Template,
    wave: jnp.ndarray | np.ndarray,
    flux: jnp.ndarray | np.ndarray,
    flux_err: jnp.ndarray | np.ndarray,
    output_path: str | Path,
    label: str,
):
    """
    Plot the model fit using the posterior distributions of the disk and line fluxes.

    Parameters
    ----------
    idata : InferenceData
        The inference data containing the posterior predictive samples.
    template : Template
        The template object containing the model parameters.
    summary : pd.DataFrame
        A DataFrame containing the summary statistics of the posterior distributions.
    wave : jnp.ndarray or np.ndarray
        The wavelength array.
    flux : jnp.ndarray or np.ndarray
        The observed flux values.
    flux_err : jnp.ndarray or np.ndarray
        The observed flux errors.
    output_path : str or Path
        The path where the plot will be saved.
    label : str
        A label for the plot, typically the name of the template or object being modeled.
    """
    fig, ax = plt.subplots(layout="constrained")

    ax.errorbar(
        wave, flux, yerr=flux_err, fmt="o", color="grey", zorder=-10, alpha=0.25
    )

    # Plot the posterior distributions for disk and line flux
    for var in ["disk_flux", "line_flux"]:
        var_name = " ".join([x.capitalize() for x in var.split("_")])
        var_dist = idata.posterior_predictive[var].mean(dim=("chain",)).values
        median = np.percentile(var_dist, 50, axis=0)
        ax.plot(wave, median, label=f"Sampled {var_name}")

    obs_dist = (
        idata.posterior_predictive["total_flux"].stack(sample=("chain", "draw")).values
    )
    median = np.percentile(obs_dist, 50, axis=1)
    lower = np.percentile(obs_dist, 16, axis=1)
    upper = np.percentile(obs_dist, 84, axis=1)

    ax.plot(wave, median, label="Sampled Model Fit", color="C3")
    ax.fill_between(wave, lower, upper, alpha=0.5, color="C3")

    # Reconstruct the model from the median of the posteriors
    param_mods = summary["median"].to_dict()
    param_mods.update(
        {
            f"{prof.name}_{param.name}": param.value
            for prof in template.disk_profiles + template.line_profiles
            for param in prof.fixed
        }
    )
    tot_flux, disk_flux, line_flux = evaluate_model(template, wave, param_mods)

    ax.plot(wave, tot_flux, label="Reconstructed Model", linestyle="--")
    ax.plot(wave, disk_flux, label="Reconstructed Disk Flux", linestyle="--")
    ax.plot(wave, line_flux, label="Reconstructed Line Flux", linestyle="--")

    ax.set_ylabel("Flux [mJy]")
    ax.set_xlabel("Wavelength [AA]")
    ax.set_title(f"{label} Model Fit")
    ax.legend()

    fig.savefig(f"{output_path}/model_fit.png")
    plt.close(fig)


def plot_corner(
    idata: InferenceData,
    output_path: str | Path,
    ignored_vars: list[str] = None,
):
    """
    Create a corner plot of the posterior distributions of the model parameters.

    Parameters
    ----------
    idata : InferenceData
        The inference data containing the posterior samples.
    output_path : str or Path
        The path where the corner plot will be saved.
    ignored_vars : list of str, optional
        A list of variable names to ignore in the corner plot. Defaults to None.
    """
    if ignored_vars is None:
        ignored_vars = []

    # Filter out ignored variables
    var_names = [var for var in idata.posterior.data_vars if var not in ignored_vars]

    samples_ds = az.extract(idata, var_names=var_names, combined=True)
    samples = np.vstack([samples_ds[var].values for var in var_names]).T

    # Compute quantiles
    quantiles = [0.16, 0.5, 0.84]

    # Create the corner plot
    fig = corner.corner(
        samples,
        labels=var_names,
        quantiles=quantiles,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        plot_density=True,
        plot_contours=True,
        fill_contours=True,
        axes_scale=[
            "log" if "vel_width" in x or "radius" in x or "sigma" in x else "linear"
            for x in var_names
        ],
    )

    fig.savefig(f"{output_path}/corner_plot.png")
    plt.close(fig)


def plot_corner_priors(
    idata: InferenceData,
    output_path: str | Path,
    ignored_vars: list[str] = None,
):
    """
    Create a corner plot of the prior distributions of the model parameters.

    Parameters
    ----------
    idata : InferenceData
        The inference data containing the prior samples.
    output_path : str or Path
        The path where the corner plot will be saved.
    ignored_vars : list of str, optional
        A list of variable names to ignore in the corner plot. Defaults to None.
    """
    if ignored_vars is None:
        ignored_vars = []

    # Filter out ignored variables
    var_names = [var for var in idata.posterior.data_vars if var not in ignored_vars]

    samples_ds = az.extract(idata, group="prior", var_names=var_names, combined=True)
    samples = np.vstack([samples_ds[var].values for var in var_names]).T

    # Compute quantiles
    quantiles = [0.16, 0.5, 0.84]

    # Create the corner plot
    fig = corner.corner(
        samples,
        labels=var_names,
        quantiles=quantiles,
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 12},
        label_kwargs={"fontsize": 14},
        plot_density=True,
        plot_contours=True,
        fill_contours=True,
        axes_scale=[
            "log" if "vel_width" in x or "radius" in x or "sigma" in x else "linear"
            for x in var_names
        ],
    )

    fig.savefig(f"{output_path}/corner_plot_priors.png")
    plt.close(fig)
