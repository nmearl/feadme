import matplotlib.pyplot as plt
import arviz as az
from arviz import InferenceData
import numpy as np
from .parser import Template
from pathlib import Path


def plot_hdi(
    idata: InferenceData, wave, flux, flux_err, output_path: str | Path, hdi_prob=0.9
):
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
    idata,
    wave,
    flux,
    flux_err,
    output_path: str | Path,
    label,
):
    fig, ax = plt.subplots(layout="constrained")

    ax.errorbar(
        wave, flux, yerr=flux_err, fmt="o", color="grey", zorder=-10, alpha=0.25
    )

    for var in ["disk_flux", "line_flux"]:
        var_dist = idata.posterior_predictive[var].mean(dim=("chain",)).values
        median = np.percentile(var_dist, 50, axis=0)
        ax.plot(wave, median, label=f"{var}")

    obs_dist = (
        idata.posterior_predictive["total_flux"].stack(sample=("chain", "draw")).values
    )
    median = np.percentile(obs_dist, 50, axis=1)
    lower = np.percentile(obs_dist, 16, axis=1)
    upper = np.percentile(obs_dist, 84, axis=1)
    ax.plot(wave, median, label="Model Fit", color="C3")
    ax.fill_between(wave, lower, upper, alpha=0.5, color="C3")

    ax.set_ylabel("Flux [mJy]")
    ax.set_xlabel("Wavelength [AA]")
    ax.set_title(f"{label} Model Fit")
    ax.legend()

    fig.savefig(f"{output_path}/model_fit.png")
    plt.close(fig)
