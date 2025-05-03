import jax

from numpyro.infer import Predictive
import matplotlib.pyplot as plt
import astropy.uncertainty as unc

import corner
import arviz as az
import numpy as np

from .compose import evaluate_disk_model

az.rcParams["plot.max_subplots"] = 200


def plot_results(
    template,
    output_dir,
    posterior_predictive_samples_transformed,
    idata_transformed,
    wave,
    flux,
    flux_err,
    label,
):
    axes = az.plot_trace(
        idata_transformed,
        var_names=list(idata_transformed.posterior.keys()),
        compact=True,
        backend_kwargs={"layout": "constrained"},
    )

    fig = axes.ravel()[0].figure
    fig.savefig(f"{output_dir}/trace_plot.png")

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")

    ax.plot(wave, flux)
    az.plot_hdi(
        ax=ax,
        x=wave,
        y=idata_transformed["posterior_predictive"]["total_flux"],
        fill_kwargs={"alpha": 0.5},
        color="C1",
    )
    fig.savefig(f"{output_dir}/hdi_plot.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.errorbar(
        wave,
        flux,
        yerr=flux_err,
        fmt="o",
        color="grey",
        # markeredgecolor="grey",
        # ecolor="grey",
        zorder=-10,
        alpha=0.25,
    )

    for var in ["disk_flux", "line_flux"]:
        var_dist = posterior_predictive_samples_transformed[var]
        median = np.percentile(var_dist, 50, axis=0)
        ax.plot(wave, median, label=f"{var}")

    obs_dist = posterior_predictive_samples_transformed["total_flux"]
    median = np.percentile(obs_dist, 50, axis=0)
    lower_lim = np.percentile(obs_dist, 16, axis=0)
    upper_lim = np.percentile(obs_dist, 84, axis=0)
    ax.plot(wave, median, label="Model Fit", color="C3")
    ax.fill_between(wave, lower_lim, upper_lim, alpha=0.5, color="C3")

    res_pars = {}

    for var in posterior_predictive_samples_transformed.keys():
        if "_flux" in var:
            continue

        var_dist = posterior_predictive_samples_transformed[var]

        if var.endswith('apocenter'):
            median = np.arctan2(np.mean(np.sin(var_dist)), np.mean(np.cos(var_dist))) % (2 * np.pi)
        else:
            median = np.percentile(var_dist, 50, axis=0)

        res_pars[var] = median

    res_flux, res_disk_flux, res_line_flux = evaluate_disk_model(
        template, wave, res_pars
    )

    ax.plot(wave, res_flux, label="R. Model Fit", color="C3")
    ax.plot(wave, res_disk_flux, label="R. Disk Model", color="C4")
    ax.plot(wave, res_line_flux, label="R. Line Model", color="C5")

    ax.set_ylabel("Flux [mJy]")
    ax.set_xlabel("Wavelength [AA]")
    ax.set_title(f"{label} Model Fit")

    ax.legend()
    fig.savefig(f"{output_dir}/model_fit.png")
    plt.close(fig)

    names = [
        x
        for x in idata_transformed.posterior.keys()
        if "_flux" not in x and "_base" not in x
    ]

    fig = corner.corner(
        idata_transformed,
        var_names=names,
        labels=names,
        quantiles=[0.16, 0.5, 0.84],
        smooth=1,
        show_titles=True,
        axes_scale=[
            "log" if "vel_width" in x or "radius" in x else "linear" for x in names
        ],
    )
    fig.savefig(f"{output_dir}/corner_plot.png")
    plt.close(fig)
    plt.close(fig)
