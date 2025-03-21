import jax

from numpyro.infer import Predictive
import matplotlib.pyplot as plt
import astropy.uncertainty as unc

import corner
import arviz as az
import numpy as np


def plot_results(
    output_dir,
    posterior_predictive_samples_transformed,
    idata_transformed,
    wave,
    flux,
    flux_err,
    label,
):
    # axes = az.plot_trace(
    #     idata_transformed,
    #     var_names=[x for x in idata_transformed.posterior.keys() if "_base" not in x],
    #     compact=True,
    #     backend_kwargs={"layout": "constrained"},
    # )
    #
    # fig = axes.ravel()[0].figure
    # fig.tight_layout()
    # fig.savefig(f"{output_dir}/trace_plot.png")

    axes = az.plot_trace(
        idata_transformed,
        var_names=[x for x in idata_transformed.posterior.keys() if "_base" in x],
        compact=True,
        backend_kwargs={"layout": "constrained"},
    )

    fig = axes.ravel()[0].figure
    fig.tight_layout()
    fig.savefig(f"{output_dir}/base_trace_plot.png")

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")

    ax.plot(wave, flux)
    az.plot_hdi(
        ax=ax,
        x=wave,
        y=idata_transformed["posterior_predictive"]["obs"],
        fill_kwargs={"alpha": 0.5},
        color="C1",
    )
    fig.savefig(f"{output_dir}/hdi_plot.png")

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

    obs_dist = posterior_predictive_samples_transformed["obs"]
    median = np.percentile(obs_dist, 50, axis=0)
    lower_lim = np.percentile(obs_dist, 16, axis=0)
    upper_lim = np.percentile(obs_dist, 84, axis=0)
    ax.plot(wave, median, label="Model Fit", color="C3")
    ax.fill_between(wave, lower_lim, upper_lim, alpha=0.5, color="C3")

    ax.set_ylabel("Flux [mJy]")
    ax.set_xlabel("Wavelength [AA]")
    ax.set_title(f"{label} Model Fit")

    ax.legend()
    fig.savefig(f"{output_dir}/model_fit.png")

    fig = corner.corner(
        idata_transformed,
        var_names=[
            x
            for x in idata_transformed.posterior.keys()
            if "_base" not in x and "_flux" not in x
        ],
        labels=[
            x
            for x in idata_transformed.posterior.keys()
            if "_base" not in x and "_flux" not in x
        ],
        quantiles=[0.16, 0.5, 0.84],
        smooth=1,
        show_titles=True,
    )
    fig.savefig(f"{output_dir}/corner_plot.png")
