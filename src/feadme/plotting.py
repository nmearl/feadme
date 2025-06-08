import jax
import matplotlib.pyplot as plt
import astropy.uncertainty as unc
import corner
import arviz as az
import numpy as np

from .compose import evaluate_disk_model
from .parser import Template

az.rcParams["plot.max_subplots"] = 200


def plot_trace(idata, output_dir, fixed_fields=None):
    fixed_fields = fixed_fields or []

    axes = az.plot_trace(
        idata,
        var_names=[
            x
            for x in idata.posterior.keys()
            if "_flux" not in x and x not in fixed_fields
        ],
        compact=True,
        backend_kwargs={"layout": "constrained"},
    )
    fig = axes.ravel()[0].figure
    fig.savefig(f"{output_dir}/trace_plot.png")
    plt.close(fig)


def plot_hdi(wave, flux, flux_err, idata, output_dir, hdi_prob=0.9):
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

    fig.savefig(f"{output_dir}/hdi_plot.png")
    plt.close(fig)


def plot_model_fit(
    wave,
    flux,
    flux_err,
    idata,
    results_summary,
    template: Template,
    output_dir,
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

    res_pars = {
        var: results_summary[results_summary["param"] == var]["value"].value[0]
        for var in results_summary["param"]
        if "_flux" not in var
    }

    for prof in template.all_profiles:
        for param in prof._fixed():
            res_pars[f"{prof.name}_{param.name}"] = param.value

    new_wave = np.linspace(wave.min(), wave.max(), num=1000)

    res_flux, res_disk_flux, res_line_flux = evaluate_disk_model(
        template, new_wave, res_pars
    )
    ax.plot(new_wave, res_flux, label="R. Model Fit", color="C3")
    ax.plot(new_wave, res_disk_flux, label="R. Disk Model", color="C4")
    ax.plot(new_wave, res_line_flux, label="R. Line Model", color="C5")

    ax.set_ylabel("Flux [mJy]")
    ax.set_xlabel("Wavelength [AA]")
    ax.set_title(f"{label} {template.mjd} Model Fit")
    ax.legend()

    fig.savefig(f"{output_dir}/model_fit.png")
    plt.close(fig)


def plot_corner(idata, output_dir, fixed_fields=None):
    fixed_fields = fixed_fields or []

    names = [
        x
        for x in idata.posterior.keys()
        if "_flux" not in x and "_base" not in x and x not in fixed_fields
    ]
    fig = corner.corner(
        idata.posterior,
        var_names=names,
        labels=names,
        quantiles=[0.16, 0.5, 0.84],
        smooth=1,
        show_titles=True,
        axes_scale=[
            "log" if "vel_width" in x or "radius" in x or "sigma" in x else "linear"
            for x in names
        ],
    )
    fig.savefig(f"{output_dir}/corner_plot.png")
    plt.close(fig)


def plot_corner_priors(idata, output_dir, fixed_fields=None):
    fixed_fields = fixed_fields or []

    names = [
        x
        for x in idata.prior.keys()
        if "_flux" not in x and "_base" not in x and x not in fixed_fields
    ]
    fig = corner.corner(
        idata.prior,
        var_names=names,
        labels=names,
        quantiles=[0.16, 0.5, 0.84],
        smooth=1,
        show_titles=True,
        axes_scale=[
            "log" if "vel_width" in x or "radius" in x or "sigma" in x else "linear"
            for x in names
        ],
    )
    fig.savefig(f"{output_dir}/corner_plot_prior.png")
    plt.close(fig)


def plot_results(
    template,
    output_dir,
    idata,
    results_summary,
    wave,
    flux,
    flux_err,
    label,
    fixed_fields=None,
):
    # Wrap each plotting function in a try-except block
    # to handle potential errors
    try:
        plot_trace(idata, output_dir)
    except Exception as e:
        print(f"Error plotting trace: {e}")

    try:
        plot_hdi(wave, flux, flux_err, idata, output_dir)
    except Exception as e:
        print(f"Error plotting HDI: {e}")

    try:
        plot_model_fit(
            wave, flux, flux_err, idata, results_summary, template, output_dir, label
        )
    except Exception as e:
        print(f"Error plotting model fit: {e}")

    try:
        plot_corner(idata, output_dir, fixed_fields=fixed_fields)
    except Exception as e:
        print(f"Error plotting corner: {e}")

    try:
        plot_corner_priors(idata, output_dir, fixed_fields=fixed_fields)
    except Exception as e:
        print(f"Error plotting corner priors: {e}")
