import arviz as az
from .core.parser import Config
import corner
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from .core.evaluators import evaluate_model


class Plotter:
    def __init__(self, config: Config, idata: az.InferenceData, summary: pd.DataFrame):
        self._config = config
        self._idata = idata
        self._summary = summary

    def _representative_param_mods(self) -> tuple[dict, float] | tuple[None, None]:
        """
        Return one representative posterior sample, preferring the highest
        summed log-likelihood draw when available.
        """
        if not hasattr(self._idata, "posterior"):
            return None, None

        if hasattr(self._idata, "log_likelihood") and "total_flux" in self._idata.log_likelihood:
            log_lik = self._idata.log_likelihood["total_flux"].values
            total_log_lik = np.sum(log_lik, axis=-1)
            best_idx = np.unravel_index(np.argmax(total_log_lik), total_log_lik.shape)
            chain_idx, draw_idx = map(int, best_idx)
        else:
            chain_idx = 0
            draw_idx = 0

        posterior = self._idata.posterior
        param_mods = {}

        for param_ref in self._config.template.iter_all:
            if param_ref.name == "log_frac_noise":
                continue

            if param_ref.name in posterior.data_vars:
                param_mods[param_ref.name] = float(
                    posterior[param_ref.name].values[chain_idx, draw_idx]
                )
            elif param_ref.param.fixed:
                param_mods[param_ref.name] = float(param_ref.param.value)

        redshift = param_mods.pop("redshift", None)
        return param_mods, redshift

    def plot_model_fit(self):
        """
        Plot the model fit using the posterior distributions of the disk and line fluxes.
        """
        param_mods = self._summary["value"].to_dict()
        redshift = param_mods.pop("redshift")
        log_frac_noise = param_mods.pop("log_frac_noise")
        rest_wave = self._config.data.masked_wave / (1 + redshift)
        summary_total_flux, _, _ = evaluate_model(
            self._config.template,
            self._config.data.masked_wave,
            param_mods,
            redshift,
        )
        total_error = np.sqrt(
            self._config.data.masked_flux_err**2
            + summary_total_flux**2 * np.exp(2.0 * log_frac_noise)
        )

        fig, ax = plt.subplots(layout="constrained")

        ax.errorbar(
            rest_wave,
            self._config.data.masked_flux,
            yerr=total_error,
            fmt="o",
            color="grey",
            zorder=-10,
            alpha=0.25,
        )

        # Plot the posterior distributions for disk and line flux
        for var in ["disk_flux", "line_flux"]:
            var_name = " ".join([x.capitalize() for x in var.split("_")])
            var_dist = self._idata.posterior_predictive[var].mean(dim=("chain",)).values
            median = np.percentile(var_dist, 50, axis=0)
            ax.plot(rest_wave, median, label=f"Sampled {var_name}")

        obs_dist = (
            self._idata.posterior_predictive["total_flux"]
            .stack(sample=("chain", "draw"))
            .values
        )
        median = np.percentile(obs_dist, 50, axis=1)
        lower = np.percentile(obs_dist, 16, axis=1)
        upper = np.percentile(obs_dist, 84, axis=1)

        ax.plot(rest_wave, median, label="Sampled Model Fit", color="C3")
        ax.fill_between(rest_wave, lower, upper, alpha=0.5, color="C3")

        new_rest_wave = np.linspace(rest_wave[0], rest_wave[-1], 1000)
        new_obs_wave = np.linspace(
            self._config.data.masked_wave.min(),
            self._config.data.masked_wave.max(),
            1000,
        )
        tot_flux, disk_flux, line_flux = evaluate_model(
            self._config.template, new_obs_wave, param_mods, redshift
        )

        ax.plot(new_rest_wave, tot_flux, label="Reconstructed Model", linestyle="--")
        ax.plot(
            new_rest_wave, disk_flux, label="Reconstructed Disk Flux", linestyle="--"
        )
        ax.plot(
            new_rest_wave, line_flux, label="Reconstructed Line Flux", linestyle="--"
        )

        rep_param_mods, rep_redshift = self._representative_param_mods()
        if rep_param_mods is not None and rep_redshift is not None:
            rep_tot_flux, _, _ = evaluate_model(
                self._config.template, new_obs_wave, rep_param_mods, rep_redshift
            )
            ax.plot(
                new_rest_wave,
                rep_tot_flux,
                label="Representative Posterior Sample",
                linestyle=":",
                color="C1",
            )

        ax.set_ylabel("Flux [mJy]")
        ax.set_xlabel("Wavelength [AA]")
        ax.set_title(
            f"{self._config.template.name}{' ' + str(self._config.template.obs_date or '')} Model Fit"
        )
        ax.legend(fontsize=8)

        fig.savefig(f"{self._config.output_path}/model_fit.png")
        plt.close(fig)

    def plot_corner(self, include_shared=False):
        """
        Create a corner plot of the posterior distributions of the model parameters.
        """
        var_names = [
            var
            for var in self._idata.posterior.data_vars
            if var
            in self._config.template.fitted_parameter_names(
                include_shared=include_shared, include_circ=True
            )
        ]

        samples_ds = az.extract(
            self._idata, group="posterior", var_names=var_names, combined=True
        )
        samples = np.vstack([samples_ds[var].values for var in var_names]).T

        axes_scale = ["linear"] * len(var_names)

        for i, var in enumerate(var_names):
            param_ref = next(
                (x for x in self._config.template.iter_all if x.name == var), None
            )

            if param_ref is not None and "log" in param_ref.param.distribution.value:
                axes_scale[i] = "log"

        # Create the corner plot
        fig = corner.corner(
            samples,
            labels=var_names,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 14},
            plot_density=True,
            plot_contours=True,
            fill_contours=True,
            axes_scale=axes_scale,
        )

        fig.savefig(f"{self._config.output_path}/corner_plot.png")
        plt.close(fig)

    def plot_prior_corner(self, include_shared=False):
        """
        Create a corner plot of the prior distributions of the model parameters.
        """
        var_names = [
            var
            for var in self._idata.prior.data_vars
            if var
            in self._config.template.fitted_parameter_names(
                include_shared=include_shared, include_circ=True
            )
        ]

        samples_ds = az.extract(
            self._idata, group="prior", var_names=var_names, combined=True
        )
        samples = np.vstack([samples_ds[var].values for var in var_names]).T

        axes_scale = ["linear"] * len(var_names)

        for i, var in enumerate(var_names):
            param_ref = next(x for x in self._config.template.iter_all if x.name == var)

            if param_ref is not None and "log" in param_ref.param.distribution.value:
                axes_scale[i] = "log"

        # Create the corner plot
        fig = corner.corner(
            samples,
            labels=var_names,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            title_fmt=".2f",
            title_kwargs={"fontsize": 12},
            label_kwargs={"fontsize": 14},
            plot_density=True,
            plot_contours=True,
            fill_contours=True,
            axes_scale=axes_scale,
        )

        fig.savefig(f"{self._config.output_path}/prior_corner_plot.png")
        plt.close(fig)

    def plot_trace(self):
        var_names = [
            var
            for var in self._idata.posterior.data_vars
            if var
            in self._config.template.fitted_parameter_names(
                include_shared=False, include_circ=True
            )
        ]

        fig, axes = plt.subplots(
            nrows=len(var_names),
            ncols=2,
            figsize=(10, 3 * len(var_names)),
            layout="constrained",
        )

        with az.rc_context({"plot.max_subplots": 50}):
            az.plot_trace(self._idata.posterior, var_names=var_names, axes=axes)

        plt.savefig(f"{self._config.output_path}/trace_plot.png")
        plt.close(fig)
