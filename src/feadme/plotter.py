import arviz as az
import xarray as xr
from .core.parser import Config
import corner
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .core.evaluators import evaluate_model


class Plotter:
    def __init__(self, config: Config, idata: xr.DataTree, summary: pd.DataFrame):
        self._config = config
        self._idata = idata
        self._summary = summary

    def _summary_param_mods(self) -> tuple[dict, float]:
        summary_param_mods = {}
        redshift = None
        for param_ref in self._config.template.iter_all:
            if param_ref.name == "log_frac_noise":
                continue
            if param_ref.name in self._summary.index and pd.notna(
                self._summary.loc[param_ref.name, "value"]
            ):
                value = float(self._summary.loc[param_ref.name, "value"])
                if param_ref.name == "redshift":
                    redshift = value
                summary_param_mods[param_ref.name] = value
            elif getattr(param_ref.param, "shared", None):
                target_name = getattr(param_ref, "target_name", None)
                if target_name in summary_param_mods:
                    summary_param_mods[param_ref.name] = float(
                        summary_param_mods[target_name]
                    )
            elif param_ref.param.fixed:
                summary_param_mods[param_ref.name] = float(param_ref.param.value)
            elif getattr(param_ref.param, "loc", None) is not None:
                summary_param_mods[param_ref.name] = float(param_ref.param.loc)
        if redshift is None:
            redshift = float(getattr(self._config.template.redshift, "value", 0.0))
        return summary_param_mods, redshift

    def plot_model_fit(self):
        """
        Plot the model fit using the posterior predictive distribution.

        The shaded band and median curve come from posterior predictive samples.
        The component curves (disk / line) are evaluated on a high-resolution
        wavelength grid using the posterior-summary reconstruction.
        """
        summary_param_mods, summary_redshift = self._summary_param_mods()

        rest_wave = self._config.data.masked_wave / (1 + summary_redshift)

        # Error bars: use the summary reconstruction to set the noise floor
        summary_total_flux, _, _ = evaluate_model(
            self._config.template,
            self._config.data.masked_wave,
            summary_param_mods,
            summary_redshift,
        )
        log_frac_noise = float(self._summary.loc["log_frac_noise", "value"])
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

        # Posterior predictive envelope (data-resolution)
        obs_dist = (
            self._idata["posterior_predictive"].dataset["total_flux"]
            .stack(sample=("chain", "draw"))
            .values
        )  # (n_wave, n_samples)
        pp_median = np.percentile(obs_dist, 50, axis=1)
        pp_lower = np.percentile(obs_dist, 16, axis=1)
        pp_upper = np.percentile(obs_dist, 84, axis=1)

        ax.plot(rest_wave, pp_median, label="Posterior Predictive Median", color="C3")
        ax.fill_between(rest_wave, pp_lower, pp_upper, alpha=0.5, color="C3")

        # High-resolution component decomposition from the summary fiducial
        new_rest_wave = np.linspace(rest_wave[0], rest_wave[-1], 1000)
        new_obs_wave = np.linspace(
            self._config.data.masked_wave.min(),
            self._config.data.masked_wave.max(),
            1000,
        )
        sum_tot, sum_disk, sum_line = evaluate_model(
            self._config.template, new_obs_wave, summary_param_mods, summary_redshift
        )

        ax.plot(new_rest_wave, sum_tot, label="Summary Total", color="C3", lw=1.6)
        ax.plot(new_rest_wave, sum_disk, label="Summary Disk", color="C1", lw=1.4)
        ax.plot(new_rest_wave, sum_line, label="Summary Lines", color="C2", lw=1.4)

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
            for var in self._idata["posterior"].dataset.data_vars
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
            for var in self._idata["prior"].dataset.data_vars
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
            for var in self._idata["posterior"].dataset.data_vars
            if var
            in self._config.template.fitted_parameter_names(
                include_shared=False, include_circ=True
            )
        ]

        with az.rc_context({"plot.max_subplots": 50}):
            pc = az.plot_trace(self._idata, var_names=var_names)

        pc.savefig(f"{self._config.output_path}/trace_plot.png")
        plt.close("all")
