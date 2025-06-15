from abc import ABC, abstractmethod
from typing import Callable

import arviz as az
import numpy as np
import pandas as pd
from scipy.stats import circmean
import flax.struct
import jax
import jax.numpy as jnp
import pandas as pd
from numpyro.infer.mcmc import MCMCKernel, MCMC
from numpyro.infer.util import Predictive
import xarray as xr
from pathlib import Path
import loguru

from ..parser import Config, Sampler, Template
from ..plotting import plot_hdi, plot_model_fit, plot_corner, plot_corner_priors
from ..utils import parse_circular_parameters

logger = loguru.logger.opt(colors=True)


@flax.struct.dataclass
class SamplerResult:
    """
    A dataclass to hold the results of the sampling process.

    Attributes
    ----------
    samples : dict
        Dictionary containing the sampled parameters.
    summary : dict
        Summary statistics of the sampled parameters.
    diagnostics : dict
        Diagnostics of the sampling process.
    sampler_state : any, optional
        State of the sampler, if applicable.
    """

    samples: dict
    summary: dict
    diagnostics: dict
    sampler_state: any = None


class BaseSampler(ABC):
    def __init__(self, model: Callable, config: Config):
        """
        Base class for samplers.

        Parameters
        ----------
        model : Callable
            The model to sample from.
        config : Config
            Configuration object containing sampler settings.
        """
        self._model = model
        self._config = config
        self._idata = None
        self._summary = None

    @property
    def model(self):
        return self._model

    @property
    def wave(self) -> jnp.ndarray:
        return self._config.data.masked_wave

    @property
    def flux(self) -> jnp.ndarray:
        return self._config.data.masked_flux

    @property
    def flux_err(self) -> jnp.ndarray:
        return self._config.data.masked_flux_err

    @property
    def template(self) -> Template:
        return self._config.template

    @property
    def sampler(self) -> Sampler:
        return self._config.sampler

    @abstractmethod
    def sample(self):
        pass

    @abstractmethod
    def get_kernel(self) -> MCMCKernel:
        pass

    def run(self):
        """
        Run the sampler, which includes sampling, writing results, and plotting.
        """
        self.sample()
        self.write_results()
        self.plot_results()

    def _compose_inference_data(self, mcmc: MCMC) -> az.InferenceData:
        """
        Create an ArviZ `InferenceData` object from a NumPyro MCMC run.
        Includes posterior, posterior predictive, and prior samples.

        Parameters
        ----------
        mcmc : MCMC
            The MCMC object containing the sampling results.

        Returns
        -------
        az.InferenceData
            An ArviZ InferenceData object containing the posterior, posterior predictive,
            and prior samples.
        """
        posterior_samples = mcmc.get_samples()

        rng_key = jax.random.PRNGKey(0)

        predictive_post = Predictive(
            self.model,
            posterior_samples=posterior_samples,
        )(
            rng_key,
            wave=self.wave,
            flux=None,
            flux_err=self.flux_err,
            # template=self.template,
        )

        predictive_prior = Predictive(
            self.model,
            num_samples=1000,
        )(
            rng_key,
            wave=self.wave,
            flux=None,
            flux_err=self.flux_err,
            # template=self.template,
        )

        def reshape(pred_dict, n_chains, n_draws):
            reshaped = {}
            for k, v in pred_dict.items():
                reshaped[k] = v.reshape((n_chains, n_draws) + v.shape[1:])
            return reshaped

        idata = az.from_numpyro(
            mcmc,
            posterior_predictive=predictive_post,
            prior=predictive_prior,
        )

        return idata

    @property
    def flat_posterior_samples(self):
        """
        Get the flat posterior samples from the inference data.
        """
        if self._idata is None:
            raise ValueError("Inference data not available. Run the sampler first.")

        return {
            var: self._idata.posterior[var].stack(sample=("chain", "draw")).values
            for var in self._idata.posterior.data_vars
        }

    @property
    def summary(self) -> pd.DataFrame:
        """
        Get the summary statistics of the posterior samples.
        """
        if self._summary is None:
            # Compute the original summary
            summary = az.summary(
                self._idata,
                stat_focus="median",
                hdi_prob=0.68,
                var_names=[
                    x
                    for x in self._idata.posterior.data_vars
                    if x not in self._get_ignored_vars()
                ],
            )

            col_stat = "hdi" if "hdi_16%" in summary.columns else "eti"

            summary["err_lo"] = summary["median"] - summary[f"{col_stat}_16%"]
            summary["err_hi"] = summary[f"{col_stat}_84%"] - summary["median"]

            # Extract posterior samples of the circular parameters
            circ_vars = list(
                set(
                    [
                        x.replace("_circ_x_base", "").replace("_circ_y_base", "")
                        for x in [
                            x for x in self._idata.posterior.data_vars if "circ" in x
                        ]
                    ]
                )
            )

            if len(circ_vars) > 0:
                posterior = az.extract(
                    self._idata, var_names=circ_vars, group="posterior", combined=True
                )

                if isinstance(posterior, xr.DataArray):
                    posterior = posterior.to_dataset(name=posterior.name)

                for var in circ_vars:
                    theta = posterior[var].values  # shape: (n_samples,)

                    theta_circ = parse_circular_parameters(theta)
                    theta_median = theta_circ["circular_median"]
                    theta_mean = theta_circ["circular_mean"]
                    theta_16 = theta_circ["percentile_16th"]
                    theta_84 = theta_circ["percentile_84th"]
                    theta_err_lo = theta_circ["err_lo"]
                    theta_err_hi = theta_circ["err_hi"]

                    # Update the summary DataFrame
                    row = summary.loc[var].copy()
                    row["mean"] = theta_mean
                    row["median"] = theta_median
                    row["err_lo"] = theta_err_lo
                    row["err_hi"] = theta_err_hi

                    # Replace the 68% HDI/ETI percentiles
                    row[f"{col_stat}_16%"] = theta_16
                    row[f"{col_stat}_84%"] = theta_84

                    # Save the corrected row back into the summary
                    summary.loc[var] = row

            self._summary = summary

        return self._summary

    def _get_ignored_vars(self) -> list[str]:
        """
        Get a list of variables to ignore in the pair plot and summary.
        """
        return [
            x
            for x in self._idata.posterior.data_vars
            if x.endswith("_flux")
            or x.endswith("_base")
            or x
            in [
                f"{prof.name}_{param.name}"
                for prof in self.template.disk_profiles + self.template.line_profiles
                for param in prof.fixed
            ]
        ]

    def write_results(self):
        """
        Write the results of the sampling to the output path specified in the config.
        """
        if self._idata is None:
            raise ValueError("Inference data not available. Run the sampler first.")

        out_path = Path(f"{self._config.output_path}/results.nc")

        if not out_path.exists():
            logger.info(
                f"Results written to <green>{self._config.output_path}/results.nc</green>."
            )
            az.to_netcdf(self._idata, str(out_path))

        self.summary.to_csv(
            f"{self._config.output_path}/summary.csv",
            index=True,
        )

    def plot_results(self):
        """
        Plot the results of the sampling, including HDI, model fit, and pair plots.
        """
        plot_hdi(
            self._idata, self.wave, self.flux, self.flux_err, self._config.output_path
        )
        plot_model_fit(
            self._idata,
            self.summary,
            self.template,
            self.wave,
            self.flux,
            self.flux_err,
            self._config.output_path,
            label=self.template.name,
        )
        plot_corner(
            self._idata,
            self._config.output_path,
            ignored_vars=self._get_ignored_vars(),
        )
        plot_corner_priors(
            self._idata,
            self._config.output_path,
            ignored_vars=self._get_ignored_vars(),
        )
