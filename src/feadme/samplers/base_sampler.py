from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable

import arviz as az
import flax.struct
import jax
import jax.numpy as jnp
import loguru
import pandas as pd
import xarray as xr
from jax.typing import ArrayLike
from numpyro.handlers import reparam
from numpyro.infer.mcmc import MCMCKernel, MCMC
from numpyro.infer.reparam import NeuTraReparam
from numpyro.infer.util import Predictive

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
    def _data(self):
        return self._config.data

    @property
    def wave(self) -> ArrayLike:
        return self._config.data.masked_wave

    @property
    def flux(self) -> ArrayLike:
        return self._config.data.masked_flux

    @property
    def flux_err(self) -> ArrayLike:
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
    def get_posterior_samples(self, mcmc: MCMC) -> ArrayLike:
        pass

    def run(self):
        """
        Run the sampler, which includes sampling, writing results, and plotting.
        """
        self.sample()

    def _compose_inference_data(
        self,
        mcmc: MCMC,
        posterior_samples: dict[str, ArrayLike],
        prior_model: Callable = None,
    ) -> az.InferenceData:
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
        rng_key = jax.random.PRNGKey(0)

        # Posterior predictive
        predictive_post = Predictive(self.model, posterior_samples=posterior_samples)(
            rng_key,
            template=self.template,
            wave=self.wave,
            flux=None,
            flux_err=self.flux_err,
        )

        predictive_post.update(
            {
                k: jnp.zeros_like(v)
                for k, v in posterior_samples.items()
                if k not in predictive_post
            }
        )

        # Prior predictive
        predictive_prior = Predictive(prior_model, num_samples=5000)(
            rng_key,
            template=self.template,
            wave=self.wave,
            flux=None,
            flux_err=self.flux_err,
        )

        predictive_prior.update(
            {
                k: jnp.zeros_like(v)
                for k, v in posterior_samples.items()
                if k not in predictive_prior
            }
        )

        # Compute log-likelihood for each posterior sample
        log_likelihood = Predictive(
            self.model, posterior_samples=posterior_samples, return_sites=["total_flux"]
        )(
            rng_key,
            template=self.template,
            wave=self.wave,
            flux=self.flux,
            flux_err=self.flux_err,
        )

        idata = az.from_numpyro(
            mcmc,
            posterior_predictive=predictive_post,
            prior=predictive_prior,
            log_likelihood=log_likelihood,
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
            with pd.option_context("display.precision", 10):
                summary = az.summary(
                    self._idata,
                    stat_focus="median",
                    hdi_prob=0.68,
                    var_names=[
                        x
                        for x in self._idata.posterior.data_vars
                        if x not in self._get_ignored_vars()
                    ],
                    round_to=10,
                )

            col_stat = "hdi" if "hdi_16%" in summary.columns else "eti"

            summary["err_lo"] = summary["median"] - summary[f"{col_stat}_16%"]
            summary["err_hi"] = summary[f"{col_stat}_84%"] - summary["median"]

            # Extract posterior samples of the circular parameters
            circ_vars = self._get_circular_vars()

            if len(circ_vars) > 0:
                posterior = az.extract(
                    self._idata, var_names=circ_vars, group="posterior", combined=True
                )

                if isinstance(posterior, xr.DataArray):
                    posterior = posterior.to_dataset(name=posterior.name)

                for var in circ_vars:
                    theta = posterior[var].values

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

    def _get_divergences(self) -> tuple[int, float]:
        # Access sample statistics
        sample_stats = self._idata.sample_stats

        # Check divergences
        divergences = sample_stats["diverging"]
        num_divergences = divergences.sum().values
        divergence_rate = 100 * num_divergences / divergences.size

        return num_divergences, divergence_rate

    def _get_ignored_vars(self, include_shared=False) -> list[str]:
        """
        Get a list of variables to ignore in the pair plot and summary.
        """
        fixed_vars = [
            f"{prof.name}_{param.name}"
            for prof in self.template.disk_profiles + self.template.line_profiles
            for param in prof.fixed
        ]

        if self.template.redshift.fixed:
            fixed_vars.append(self.template.redshift.name)

        if self.template.white_noise.fixed:
            fixed_vars.append(self.template.white_noise.name)

        orphaned_vars = [
            f"{prof.name}_{param.name}"
            for prof in self.template.disk_profiles + self.template.line_profiles
            for param in prof.shared
            if f"{param.shared}_{param.name}" in fixed_vars
        ]

        shared_vars = [
            f"{prof.name}_{param.name}"
            for prof in self.template.disk_profiles + self.template.line_profiles
            for param in prof.shared
            if f"{param.shared}_{param.name}" not in fixed_vars + orphaned_vars
        ]

        ignored_vars = [
            x
            for x in self._idata.posterior.data_vars
            if x.endswith("_flux")
            or x.endswith("_base")
            or x.endswith("_unwrapped")
            or "auto_shared_latent" in x
            or x in fixed_vars + orphaned_vars
        ]

        if include_shared:
            ignored_vars += [
                x for x in self._idata.posterior.data_vars if x in shared_vars
            ]

        return ignored_vars

    def _get_circular_vars(self) -> list[str]:
        """
        Get a list of circular variables in the posterior samples.
        """
        circ_vars = [
            f"{prof.name}_{param.name}"
            for prof in self.template.disk_profiles + self.template.line_profiles
            for param in prof.independent
            if param.circular
        ]

        return [x for x in self._idata.posterior.data_vars if x in circ_vars]

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
            ignored_vars=self._get_ignored_vars(include_shared=True),
            log_vars=[
                x.name for x in self.template.all_parameters if "log" in x.distribution
            ],
        )
        plot_corner_priors(
            self._idata,
            self._config.output_path,
            ignored_vars=self._get_ignored_vars(include_shared=True),
            log_vars=[
                x.name for x in self.template.all_parameters if "log" in x.distribution
            ],
        )
