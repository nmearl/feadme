from abc import ABC, abstractmethod
from typing import Callable

import flax.struct
import arviz as az
import jax
from numpyro.infer.util import Predictive
from numpyro.infer.mcmc import MCMCKernel

from feadme.parser import Config, Sampler, Template
import jax.numpy as jnp

from feadme.plotting import plot_hdi, plot_model_fit


@flax.struct.dataclass
class SamplerResult:
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

    def _compose_inference_data(self, mcmc):
        """
        Create an ArviZ InferenceData object from a NumPyro MCMC run.
        Includes posterior, posterior predictive, and prior samples.
        """
        posterior_samples = {k: v for k, v in mcmc.get_samples().items()}

        rng_key = jax.random.PRNGKey(0)

        predictive_post = Predictive(
            self.model,
            posterior_samples=posterior_samples,
        )(
            rng_key,
            wave=self.wave,
            flux=None,
            flux_err=self.flux_err,
            template=self.template,
        )

        predictive_prior = Predictive(
            self.model,
            num_samples=1000,
        )(
            rng_key,
            wave=self.wave,
            flux=None,
            flux_err=self.flux_err,
            template=self.template,
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
        if self._idata is None:
            raise ValueError("Inference data not available. Run the sampler first.")

        return {
            var: self._idata.posterior[var].stack(sample=("chain", "draw")).values
            for var in self._idata.posterior.data_vars
        }

    def plot_results(self):
        plot_hdi(
            self._idata, self.wave, self.flux, self.flux_err, self._config.output_path
        )
        plot_model_fit(
            self._idata,
            self.wave,
            self.flux,
            self.flux_err,
            self._config.output_path,
            label=self.template.label,
        )
