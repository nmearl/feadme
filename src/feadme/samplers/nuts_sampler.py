from typing import Dict, Optional

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer.util import Predictive

from .base_sampler import Sampler


class NUTSSampler(Sampler):
    """NUTS (No-U-Turn Sampler) implementation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mcmc = None

    def flat_posterior_samples(self) -> Dict[str, np.ndarray]:
        """Return flattened posterior samples."""
        if self._idata is None:
            return {}

        return {
            var: self._idata.posterior[var].stack(sample=("chain", "draw")).values
            for var in self._idata.posterior.data_vars
            if "_flux" not in var
        }

    def _create_sampler(
        self, init_strategy=init_to_median, dense_mass=True, **kwargs
    ) -> MCMC:
        """Create NUTS sampler."""
        chain_method = (
            "vectorized"
            if jax.local_device_count() == 1 and self._config.num_chains > 1
            else "parallel"
        )

        logger.info(
            f"Creating NUTS sampler for {self._label} using '{chain_method}' method "
            f"with {jax.local_device_count()} devices and {self._config.num_chains} chains."
        )

        nuts_kernel = NUTS(
            self._model,
            init_strategy=init_strategy,
            dense_mass=dense_mass,
            target_accept_prob=0.9,
            **kwargs,
        )

        mcmc = MCMC(
            nuts_kernel,
            num_warmup=self._config.num_warmup,
            num_samples=self._config.num_samples,
            num_chains=self._config.num_chains,
            chain_method=chain_method,
            progress_bar=self._config.progress_bar,
        )

        self._mcmc = mcmc
        return mcmc

    def _run_sampling(self, mcmc: MCMC, **kwargs) -> MCMC:
        """Run NUTS sampling."""
        mcmc.run(
            self._rng_keys,
            template=self._template,
            wave=jnp.asarray(self._data.wave),
            flux=jnp.asarray(self._data.flux),
            flux_err=jnp.asarray(self._data.flux_err),
            use_quad=self._config.use_quad,
        )
        return mcmc

    def _get_current_sampler(self) -> Optional[MCMC]:
        """Get current MCMC sampler for continuation."""
        if self._mcmc is not None:
            self._mcmc.post_warmup_state = self._mcmc.last_state
            self._rng_keys = self._mcmc.post_warmup_state.rng_key
        return self._mcmc

    def _compose_inference_data(self, mcmc: MCMC) -> az.InferenceData:
        """Create ArviZ InferenceData from NUTS MCMC results."""
        posterior_samples = {
            k: v for k, v in mcmc.get_samples().items() if "_flux" not in k
        }

        rng_key = jax.random.PRNGKey(0)

        # Posterior predictive
        predictive_post = Predictive(self._model, posterior_samples=posterior_samples)(
            rng_key,
            wave=self._data.wave,
            flux=None,
            flux_err=self._data.flux_err,
            template=self._template,
        )

        # Prior predictive
        predictive_prior = Predictive(self._model, num_samples=1000)(
            rng_key,
            wave=self._data.wave,
            flux=None,
            flux_err=self._data.flux_err,
            template=self._template,
        )

        return az.from_numpyro(
            mcmc,
            posterior_predictive=predictive_post,
            prior=predictive_prior,
        )
