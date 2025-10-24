import time

import arviz as az
import jax
import jax.numpy as jnp
import jax.random as random
import loguru
from ..models.lsq import lsq_model_fitter
from jax.typing import ArrayLike
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import (
    AutoBNAFNormal,
)
from numpyro.infer.mcmc import MCMC
from numpyro.infer.reparam import NeuTraReparam
from numpyro.infer.util import Predictive
from numpyro.infer.svi import SVIRunResult
from numpyro.infer.autoguide import AutoContinuous

from .base_sampler import BaseSampler

logger = loguru.logger.opt(colors=True)


class SVISampler(BaseSampler):
    def get_posterior_samples(
        self,
        guide: AutoContinuous,
        svi_result: SVIRunResult,
        rng_key: jax.random.PRNGKey,
        num_samples: int = 2000,
    ) -> dict[str, ArrayLike]:
        # Sample with flat shape first
        posterior_samples = guide.sample_posterior(
            rng_key, svi_result.params, sample_shape=(num_samples,)
        )

        return posterior_samples

    def _create_idata_from_vi(self, posterior_samples, svi_result):
        """Create InferenceData from VI samples."""
        rng_key = jax.random.PRNGKey(0)

        # Get total number of samples
        n_samples = len(next(iter(posterior_samples.values())))
        n_chains = self.sampler.num_chains
        draws_per_chain = n_samples // n_chains

        # Reshape samples to (chain, draw) format
        posterior_dict = {}
        for k, v in posterior_samples.items():
            # Trim to make divisible by n_chains, then reshape
            trimmed = v[: n_chains * draws_per_chain]
            posterior_dict[k] = trimmed.reshape(n_chains, draws_per_chain, *v.shape[1:])

        # Posterior predictive - use flat samples for Predictive
        predictive_post = Predictive(self.model, posterior_samples=posterior_samples)(
            rng_key,
            template=self.template,
            wave=self.wave,
            flux=None,
            flux_err=self.flux_err,
        )

        predictive_dict = {}
        for k, v in predictive_post.items():
            trimmed = v[: n_chains * draws_per_chain]
            predictive_dict[k] = trimmed.reshape(
                n_chains, draws_per_chain, *v.shape[1:]
            )

        # Prior samples
        prior_raw = Predictive(self.model, num_samples=n_chains * draws_per_chain)(
            rng_key,
            template=self.template,
            wave=self.wave,
            flux=None,
            flux_err=self.flux_err,
        )

        prior_dict = {}
        for k, v in prior_raw.items():
            prior_dict[k] = v.reshape(n_chains, draws_per_chain, *v.shape[1:])

        # Log likelihood - use flat samples
        log_likelihood = Predictive(
            self.model, posterior_samples=posterior_samples, return_sites=["total_flux"]
        )(
            rng_key,
            template=self.template,
            wave=self.wave,
            flux=self.flux,
            flux_err=self.flux_err,
        )

        log_likelihood_dict = {}
        for k, v in log_likelihood.items():
            trimmed = v[: n_chains * draws_per_chain]
            log_likelihood_dict[k] = trimmed.reshape(
                n_chains, draws_per_chain, *v.shape[1:]
            )

        # Check for NaN/Inf before creating InferenceData
        for name in list(prior_dict.keys()):
            samples = prior_dict[name]
            if jnp.any(jnp.isnan(samples)) or jnp.any(jnp.isinf(samples)):
                logger.warning(
                    f"Prior samples for {name} contain NaN/Inf - excluding from plot"
                )
                del prior_dict[name]

        return az.from_dict(
            posterior=posterior_dict,
            posterior_predictive=predictive_dict,
            prior=prior_dict,
            log_likelihood=log_likelihood_dict,
        )

    def sample(self):
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)

        starters, _, _, _ = lsq_model_fitter(
            self.template,
            self._data,
            out_dir=f"{self._config.output_path}",
        )
        init_values = {k: v[0] for k, v in starters.items()}

        guide = AutoBNAFNormal(
            self.model,
            hidden_factors=[16, 16, 16],
            num_flows=4,
        )
        svi = SVI(self.model, guide, optim.Adam(0.003), Trace_ELBO())
        svi_result = svi.run(
            rng_key,
            5_000,
            template=self.template,
            wave=self.wave,
            flux=self.flux,
            flux_err=self.flux_err,
            progress_bar=self.sampler.progress_bar,
        )

        # Sample flat, then reshape in _create_idata_from_vi
        num_samples = self.sampler.num_chains * 2000
        posterior_samples = self.get_posterior_samples(
            guide, svi_result, rng_key=rng_key, num_samples=num_samples
        )

        self._idata = self._create_idata_from_vi(posterior_samples, svi_result)
