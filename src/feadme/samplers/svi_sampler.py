import time

import arviz as az
import jax
import jax.numpy as jnp
import loguru
import matplotlib.pyplot as plt
import numpy as np
import optax
from numpyro.infer import SVI, Trace_ELBO, init_to_median, init_to_value
from numpyro.infer.autoguide import (
    AutoBNAFNormal,
    AutoLaplaceApproximation,
    AutoMultivariateNormal,
)
from numpyro.infer.util import log_likelihood
from typing import cast

from .base_sampler import BaseSampler
from ..models.lsq import lsq_model_fitter
from ..parser import SVISamplerSettings

logger = loguru.logger.opt(colors=True)


class SVISampler(BaseSampler):
    """
    Variational inference sampler using AutoBNAFNormal guide.
    """

    def __init__(self, model, config, prior_model=None):
        super().__init__(model, config, prior_model)
        self._svi = None
        self._svi_result = None
        self._guide = None

    @property
    def sampler_settings(self) -> SVISamplerSettings:
        return cast(SVISamplerSettings, self._config.sampler_settings)

    def sample(self):
        """Run SVI to approximate posterior"""
        num_steps = self.sampler_settings.num_steps
        learning_rate = self.sampler_settings.learning_rate
        decay_rate = self.sampler_settings.decay_rate
        decay_steps = self.sampler_settings.decay_steps
        num_flows = self.sampler_settings.num_flows
        hidden_factors = self.sampler_settings.hidden_factors

        logger.info(
            f"Starting SVI with {num_steps} steps, "
            f"{num_flows} flows, hidden_factors={hidden_factors}"
        )

        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, svi_key = jax.random.split(rng_key)

        # Generate initial parameter estimates using least squares
        # starters = lsq_model_fitter(
        #     self.template,
        #     self._data,
        # )[0]
        #
        # from pprint import pprint
        #
        # pprint(starters)

        # Create guide
        self._guide = AutoBNAFNormal(
            self.model,
            hidden_factors=hidden_factors,
            num_flows=num_flows,
            # init_loc_fn=init_to_value(values=starters),
            init_loc_fn=init_to_median(num_samples=1000),
        )

        # Setup optimizer with learning rate schedule
        schedule = optax.exponential_decay(learning_rate, decay_steps, decay_rate)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=schedule),
        )

        # Create SVI object
        self._svi = SVI(
            self.model,
            self._guide,
            optimizer,
            Trace_ELBO(),
        )

        # Run SVI
        self._svi_result = self._svi.run(
            svi_key,
            num_steps,
            template=self.template,
            wave=self.wave,
            flux=self.flux,
            flux_err=self.flux_err,
            progress_bar=self.sampler_settings.progress_bar,
            stable_update=True,
        )

        check_count = 0

        # Check convergence
        while not self._check_convergence() and check_count < 2:
            check_count += 1
            logger.info(
                f"Re-running SVI to improve convergence (attempt {check_count})..."
            )

            self._svi_result = self._svi.run(
                svi_key,
                num_steps,
                template=self.template,
                wave=self.wave,
                flux=self.flux,
                flux_err=self.flux_err,
                progress_bar=self.sampler_settings.progress_bar,
                stable_update=True,
            )

        # Plot convergence
        self._plot_convergence()

        # Sample from guide
        posterior_samples = self.get_posterior_samples()

        # Convert to arviz format
        self._idata = self._convert_to_arviz(posterior_samples)

        logger.info("SVI completed successfully")

    def _check_convergence(self) -> bool:
        """Check if SVI has converged"""
        recent_losses = self._svi_result.losses[-1000:]
        relative_std = jnp.nanstd(recent_losses) / jnp.abs(jnp.nanmean(recent_losses))

        if relative_std > 0.01:
            logger.warning(
                f"SVI may not have converged! Relative std: {relative_std:.4f}. "
                f"Consider increasing num_steps or adjusting learning rate."
            )
        elif jnp.any(jnp.isnan(recent_losses)):
            logger.error("SVI encountered NaNs in losses. Training failed.")
            # raise ValueError("SVI training produced NaN losses")
        else:
            logger.info(
                f"SVI converged. Final loss: {self._svi_result.losses[-1]:.4f}, "
                f"Relative std: {relative_std:.6f}"
            )

        return not (jnp.any(jnp.isnan(recent_losses)))

    def _plot_convergence(self):
        """Plot SVI loss convergence"""
        fig, ax = plt.subplots(figsize=(10, 4))

        losses = self._svi_result.losses[1000:]
        ax.plot(losses, alpha=0.6, label="Loss")

        # Plot moving average
        window = min(1000, len(losses) // 10)
        if window > 1:
            moving_avg = jnp.convolve(losses, jnp.ones(window) / window, mode="valid")
            ax.plot(
                jnp.arange(window - 1, len(losses)),
                moving_avg,
                color="red",
                linewidth=2,
                label=f"Moving avg (window={window})",
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("ELBO Loss")
        ax.set_title("SVI Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(f"{self._config.output_path}/svi_convergence.png", dpi=150)
        plt.close(fig)

    def get_posterior_samples(self) -> dict:
        """Get posterior samples from the guide"""
        if self._guide is None or self._svi_result is None:
            raise ValueError("No results available. Run sample() first.")

        num_posterior_samples = self.sampler_settings.num_posterior_samples

        logger.info(f"Sampling {num_posterior_samples} from variational posterior...")

        return self._guide.sample_posterior(
            jax.random.PRNGKey(42),
            self._svi_result.params,
            sample_shape=(num_posterior_samples,),
        )

    def _convert_to_arviz(self, posterior_samples):
        """Convert SVI posterior samples to arviz InferenceData"""
        logger.info("Constructing inference data...")
        from numpyro.infer.util import Predictive

        n_samples = len(list(posterior_samples.values())[0])
        n_chains = 4
        samples_per_chain = n_samples // n_chains
        rng_key = jax.random.PRNGKey(42)
        parallel_predictive = jax.extend.backend.get_backend().platform == "gpu"

        # Reshape all samples to (chain, draw) or (chain, draw, dim)
        posterior_dict = {}
        for k, v in posterior_samples.items():
            if k.endswith("_base"):
                continue

            arr = np.array(v)[: samples_per_chain * n_chains]

            if arr.ndim == 1:
                posterior_dict[k] = arr.reshape(n_chains, samples_per_chain)
            elif arr.ndim == 2:
                posterior_dict[k] = arr.reshape(
                    n_chains, samples_per_chain, arr.shape[-1]
                )
            else:
                logger.warning(
                    f"Skipping parameter {k} with unexpected shape {arr.shape}"
                )

        # Posterior predictive using Predictive (includes observation noise)
        predictive_post = Predictive(
            self.model,
            posterior_samples=posterior_samples,
            parallel=parallel_predictive,
        )(
            rng_key,
            template=self.template,
            wave=self.wave,
            flux=None,
            flux_err=self.flux_err,
        )

        posterior_predictive_dict = {}
        for k in ["total_flux", "disk_flux", "line_flux"]:
            arr = np.array(predictive_post[k])[: samples_per_chain * n_chains]
            posterior_predictive_dict[k] = arr.reshape(n_chains, samples_per_chain, -1)

        # Compute log-likelihood for each posterior sample
        log_lik = log_likelihood(
            self.model,
            posterior_samples,
            wave=self.wave,
            flux=self.flux,
            flux_err=self.flux_err,
            template=self.template,
        )

        # Reshape to (chains, draws, wavelength)
        arr = np.array(log_lik["total_flux"])[: samples_per_chain * n_chains]
        log_lik_dict = {"total_flux": arr.reshape(n_chains, samples_per_chain, -1)}

        # Prior samples
        prior_predictive = Predictive(
            self._prior_model, num_samples=2000, parallel=parallel_predictive
        )(
            jax.random.PRNGKey(0),
            template=self.template,
            wave=self.wave,
            flux=None,
            flux_err=self.flux_err,
        )

        prior_predictive = {
            k: v
            for k, v in prior_predictive.items()
            if not k.endswith("_base") and not k.endswith("_flux")
        }

        # Filter valid prior samples
        valid_indices = [
            i
            for i in range(2000)
            if all(np.all(np.isfinite(v[i])) for v in prior_predictive.values())
        ]

        prior_dict = {
            k: np.array([v[i] for i in valid_indices])
            for k, v in prior_predictive.items()
        }

        logger.info(f"Generated {len(valid_indices)}/2000 valid prior samples")

        # Reshape priors to (chains, draws) format
        n_prior_valid = len(valid_indices)
        prior_chains = 4
        prior_draws = n_prior_valid // prior_chains

        for k, v in prior_dict.items():
            arr = np.array(v)[: prior_draws * prior_chains]
            if arr.ndim == 1:
                prior_dict[k] = arr.reshape(prior_chains, prior_draws)
            elif arr.ndim == 2:
                prior_dict[k] = arr.reshape(prior_chains, prior_draws, arr.shape[-1])

        # Create InferenceData
        idata = az.from_dict(
            posterior=posterior_dict,
            posterior_predictive=posterior_predictive_dict,
            prior=prior_dict,
            log_likelihood=log_lik_dict,
        )

        return idata
