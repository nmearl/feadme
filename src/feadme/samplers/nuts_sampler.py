import time
from typing import Callable

import arviz as az
import jax.numpy as jnp
import jax.random as random
import loguru
import matplotlib.pyplot as plt
import optax
from jax.typing import ArrayLike
import jax
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer import init_to_median, init_to_value
from numpyro.infer.autoguide import (
    AutoBNAFNormal,
    AutoContinuous,
    AutoMultivariateNormal,
)
from numpyro.infer.reparam import NeuTraReparam
from typing import cast

from .base_sampler import BaseSampler
from ..compose import evaluate_model
from ..models.lsq import lsq_model_fitter
from ..parser import NUTSSamplerSettings

logger = loguru.logger.opt(colors=True)


class NUTSSampler(BaseSampler):
    @property
    def sampler_settings(self) -> NUTSSamplerSettings:
        return cast(NUTSSamplerSettings, self._config.sampler_settings)

    def get_posterior_samples(
        self, mcmc: MCMC, neutra: NeuTraReparam = None
    ) -> dict[str, ArrayLike]:
        if self.sampler_settings.neutra and neutra is not None:
            # Get samples WITH chain structure preserved
            zs = mcmc.get_samples(group_by_chain=True)["auto_shared_latent"]
            # zs shape: (num_chains, num_samples, latent_dim)

            # Transform each chain separately
            def transform_chain(z_chain):
                return neutra.transform_sample(z_chain)

            # vmap over chains
            posterior_samples = jax.vmap(transform_chain)(zs)

            # Flatten back: each param should be (num_chains, num_samples, ...)
            return {k: v for k, v in posterior_samples.items()}
        else:
            return mcmc.get_samples(group_by_chain=True)

    def _compose_inference_data(
        self,
        mcmc: MCMC,
        posterior_samples: dict[str, ArrayLike],
        prior_model: Callable = None,
    ) -> az.InferenceData:
        # Flatten for predictive (it expects combined samples)
        flat_samples = {
            k: v.reshape(-1, *v.shape[2:]) if v.ndim > 2 else v.reshape(-1)
            for k, v in posterior_samples.items()
        }

        predictive_post, predictive_prior, log_likelihood = self._inference_data(
            flat_samples, prior_model
        )

        if self.sampler_settings.neutra:
            # Reshape predictive outputs back to (chains, draws, ...)
            num_chains = posterior_samples[list(posterior_samples.keys())[0]].shape[0]
            num_draws = posterior_samples[list(posterior_samples.keys())[0]].shape[1]

            predictive_post_reshaped = {
                k: v.reshape(num_chains, num_draws, -1)
                for k, v in predictive_post.items()
                if k in ["total_flux", "disk_flux", "line_flux"]
            }

            log_likelihood_reshaped = {
                k: v.reshape(num_chains, num_draws, -1)
                for k, v in log_likelihood.items()
            }

            idata = az.from_dict(
                posterior=posterior_samples,
                posterior_predictive=predictive_post_reshaped,
                prior=predictive_prior,
                log_likelihood=log_likelihood_reshaped,
            )
        else:
            idata = az.from_numpyro(...)

        return idata

    def _plot_guide_samples(self, guide: AutoContinuous, svi_result):
        # Sample from the guide to check if it matches LSQ
        guide_samples = guide.sample_posterior(
            random.PRNGKey(1), svi_result.params, sample_shape=(1000,)
        )

        line_flux = jnp.median(guide_samples["line_flux"], axis=0)
        line_flux = jnp.where(jnp.isfinite(line_flux), line_flux, 0.0)
        disk_flux = jnp.median(guide_samples["disk_flux"], axis=0)
        disk_flux = jnp.where(jnp.isfinite(disk_flux), disk_flux, 0.0)

        param_mods = {
            k: jnp.median(v)
            for k, v in guide_samples.items()
            if "_flux" not in k and "_base" not in k
        }
        q_tot_flux, q_disk_flux, q_line_flux = evaluate_model(
            self.template, self.wave / (1 + param_mods.get("redshift", 0.0)), param_mods
        )

        fig, ax = plt.subplots()
        tot_err = self.flux_err * jnp.exp(param_mods["white_noise"])

        ax.errorbar(
            self.wave, self.flux, yerr=tot_err, fmt="o", color="grey", alpha=0.5
        )
        ax.plot(self.wave, line_flux, label="Line Flux Median")
        ax.plot(self.wave, disk_flux, label="Disk Flux Median")
        ax.plot(self.wave, line_flux + disk_flux, label="Total Flux Median")

        ax.plot(self.wave, q_line_flux, linestyle="--")
        ax.plot(self.wave, q_disk_flux, linestyle="--")
        ax.plot(self.wave, q_tot_flux, linestyle="--")

        ax.legend()
        fig.savefig(f"{self._config.output_path}/guide_model_fit.png")
        plt.close(fig)

    def _initialize_svi(self) -> tuple:
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, svi_key, mcmc_key = random.split(rng_key, 3)

        # Generate starting location from LSQ fit
        starters = lsq_model_fitter(
            self.template,
            self._data,
            out_dir=f"{self._config.output_path}",
        )[0]

        # Define the guide
        # guide = AutoBNAFNormal(
        #     self.model,
        #     hidden_factors=[4],
        #     num_flows=1,
        #     init_loc_fn=init_to_median(num_samples=1000),
        # )
        guide = AutoMultivariateNormal(
            self.model,
            init_loc_fn=init_to_value(values=starters),
        )

        # Define the optimization strategy
        schedule = optax.exponential_decay(0.001, 20_000, 0.3)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=schedule),  # Clip gradients
        )

        # Setup and run the SVI
        svi = SVI(
            self.model,
            guide,
            optimizer,
            Trace_ELBO(),
        )
        svi_result = svi.run(
            svi_key,
            25_000,
            template=self.template,
            wave=self.wave,
            flux=self.flux,
            flux_err=self.flux_err,
            progress_bar=self.sampler_settings.progress_bar,
            stable_update=True,
        )

        # Plot guide samples
        self._plot_guide_samples(guide, svi_result)

        # Convergence check
        recent_losses = svi_result.losses[-1000:]
        relative_std = jnp.nanstd(recent_losses) / jnp.abs(jnp.nanmean(recent_losses))

        if relative_std > 0.01:
            logger.warning(
                f"SVI may not have converged! Relative std: {relative_std:.4f}"
            )
            # Could add logic to extend SVI or use simpler guide
        elif jnp.any(jnp.isnan(recent_losses)):
            logger.error(
                f"SVI encountered NaNs in losses. Relative std: {relative_std:.4f}"
            )
            # return self.model, init_to_median(num_samples=1000), None
        else:
            logger.info(
                f"SVI converged successfully. Final loss: {svi_result.losses[-1]:.4f}"
            )

        # Initialize from VI posterior
        init_key, mcmc_key = random.split(mcmc_key)
        chain_init_params = guide.sample_posterior(init_key, svi_result.params)

        init_strategy = init_to_value(
            values={k: jnp.median(v) for k, v in chain_init_params.items()}
        )

        return self.model, init_strategy, None, guide, svi_result

    def _initialize_neutra(self) -> tuple:
        _, init_strategy, _, guide, svi_result = self._initialize_svi()

        neutra = NeuTraReparam(guide, svi_result.params)
        neutra_model = neutra.reparam(self.model)

        # The guide learns to map standard normal -> parameters
        z0 = jnp.zeros((guide.latent_dim,))

        init_strategy = init_to_value(values={"auto_shared_latent": z0})

        return neutra_model, init_strategy, neutra

    def _initialize_basic(self) -> tuple:
        starters = lsq_model_fitter(
            self.template,
            self._data,
            out_dir=f"{self._config.output_path}",
        )[0]

        starters = {k: float(v) for k, v in starters.items()}

        return (
            self.model,
            # init_to_median(num_samples=1000),
            init_to_value(values=starters),
            None,
        )

    def sample(self):
        """
        Run the NUTS sampler to perform MCMC sampling.
        """
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, svi_key, mcmc_key = random.split(rng_key, 3)

        # Setup model and initialization strategy
        if self.sampler_settings.neutra:
            model, init_strategy, neutra = self._initialize_neutra()
        elif self.sampler_settings.prefit:
            model, init_strategy, neutra, _, _ = self._initialize_svi()
        else:
            model, init_strategy, neutra = self._initialize_basic()

        # Construct NUTS kernel
        kernel = NUTS(
            model,
            init_strategy=init_strategy,
            target_accept_prob=self.sampler_settings.target_accept_prob,
            max_tree_depth=self.sampler_settings.max_tree_depth,
            dense_mass=self.sampler_settings.dense_mass,
            find_heuristic_step_size=True,
        )

        # Setup and run MCMC sampler
        mcmc = MCMC(
            kernel,
            num_warmup=self.sampler_settings.num_warmup,
            num_samples=self.sampler_settings.num_samples,
            num_chains=self.sampler_settings.num_chains,
            chain_method=self.sampler_settings.chain_method,
            progress_bar=self.sampler_settings.progress_bar,
        )

        mcmc.run(
            rng_key,
            template=self.template,
            wave=self.wave,
            flux=self.flux,
            flux_err=self.flux_err,
            extra_fields=("num_steps",),
        )

        # Get posterior samples
        posterior_samples = self.get_posterior_samples(mcmc, neutra)

        self._idata = self._compose_inference_data(
            mcmc, posterior_samples, prior_model=self._prior_model
        )

        # Report treedepth statistics
        def report_treedepth(mcmc, nuts_kernel):
            info = mcmc.get_extra_fields()
            num_steps = info["num_steps"]
            tree_depth = jnp.log2(num_steps).astype(int) + 1
            max_depth = nuts_kernel._max_tree_depth
            frac = (tree_depth >= max_depth).mean()
            logger.info(
                f"Treedepth hits: {100*frac:.2f}% at depth {max_depth} "
                f"({jnp.min(tree_depth)}, {jnp.max(tree_depth)})"
            )

        report_treedepth(mcmc, kernel)
