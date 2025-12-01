import time
from typing import Callable

import arviz as az
import jax.numpy as jnp
import jax.random as random
import loguru
import matplotlib.pyplot as plt
import optax
from jax.typing import ArrayLike
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer import init_to_median, init_to_value
from numpyro.infer.autoguide import (
    AutoBNAFNormal,
)
from numpyro.infer.reparam import NeuTraReparam

from .base_sampler import BaseSampler
from ..compose import evaluate_model
from ..models.lsq import lsq_model_fitter

logger = loguru.logger.opt(colors=True)


class NUTSSampler(BaseSampler):
    def get_posterior_samples(
        self, mcmc: MCMC, neutra: NeuTraReparam = None
    ) -> dict[str, ArrayLike]:
        if self.sampler.use_neutra and neutra is not None:
            zs = mcmc.get_samples()["auto_shared_latent"]
            posterior_samples = neutra.transform_sample(zs)
        else:
            posterior_samples = mcmc.get_samples()

        return posterior_samples

    def _compose_inference_data(
        self,
        mcmc: MCMC,
        posterior_samples: dict[str, ArrayLike],
        prior_model: Callable = None,
    ) -> az.InferenceData:
        predictive_post, predictive_prior, log_likelihood = self._inference_data(
            posterior_samples, prior_model
        )

        idata = az.from_numpyro(
            mcmc,
            posterior_predictive=predictive_post,
            prior=predictive_prior,
            log_likelihood=log_likelihood,
        )

        return idata

    def _initialize_svi(self) -> tuple:
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, svi_key, mcmc_key = random.split(rng_key, 3)

        starters, _, _, _ = lsq_model_fitter(
            self.template,
            self._data,
            out_dir=f"{self._config.output_path}",
        )

        # guide = AutoLaplaceApproximation(self.model, init_loc_fn=init_to_median())
        guide = AutoBNAFNormal(
            self.model,
            hidden_factors=[4],
            num_flows=1,
            init_loc_fn=init_to_median(num_samples=1000),
        )

        # optimizer = optim.Adam(step_size=1e-3)
        schedule = optax.exponential_decay(0.001, 10_000, 0.1)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=schedule),  # Clip gradients
        )

        svi = SVI(
            self.model,
            guide,
            optimizer,
            Trace_ELBO(),
        )
        svi_result = svi.run(
            svi_key,
            20_000,
            template=self.template,
            wave=self.wave,
            flux=self.flux,
            flux_err=self.flux_err,
            progress_bar=self.sampler.progress_bar,
            stable_update=True,
        )

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
        tot_err = jnp.sqrt(
            self.flux_err**2 + self.flux**2 * jnp.exp(2 * param_mods["white_noise"])
        )
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

        init_strategy = init_to_value(values=chain_init_params)

        return self.model, init_strategy, None, guide, svi_result

    def _initialize_neutra(self) -> tuple:
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, svi_key, mcmc_key = random.split(rng_key, 3)

        _, _, _, guide, svi_result = self._initialize_svi()

        neutra = NeuTraReparam(guide, svi_result.params)
        neutra_model = neutra.reparam(self.model)

        # Initialize from VI posterior
        init_key, mcmc_key = random.split(mcmc_key)

        if self.sampler.num_chains > 1:
            # Sample one set of parameters per chain
            init_params = guide.sample_posterior(
                init_key, svi_result.params, sample_shape=(self.sampler.num_chains,)
            )
            # init_params now has shape (num_chains, ...) for each parameter
            chain_init_params = init_params
        else:
            # Single chain - sample one set of parameters
            chain_init_params = guide.sample_posterior(init_key, svi_result.params)

        init_strategy = init_to_value(values=chain_init_params)

        return neutra_model, init_strategy, neutra

    def _initialize_basic(self):
        starters, _, _, _ = lsq_model_fitter(
            self.template,
            self._data,
            out_dir=f"{self._config.output_path}",
        )
        # init_values = {k: v[0] for k, v in starters.items()}
        # init_values = lsq_to_base_space(starters, self.template)

        model, init_strategy, neutra = (
            self.model,
            init_to_median(num_samples=1000),
            None,
        )

        return model, init_strategy, neutra

    def sample(self):
        """
        Run the NUTS sampler to perform MCMC sampling.
        """
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, svi_key, mcmc_key = random.split(rng_key, 3)

        if self.sampler.use_neutra:
            # model, init_strategy, neutra = self._initialize_neutra()
            model, init_strategy, neutra, _, _ = self._initialize_svi()
        else:
            model, init_strategy, neutra = self._initialize_basic()

        kernel = NUTS(
            model,
            init_strategy=init_strategy,
            target_accept_prob=self.sampler.target_accept_prob,
            max_tree_depth=self.sampler.max_tree_depth,
            dense_mass=self.sampler.dense_mass,
            find_heuristic_step_size=True,
        )

        mcmc = MCMC(
            kernel,
            num_warmup=self.sampler.num_warmup,
            num_samples=self.sampler.num_samples,
            num_chains=self.sampler.num_chains,
            chain_method=self.sampler.chain_method,
            progress_bar=self.sampler.progress_bar,
        )

        mcmc.run(
            rng_key,
            template=self.template,
            wave=self.wave,
            flux=self.flux,
            flux_err=self.flux_err,
            extra_fields=("num_steps",),
        )

        posterior_samples = self.get_posterior_samples(mcmc, neutra)

        self._idata = self._compose_inference_data(
            mcmc, posterior_samples, prior_model=self._prior_model
        )

        def report_treedepth(mcmc, nuts_kernel):
            info = mcmc.get_extra_fields()
            num_steps = info["num_steps"]
            tree_depth = jnp.log2(num_steps).astype(int) + 1
            max_depth = nuts_kernel._max_tree_depth
            frac = (tree_depth >= max_depth).mean()
            logger.info(
                f"Treedepth hits: {100*frac:.2f}% at depth {max_depth} ({jnp.min(tree_depth)}, {jnp.max(tree_depth)})"
            )

        report_treedepth(mcmc, kernel)
