import loguru
import time

import jax.numpy as jnp
import jax.random as random
import loguru
from jax.typing import ArrayLike
from numpyro import optim
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, HMC
from numpyro.infer import init_to_median, init_to_value
from numpyro.infer.autoguide import (
    AutoBNAFNormal,
    AutoIAFNormal,
    AutoMultivariateNormal,
    AutoLaplaceApproximation,
)
from numpyro.infer.reparam import NeuTraReparam
import optax
import matplotlib.pyplot as plt

from .base_sampler import BaseSampler
from ..models.lsq import lsq_model_fitter
from ..utils import lsq_to_base_space

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

    def _initialize_neutra(self) -> tuple:
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, svi_key, mcmc_key = random.split(rng_key, 3)

        starters, _, _, _ = lsq_model_fitter(
            self.template,
            self._data,
            out_dir=f"{self._config.output_path}",
        )
        # init_values = {k: v[0] for k, v in starters.items()}
        # init_values = lsq_to_base_space(starters, self.template)

        guide = AutoMultivariateNormal(
            self.model, init_loc_fn=init_to_median(num_samples=1000)
        )
        optimizer = optim.Adam(0.003)

        svi = SVI(self.model, guide, optimizer, Trace_ELBO())
        svi_result = svi.run(
            svi_key,
            20_000,
            template=self.template,
            wave=self.wave,
            flux=self.flux,
            flux_err=self.flux_err,
            progress_bar=self.sampler.progress_bar,
        )

        # Sample from the guide to check if it matches LSQ
        guide_samples = guide.sample_posterior(
            random.PRNGKey(1), svi_result.params, sample_shape=(1000,)
        )

        line_flux = jnp.median(guide_samples["line_flux"], axis=0)
        disk_flux = jnp.median(guide_samples["disk_flux"], axis=0)

        fig, ax = plt.subplots()
        ax.errorbar(
            self.wave, self.flux, yerr=self.flux_err, fmt="o", color="grey", alpha=0.5
        )
        ax.plot(self.wave, line_flux, label="Line Flux Median")
        ax.plot(self.wave, disk_flux, label="Disk Flux Median")
        ax.plot(self.wave, line_flux + disk_flux, label="Total Flux Median")
        ax.legend()
        fig.savefig(f"{self._config.output_path}/guide_model_fit.png")

        # Convergence check
        recent_losses = svi_result.losses[-1000:]
        relative_std = jnp.std(recent_losses) / jnp.abs(jnp.mean(recent_losses))

        if relative_std > 0.01:
            logger.warning(
                f"SVI may not have converged! Relative std: {relative_std:.4f}"
            )
            # Could add logic to extend SVI or use simpler guide
        elif jnp.isnan(relative_std):
            logger.error("SVI failed: NaN encountered in losses. Disabling NeuTra.")
            return self.model, init_to_median(num_samples=1000), None
        else:
            logger.info(
                f"SVI converged successfully. Final loss: {svi_result.losses[-1]:.4f}"
            )

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

    def sample(self):
        """
        Run the NUTS sampler to perform MCMC sampling.
        """
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, svi_key, mcmc_key = random.split(rng_key, 3)

        if self.sampler.use_neutra:
            model, init_strategy, neutra = self._initialize_neutra()
        else:
            starters, _, _, _ = lsq_model_fitter(
                self.template,
                self._data,
                out_dir=f"{self._config.output_path}",
            )
            # init_values = {k: v[0] for k, v in starters.items()}
            init_values = lsq_to_base_space(starters, self.template)

            model, init_strategy, neutra = (
                self.model,
                init_to_median(num_samples=1000),
                None,
            )

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
        )

        posterior_samples = self.get_posterior_samples(mcmc, neutra)

        self._idata = self._compose_inference_data(
            mcmc, posterior_samples, prior_model=model
        )
