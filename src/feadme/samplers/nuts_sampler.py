import arviz as az
import loguru
import jax.random as random
from numpyro.infer import MCMC, NUTS, init_to_median, init_to_value
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO, SA
from numpyro.infer.autoguide import (
    AutoBNAFNormal,
    AutoIAFNormal,
    AutoMultivariateNormal,
    AutoDAIS,
)
from numpyro.infer.reparam import NeuTraReparam
from numpyro import optim
import jax.numpy as jnp
import time

from .base_sampler import BaseSampler


logger = loguru.logger.opt(colors=True)


class NUTSSampler(BaseSampler):
    def get_kernel(self):
        """
        Create a NUTS kernel for sampling.
        """
        return NUTS(
            self.model,
            init_strategy=init_to_median(num_samples=1000),
            target_accept_prob=self.sampler.target_accept_prob,
            max_tree_depth=self.sampler.max_tree_depth,
            dense_mass=self.sampler.dense_mass,
        )

    def sample(self):
        """
        Run the NUTS sampler to perform MCMC sampling.
        """
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, svi_key, mcmc_key = random.split(rng_key, 3)

        guide = AutoBNAFNormal(self.model, hidden_factors=[8, 8], num_flows=2)
        # guide = AutoIAFNormal(self.model, hidden_dims=[32, 32], num_flows=2)
        # guide = AutoMultivariateNormal(self.model)
        optimizer = optim.Adam(0.003)
        svi = SVI(self.model, guide, optimizer, Trace_ELBO())
        svi_result = svi.run(
            svi_key,
            25_000,
            template=self.template,
            wave=self.wave,
            flux=self.flux,
            flux_err=self.flux_err,
        )

        # Convergence check
        recent_losses = svi_result.losses[-1000:]
        relative_std = jnp.std(recent_losses) / jnp.abs(jnp.mean(recent_losses))

        if relative_std > 0.01:
            logger.warning(
                f"SVI may not have converged! Relative std: {relative_std:.4f}"
            )
            # Could add logic to extend SVI or use simpler guide
        else:
            logger.info(
                f"SVI converged successfully. Final loss: {svi_result.losses[-1]:.4f}"
            )

        neutra = NeuTraReparam(guide, svi_result.params)
        neutra_model = neutra.reparam(self.model)

        # Initialize from VI posterior
        init_key, mcmc_key = random.split(mcmc_key)
        init_params = guide.sample_posterior(
            init_key, svi_result.params, sample_shape=(self.sampler.num_chains,)
        )

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

        kernel = NUTS(
            neutra_model,
            init_strategy=init_to_value(values=chain_init_params),
            # init_strategy=init_to_median(),
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

        self._idata = self._compose_inference_data(mcmc, neutra, neutra_model)
