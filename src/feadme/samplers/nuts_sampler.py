import arviz as az
import jax.random as random
from numpyro.infer import MCMC, NUTS, init_to_median

from .base_sampler import BaseSampler


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
        kernel = self.get_kernel()
        rng_key = random.PRNGKey(0)

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
            wave=self.wave,
            flux=self.flux,
            flux_err=self.flux_err,
            # template=self.template,
        )

        self._idata = self._compose_inference_data(mcmc)
