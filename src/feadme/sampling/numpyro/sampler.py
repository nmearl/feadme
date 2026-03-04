import time

import arviz as az
import flax.struct
import jax
import jax.numpy as jnp
import jax.random as random
import loguru
from jax.typing import ArrayLike
from numpyro.infer import MCMC, NUTS

from .model import BaseModel
from ..base_sampler import BaseSampler
from ..initializers import BaseInitializer, DefaultInitializer
from ...core.parser import Config
from ..utils import make_init_params

logger = loguru.logger.opt(colors=True)


@flax.struct.dataclass
class NUTSSampler(BaseSampler):
    sampler_type = "nuts"
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 1
    target_accept_prob: float = 0.85
    max_tree_depth: int = 10
    dense_mass: bool = True
    initializer: BaseInitializer = DefaultInitializer()

    @property
    def chain_method(self) -> str:
        return "vectorized" if jax.local_device_count() == 1 else "parallel"

    def __call__(self, config: Config, model: BaseModel) -> az.InferenceData:
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, mcmc_key = random.split(rng_key)

        init_params, init_strategy = self.initializer(config, model)

        kernel = NUTS(
            model,
            init_strategy=init_strategy,
            target_accept_prob=self.target_accept_prob,
            max_tree_depth=self.max_tree_depth,
            dense_mass=self.dense_mass,
            find_heuristic_step_size=True,
        )

        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            chain_method=self.chain_method,
            progress_bar=self.progress_bar,
        )

        mcmc.run(
            mcmc_key,
            wave=config.data.masked_wave,
            flux=config.data.masked_flux,
            flux_err=config.data.masked_flux_err,
            extra_fields=("num_steps", "diverging"),
        )

        posterior_samples = mcmc.get_samples(group_by_chain=True)

        extra = mcmc.get_extra_fields()
        num_steps = extra["num_steps"]
        tree_depth = jnp.log2(num_steps).astype(int) + 1
        max_depth = kernel._max_tree_depth

        divergences = extra["diverging"].sum()
        div_rate = 100 * divergences / extra["diverging"].size
        depth_hits = 100 * (tree_depth >= max_depth).mean()

        logger.info(
            f"Treedepth: {depth_hits:.1f}% at max={max_depth} | "
            f"Divergences: {int(divergences)} ({div_rate:.2f}%)"
        )

        return self._compose_inference_data(config, model, mcmc, posterior_samples)

    def _compose_inference_data(
        self,
        config: Config,
        model: BaseModel,
        mcmc: MCMC,
        posterior_samples: dict[str, ArrayLike],
    ) -> az.InferenceData:
        flat_samples = {
            k: v.reshape(-1, *v.shape[2:]) if v.ndim > 2 else v.reshape(-1)
            for k, v in posterior_samples.items()
        }

        predictive_post, predictive_prior, log_lik = self._inference_data(
            config, model, flat_samples
        )

        idata = az.from_numpyro(
            mcmc,
            posterior_predictive=predictive_post,
            prior=predictive_prior,
            log_likelihood=log_lik,
        )

        return idata
