import flax.struct
import jax
import jax.numpy as jnp
import loguru
from jax.typing import ArrayLike
from numpyro.infer.util import Predictive
from numpyro.infer.util import log_likelihood

from .base_model import BaseModel
from ..core.parser import Config

logger = loguru.logger.opt(colors=True)


@flax.struct.dataclass
class BaseSampler:
    progress_bar: bool = True
    sampler_type: str = ""

    @staticmethod
    def _inference_data(
        config: Config,
        model: BaseModel,
        posterior_samples: dict[str, ArrayLike],
        compute_prior_predictive: bool = False,
    ) -> tuple[dict, dict, dict]:
        """
        Create inference data for posterior predictive, prior predictive, and
        log-likelihood.

        Parameters
        ----------
        posterior_samples: dict[str, ArrayLike]
            Posterior samples obtained from the sampler.

        Returns
        -------
            A tuple containing posterior predictive, prior predictive, and
            log-likelihood data.
        """
        rng_key = jax.random.PRNGKey(0)

        # Posterior predictive
        predictive_post = Predictive(model, posterior_samples=posterior_samples)(
            rng_key,
            wave=config.data.masked_wave,
            flux=None,
            flux_err=config.data.masked_flux_err,
        )

        predictive_post.update(
            {
                k: jnp.zeros_like(v)
                for k, v in posterior_samples.items()
                if k not in predictive_post
            }
        )

        # Prior predictive
        if compute_prior_predictive:
            predictive_prior = Predictive(model, num_samples=300)(
                rng_key,
                wave=config.data.masked_wave,
                flux=None,
                flux_err=config.data.masked_flux_err,
            )

            predictive_prior.update(
                {
                    k: jnp.zeros_like(v)
                    for k, v in posterior_samples.items()
                    if k not in predictive_prior
                }
            )
        else:
            predictive_prior = None

        # Compute log-likelihood for each posterior sample
        log_lik = log_likelihood(
            model,
            posterior_samples,
            wave=config.data.masked_wave,
            flux=config.data.masked_flux,
            flux_err=config.data.masked_flux_err,
        )

        return predictive_post, predictive_prior, log_lik
