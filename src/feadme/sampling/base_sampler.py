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


def _concat_tree(list_of_trees):
    return jax.tree.map(lambda *xs: jnp.concatenate(xs, axis=0), *list_of_trees)


@flax.struct.dataclass
class BaseSampler:
    progress_bar: bool = True
    sampler_type: str = ""

    @staticmethod
    def _inference_data(
        config,
        model,
        posterior_samples,
        compute_prior_predictive: bool = False,
        chunk_size: int = 64,
    ):
        rng_key = jax.random.PRNGKey(0)
        wave = config.data.masked_wave
        flux = config.data.masked_flux
        flux_err = config.data.masked_flux_err

        n_samples = next(iter(posterior_samples.values())).shape[0]

        post_chunks = []
        loglik_chunks = []

        for start in range(0, n_samples, chunk_size):
            stop = min(start + chunk_size, n_samples)
            actual_size = stop - start

            samp_chunk = {k: v[start:stop] for k, v in posterior_samples.items()}

            if actual_size < chunk_size:
                pad = chunk_size - actual_size
                samp_chunk = {
                    k: jnp.concatenate([v, v[:pad]], axis=0)
                    for k, v in samp_chunk.items()
                }

            rng_key, subkey = jax.random.split(rng_key)

            pred_chunk = Predictive(model, posterior_samples=samp_chunk)(
                subkey,
                wave=wave,
                flux=None,
                flux_err=flux_err,
            )
            pred_chunk = {k: v[:actual_size] for k, v in pred_chunk.items()}
            post_chunks.append(pred_chunk)

            ll_chunk = log_likelihood(
                model,
                samp_chunk,
                wave=wave,
                flux=flux,
                flux_err=flux_err,
            )
            ll_chunk = {k: v[:actual_size] for k, v in ll_chunk.items()}
            loglik_chunks.append(ll_chunk)

        predictive_post = _concat_tree(post_chunks)
        log_lik = _concat_tree(loglik_chunks)

        if compute_prior_predictive:
            predictive_prior = Predictive(model, num_samples=300)(
                rng_key,
                wave=wave,
                flux=None,
                flux_err=flux_err,
            )
        else:
            predictive_prior = None

        return predictive_post, predictive_prior, log_lik
