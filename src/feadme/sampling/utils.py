import jax
import jax.numpy as jnp
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model


def jitter_unconstrained(init_params, key, scale=0.05):
    return jax.tree.map(
        lambda x, k: x + scale * jax.random.normal(k, x.shape),
        init_params,
        jax.tree.unflatten(
            jax.tree.structure(init_params),
            jax.random.split(key, len(jax.tree.leaves(init_params))),
        ),
    )


def make_init_params(model, model_args, model_kwargs, base_values, rng_keys):
    inits = []
    for k in rng_keys:
        init_params, *_ = initialize_model(
            k,
            model,
            model_args=model_args,
            model_kwargs=model_kwargs,
            init_strategy=init_to_value(values=base_values),
        )
        inits.append(jitter_unconstrained(init_params, k, scale=0.02))
    return jax.tree.map(lambda *xs: jnp.stack(xs, 0), *inits)
