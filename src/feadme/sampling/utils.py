import jax
import jax.numpy as jnp
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model
import jax.random as random
import loguru

logger = loguru.logger.opt(colors=True)


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
    """
    Convert constrained starting point(s) to unconstrained init_params for
    mcmc.run(), with small jitter applied to each chain.

    Parameters
    ----------
    base_values : dict | list[dict]
        Either a single constrained parameter dict (all chains share it) or
        a list of dicts of length num_chains (one starting point per chain).
    rng_keys : array
        Per-chain RNG keys, shape (num_chains, 2).
    """
    if isinstance(base_values, dict):
        base_values = [base_values] * len(rng_keys)

    inits = []
    for values, k in zip(base_values, rng_keys):
        init_params, *_ = initialize_model(
            k,
            model,
            model_args=model_args,
            model_kwargs=model_kwargs,
            init_strategy=init_to_value(values=values),
        )
        inits.append(jitter_unconstrained(init_params, k, scale=0.02))

    return jax.tree.map(lambda *xs: jnp.stack(xs, 0), *inits)


def diagnose_init_params(model, init_params, config):
    rng = random.PRNGKey(0)
    model_kwargs = dict(
        wave=config.data.masked_wave,
        flux=config.data.masked_flux,
        flux_err=config.data.masked_flux_err,
    )

    # First check if the full set fails
    try:
        initialize_model(
            rng,
            model,
            model_kwargs=model_kwargs,
            init_strategy=init_to_value(values=init_params),
        )
        logger.info("Full init_params: OK")
        return
    except Exception as e:
        logger.warning(f"Full init_params failed: {e}")

    # Binary-search: try dropping each param one at a time
    bad_params = []
    for key in init_params:
        reduced = {k: v for k, v in init_params.items() if k != key}
        try:
            initialize_model(
                rng,
                model,
                model_kwargs=model_kwargs,
                init_strategy=init_to_value(values=reduced),
            )
            bad_params.append(key)
            logger.warning(f"  Removing '{key}' fixes initialization — likely culprit")
        except Exception:
            pass  # Still broken without this param, not the sole cause

    if not bad_params:
        logger.warning("No single parameter identified — may be an interaction effect")

    # For each suspected bad param, log its value vs prior bounds
    for key in bad_params:
        val = init_params[key]
        param = next(
            (p for p in config.template.parameters if p.qualified_name == key), None
        )
        if param:
            logger.warning(
                f"  {key} = {val:.4g}  bounds=[{param.low:.4g}, {param.high:.4g}]"
                f"  {'OUT OF BOUNDS' if val <= param.low or val >= param.high else 'in bounds'}"
            )
        else:
            logger.warning(
                f"  {key} = {val:.4g}  (no matching template param found — may be a _base site)"
            )
