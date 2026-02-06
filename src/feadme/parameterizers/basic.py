import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike

from ..parser import Distribution, Parameter

FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
ERR = 1e-5
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value


def _logit01(name: str, *, eps: float = 0.0) -> ArrayLike:
    """
    Sample u ~ Uniform(0,1) exactly via unconstrained latent:
      z ~ Logistic(0,1), u = sigmoid(z)
    If eps>0, returns u in (eps, 1-eps) (no longer exactly uniform).
    """
    z = numpyro.sample(f"{name}_base", dist.Logistic(0.0, 1.0))
    u01 = jax.nn.sigmoid(z)

    return eps + (1.0 - 2.0 * eps) * u01


def _logit_uniform(
    name: str, low: ArrayLike, high: ArrayLike, *, eps: float = 0.0
) -> ArrayLike:
    """
    Sample x ~ Uniform(low, high) exactly via unconstrained latent.
    """
    u01 = _logit01(name, eps=eps)
    x = low + (high - low) * u01

    return numpyro.deterministic(name, x)


def _logit_loguniform(
    name: str, low: ArrayLike, high: ArrayLike, *, eps: float = 0.0
) -> ArrayLike:
    """
    Sample x ~ LogUniform(low, high) exactly via unconstrained latent:
      logx = log(low) + u*(log(high)-log(low)), x = exp(logx)
    """
    u01 = _logit01(name, eps=eps)
    log_low = jnp.log(low)
    log_high = jnp.log(high)
    logx = log_low + (log_high - log_low) * u01
    x = jnp.exp(logx)

    return numpyro.deterministic(name, x)


def sample_param(
    samp_name: str,
    param: Parameter,
    lower_bound: float,
    upper_bound: float,
) -> ArrayLike:
    if param.circular:
        circ_x_base = numpyro.sample(f"{samp_name}_x_base", dist.Normal(0.0, 1.0))
        circ_y_base = numpyro.sample(f"{samp_name}_y_base", dist.Normal(0.0, 1.0))
        return numpyro.deterministic(
            samp_name, jnp.mod(jnp.arctan2(circ_y_base, circ_x_base), 2.0 * jnp.pi)
        )

    if param.name == "inclination":
        mu_min = jnp.cos(upper_bound)  # cos(i_max)
        mu_max = jnp.cos(lower_bound)  # cos(i_min)
        # mu = _logit_uniform(f"{samp_name}_base", mu_min, mu_max)
        mu = numpyro.sample(f"{samp_name}_base", dist.Uniform(mu_min, mu_max))
        incl = jnp.arccos(mu)

        return numpyro.deterministic(samp_name, incl)

    if param.distribution == Distribution.UNIFORM:
        # param_samp = _logit_uniform(samp_name, lower_bound, upper_bound)
        param_samp = numpyro.sample(samp_name, dist.Uniform(lower_bound, upper_bound))

    elif param.distribution == Distribution.LOG_UNIFORM:
        # param_samp = _logit_loguniform(samp_name, lower_bound, upper_bound)
        param_samp = numpyro.sample(
            samp_name,
            dist.LogUniform(lower_bound, upper_bound),
        )

    elif param.distribution == Distribution.NORMAL:
        param_samp = numpyro.sample(
            samp_name,
            dist.TruncatedNormal(
                param.loc, param.scale, low=lower_bound, high=upper_bound
            ),
        )

    elif param.distribution == Distribution.LOG_NORMAL:
        sigma_log = jnp.sqrt(jnp.log(1.0 + (param.scale / param.loc) ** 2))
        mu_log = jnp.log(param.loc) - 0.5 * sigma_log**2

        base = numpyro.sample(
            f"{samp_name}_base",
            dist.TruncatedNormal(
                loc=mu_log,
                scale=sigma_log,
                low=jnp.log(lower_bound),
                high=jnp.log(upper_bound),
            ),
        )
        param_samp = numpyro.deterministic(samp_name, jnp.exp(base))

    else:
        raise ValueError(f"Unsupported distribution: {param.distribution}")

    return param_samp
