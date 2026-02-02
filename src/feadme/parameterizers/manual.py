import astropy.constants as const
import astropy.units as u
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.scipy.stats import norm
from jax.typing import ArrayLike
from jax.scipy.special import erf, erfinv

from ..parser import Distribution, Parameter

FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
ERR = 1e-5
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value


def trunchalfnorm_ppf(q, loc, scale, upper_limit):
    """
    PPF of a truncated half-normal distribution.
    X >= loc
    truncated to (loc, upper_limit)
    """
    # upper boundary CDF for half-normal
    su = (upper_limit - loc) / (scale * jnp.sqrt(2))
    cdf_upper = erf(su)

    # rescale q into (0, cdf_upper)
    eps = 1e-12
    q_scaled = eps + (cdf_upper - eps) * q  # avoids hard clipping

    return loc + scale * jnp.sqrt(2.0) * erfinv(q_scaled)


def truncnorm_ppf(q, loc, scale, lower_limit, upper_limit):
    """
    PPF of truncated normal:
    lower_limit < X < upper_limit
    """
    a = (lower_limit - loc) / scale
    b = (upper_limit - loc) / scale

    cdf_a = norm.cdf(a)
    cdf_b = norm.cdf(b)

    # safe interpolation
    eps = 1e-12
    q_scaled = cdf_a + q * (cdf_b - cdf_a - eps)

    return norm.ppf(q_scaled) * scale + loc


def sample_param(samp_name: str, param: Parameter) -> ArrayLike:
    if param.circular:
        circ_x_base = numpyro.sample(f"{samp_name}_x_base", dist.Normal(0, 1))
        circ_y_base = numpyro.sample(f"{samp_name}_y_base", dist.Normal(0, 1))

        param_samp = numpyro.deterministic(
            samp_name, jnp.mod(jnp.arctan2(circ_y_base, circ_x_base), 2 * jnp.pi)
        )

        return param_samp

    z = numpyro.sample(f"{samp_name}_base", dist.Normal(0.0, 1.0))

    # smooth CDF mapping that avoids clipping
    eps = 1e-7
    u = eps + (1 - 2 * eps) * norm.cdf(z)

    lower_limit = param.low
    upper_limit = param.high

    # if param.name == "inclination":
    #     mu_min = jnp.cos(upper_limit)  # cos(i_max)
    #     mu_max = jnp.cos(lower_limit)  # cos(i_min)
    #     mu = mu_min + u * (mu_max - mu_min)
    #     val = jnp.arccos(mu)
    #
    #     return numpyro.deterministic(samp_name, val)

    if param.distribution == Distribution.UNIFORM:
        val = lower_limit + u * (upper_limit - lower_limit)

    elif param.distribution == Distribution.LOG_UNIFORM:
        log_low = jnp.log(lower_limit)
        log_high = jnp.log(upper_limit)
        val = jnp.exp(log_low + u * (log_high - log_low))

    elif param.distribution == Distribution.NORMAL:
        val = truncnorm_ppf(u, param.loc, param.scale, lower_limit, upper_limit)

    elif param.distribution == Distribution.LOG_NORMAL:
        mu = jnp.log(param.loc)
        sigma = jnp.log(param.scale)
        y = truncnorm_ppf(u, mu, sigma, jnp.log(lower_limit), jnp.log(upper_limit))
        val = jnp.exp(y)

    elif param.distribution == Distribution.HALF_NORMAL:
        val = trunchalfnorm_ppf(
            u, loc=lower_limit, scale=param.scale, upper_limit=upper_limit
        )

    elif param.distribution == Distribution.LOG_HALF_NORMAL:
        mu = jnp.log(lower_limit)
        sigma = jnp.log(param.scale)
        y = trunchalfnorm_ppf(u, loc=mu, scale=sigma, upper_limit=jnp.log(upper_limit))
        val = jnp.exp(y)

    else:
        raise ValueError(f"Unsupported distribution: {param.distribution}")

    return numpyro.deterministic(samp_name, val)
