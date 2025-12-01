from typing import Dict, Optional

import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import CircularReparam, TransformReparam
from numpyro.handlers import reparam
from jax.typing import ArrayLike
from jax.scipy.special import erf, erfinv
from jax.scipy.stats import norm
from numpyro.distributions import constraints
from numpyro.distributions.transforms import biject_to


from .parser import Distribution, Template, Shape, Parameter

FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
ERR = 1e-5
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value


def _sample_no_reparam(samp_name: str, param: Parameter) -> ArrayLike:
    if param.circular:
        circ_base = numpyro.sample(f"{samp_name}_base", dist.Normal(0, 1).expand([2]))
        param_samp = numpyro.deterministic(
            samp_name, jnp.mod(jnp.arctan2(circ_base[1], circ_base[0]), 2 * jnp.pi)
        )

        return param_samp

    if param.name == "inclination":
        mu_min = jnp.cos(1.5)  # cos(i_max)
        mu_max = jnp.cos(param.low)  # cos(i_min)

        mu = numpyro.sample(
            f"{samp_name}_base",
            dist.Uniform(mu_min, mu_max),
        )
        incl = jnp.arccos(mu)
        return numpyro.deterministic(samp_name, incl)

    if param.distribution == Distribution.UNIFORM:
        param_samp = numpyro.sample(samp_name, dist.Uniform(param.low, param.high))

    elif param.distribution == Distribution.LOG_UNIFORM:
        param_samp = numpyro.sample(samp_name, dist.LogUniform(param.low, param.high))

    elif param.distribution == Distribution.NORMAL:
        param_samp = numpyro.sample(
            samp_name,
            dist.TruncatedNormal(
                param.loc, param.scale, low=param.low, high=param.high
            ),
        )

    elif param.distribution == Distribution.LOG_NORMAL:
        param_samp = numpyro.sample(
            samp_name,
            dist.TransformedDistribution(
                dist.TruncatedNormal(
                    jnp.log(param.loc),
                    jnp.log(param.scale),
                    low=jnp.log(param.low),
                    high=jnp.log(param.high),
                ),
                dist.transforms.ExpTransform(),
            ),
        )

    elif param.distribution == Distribution.HALF_NORMAL:
        param_samp = numpyro.sample(
            samp_name,
            dist.TransformedDistribution(
                dist.TruncatedDistribution(
                    dist.Normal(0, param.scale),
                    low=0,
                    high=param.high - param.low,
                ),
                dist.transforms.AffineTransform(loc=param.low, scale=1.0),
            ),
        )

    elif param.distribution == Distribution.LOG_HALF_NORMAL:
        param_samp = numpyro.sample(
            samp_name,
            dist.TransformedDistribution(
                dist.TruncatedDistribution(
                    dist.Normal(0, jnp.log(param.scale)),
                    low=0,
                    high=jnp.log(param.high) - jnp.log(param.low),
                ),
                [
                    dist.transforms.AffineTransform(loc=jnp.log(param.low), scale=1.0),
                    dist.transforms.ExpTransform(),
                ],
            ),
        )

    return param_samp


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


def _sample_manual_reparam(samp_name: str, param: Parameter) -> ArrayLike:
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

    if param.name == "inclination":
        mu_min = jnp.cos(1.5)  # cos(i_max)
        mu_max = jnp.cos(lower_limit)  # cos(i_min)
        mu = mu_min + u * (mu_max - mu_min)
        val = jnp.arccos(mu)

        return numpyro.deterministic(samp_name, val)

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


def create_reparam_config(template: Template, circ_only: bool = False) -> dict:
    """Create reparameterization configuration for parameters."""
    reparam_config = {}

    for prof in template.disk_profiles + template.line_profiles:
        for param in prof.independent:
            samp_name = param.qualified_name

            if param.circular:
                reparam_config[f"{samp_name}_base"] = CircularReparam()
            elif not circ_only:
                reparam_config[samp_name] = TransformReparam()

    # Template-level parameters
    if not circ_only:
        if not template.redshift.fixed:
            reparam_config["redshift"] = TransformReparam()

        if not template.white_noise.fixed:
            reparam_config["white_noise"] = TransformReparam()

    return reparam_config


def _sample_auto_reparam(samp_name: str, param: Parameter) -> ArrayLike:
    if param.circular:
        circ_base = numpyro.sample(
            f"{samp_name}_base",
            dist.VonMises(loc=param.loc, concentration=1e-3),
        )
        return numpyro.deterministic(samp_name, circ_base % (2 * jnp.pi))

    if param.name == "inclination":
        # param.low, param.high are the inclination bounds (in radians)
        i_min, i_max = param.low, param.high

        # cos(i) decreases with i, so the interval is [cos(i_max), cos(i_min)]
        mu = numpyro.sample(
            f"{samp_name}_base",
            dist.Uniform(jnp.cos(i_max), jnp.cos(i_min)),
        )
        incl = jnp.arccos(mu)
        return numpyro.deterministic(samp_name, incl)

    if param.distribution == Distribution.UNIFORM:
        base = dist.Uniform(0.0, 1.0)
        transform = dist.transforms.AffineTransform(
            loc=param.low, scale=param.high - param.low
        )
        return numpyro.sample(
            samp_name,
            dist.TransformedDistribution(base, transform),
        )

    if param.distribution == Distribution.LOG_UNIFORM:
        base = dist.Uniform(0.0, 1.0)
        transform = dist.transforms.ComposeTransform(
            [
                dist.transforms.AffineTransform(
                    loc=jnp.log(param.low),
                    scale=jnp.log(param.high) - jnp.log(param.low),
                ),
                dist.transforms.ExpTransform(),
            ]
        )
        return numpyro.sample(
            samp_name,
            dist.TransformedDistribution(base, transform),
        )

    if param.distribution == Distribution.NORMAL:
        base = dist.Normal(0.0, 1.0)
        interval = constraints.interval(param.low, param.high)
        transform = biject_to(interval)  # typically Sigmoid + Affine under the hood
        return numpyro.sample(
            samp_name,
            dist.TransformedDistribution(base, transform),
        )

    if param.distribution == Distribution.LOG_NORMAL:
        log_low = jnp.log(param.low)
        log_high = jnp.log(param.high)

        base = dist.Normal(0.0, 1.0)
        interval = constraints.interval(log_low, log_high)
        log_transform = biject_to(interval)

        transform = dist.transforms.ComposeTransform(
            [log_transform, dist.transforms.ExpTransform()]
        )

        return numpyro.sample(
            samp_name,
            dist.TransformedDistribution(base, transform),
        )

    if param.distribution == Distribution.HALF_NORMAL:
        base = dist.Normal(0.0, 1.0)
        interval = constraints.interval(param.low, param.high)
        transform = biject_to(interval)
        return numpyro.sample(
            samp_name,
            dist.TransformedDistribution(base, transform),
        )

    if param.distribution == Distribution.LOG_HALF_NORMAL:
        log_low = jnp.log(param.low)
        log_high = jnp.log(param.high)

        base = dist.Normal(0.0, 1.0)
        interval = constraints.interval(log_low, log_high)
        log_transform = biject_to(interval)

        transform = dist.transforms.ComposeTransform(
            [log_transform, dist.transforms.ExpTransform()]
        )

        return numpyro.sample(
            samp_name,
            dist.TransformedDistribution(base, transform),
        )

    raise ValueError(f"Unsupported distribution type: {param.distribution}")
