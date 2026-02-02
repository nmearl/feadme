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
from numpyro.distributions.transforms import biject_to, ExpTransform
from numpyro.distributions import transforms as T

from .parser import Distribution, Template, Shape, Parameter

FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
ERR = 1e-5
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value


def _sample_no_reparam(samp_name: str, param: Parameter) -> ArrayLike:
    if param.circular:
        circ_x_base = numpyro.sample(f"{samp_name}_x_base", dist.Normal(0, 1))
        circ_y_base = numpyro.sample(f"{samp_name}_y_base", dist.Normal(0, 1))

        param_samp = numpyro.deterministic(
            samp_name, jnp.mod(jnp.arctan2(circ_y_base, circ_x_base), 2 * jnp.pi)
        )

        return param_samp

    # if param.name == "inclination":
    #     mu_min = jnp.cos(param.high)  # cos(i_max)
    #     mu_max = jnp.cos(param.low)  # cos(i_min)
    #     mu = numpyro.sample(
    #         f"{samp_name}_base",
    #         dist.Uniform(mu_min, mu_max),
    #     )
    #     incl = jnp.arccos(mu)
    #     return numpyro.deterministic(samp_name, incl)

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
        sigma_log = jnp.sqrt(jnp.log(1 + (param.scale / param.loc) ** 2))
        mu_log = jnp.log(param.loc) - sigma_log**2 / 2

        base_dist = numpyro.sample(
            f"{samp_name}_base",
            dist.TruncatedNormal(
                loc=mu_log,
                scale=sigma_log,
                low=jnp.log(param.low),
                high=jnp.log(param.high),
            ),
        )

        param_samp = numpyro.deterministic(samp_name, jnp.exp(base_dist))

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


def create_reparam_config(template: Template) -> dict:
    """Create reparameterization configuration for parameters."""
    reparam_config = {}

    for prof in template.disk_profiles + template.line_profiles:
        for param in prof.independent:
            samp_name = param.qualified_name

            if param.circular:
                reparam_config[f"{samp_name}_base"] = CircularReparam()
            else:
                reparam_config[samp_name] = TransformReparam()

    return reparam_config


def _uniform_affine(low, high):
    # u ~ Uniform(0,1); x = low + (high-low)*u
    return dist.TransformedDistribution(
        dist.Uniform(0.0, 1.0),
        T.AffineTransform(loc=low, scale=(high - low)),
    )


def _loguniform(low, high):
    # u ~ Uniform(0,1); z = loglow + (loghigh-loglow)*u; x = exp(z)
    loglow, loghigh = jnp.log(low), jnp.log(high)
    return dist.TransformedDistribution(
        dist.Uniform(0.0, 1.0),
        [T.AffineTransform(loc=loglow, scale=(loghigh - loglow)), T.ExpTransform()],
    )


TAU = 2.0 * jnp.pi


def _sample_circular(samp_name: str, low: float, high: float):
    # VonMises(kappa=0) is uniform on [-pi, pi).
    theta = numpyro.sample(samp_name, dist.VonMises(0.0, 0.0))

    numpyro.deterministic(f"{samp_name}_wrapped", jnp.mod(theta, TAU))

    return theta


def _bounded_normal_like(low, high, loc, scale):
    """
    Smoothly map a normal into [low, high] with a logistic transform.
    loc/scale are interpreted as *initialization-ish*; not exact moments.
    """
    # base ~ Normal(0,1)
    base = dist.Normal(0.0, 1.0)

    # affine to approx loc/scale in unconstrained space
    # then sigmoid to (0,1), then affine to (low, high)
    transform = [
        T.AffineTransform(loc=loc, scale=scale),
        T.SigmoidTransform(),
        T.AffineTransform(loc=low, scale=(high - low)),
    ]
    return dist.TransformedDistribution(base, transform)


def _bounded_lognormal(mean, std, low, high):
    sigma_log = jnp.sqrt(jnp.log1p((std / mean) ** 2))
    mu_log = jnp.log(mean) - 0.5 * sigma_log**2

    log_low, log_high = jnp.log(low), jnp.log(high)
    span = log_high - log_low

    # z ~ Normal(0,1)
    # y = mu_log + sigma_log*z   (roughly centered)
    # u = sigmoid(y) in (0,1)
    # t = log_low + span*u in (log_low, log_high)
    # x = exp(t) in (low, high)
    return dist.TransformedDistribution(
        dist.Normal(0.0, 1.0),
        [
            T.AffineTransform(loc=mu_log, scale=sigma_log),
            T.SigmoidTransform(),
            T.AffineTransform(loc=log_low, scale=span),
            T.ExpTransform(),
        ],
    )


def _sample_auto_reparam(samp_name: str, param: Parameter):
    """
    Sampling compatible with numpyro.handlers.reparam + your create_reparam_config().

    - circular params: config targets f"{samp_name}_base" with CircularReparam()
      -> we sample that site and then expose samp_name deterministically.

    - non-circular params: config targets samp_name with TransformReparam()
      -> we sample samp_name directly from the intended constrained distribution.
    """

    # --- Circular parameters: sample the site that reparam config targets
    if param.circular:
        return _sample_circular(samp_name, param.low, param.high)

    # --- Non-circular parameters: sample at samp_name (TransformReparam will act here)

    if param.distribution == Distribution.UNIFORM:
        return numpyro.sample(samp_name, _uniform_affine(param.low, param.high))

    if param.distribution == Distribution.LOG_UNIFORM:
        return numpyro.sample(samp_name, _loguniform(param.low, param.high))

    if param.distribution == Distribution.NORMAL:
        return numpyro.sample(
            samp_name, _bounded_normal_like(param.low, param.high, loc=0.0, scale=1.0)
        )

    if param.distribution == Distribution.LOG_NORMAL:
        return numpyro.sample(
            samp_name,
            _bounded_lognormal(param.loc, param.scale, param.low, param.high),
        )

    if param.distribution == Distribution.HALF_NORMAL:
        # Intended: shifted half-normal on [low, high]. NumPyro doesn't always provide a
        # "shifted+truncated HalfNormal" distribution in older versions, so we use a proper,
        # bounded approximation:
        return numpyro.sample(
            samp_name,
            dist.TruncatedNormal(
                loc=param.low, scale=param.scale, low=param.low, high=param.high
            ),
        )

    if param.distribution == Distribution.LOG_HALF_NORMAL:
        # Same idea in log-space, then exponentiate. Uses a proper bounded approximation.
        log_low = jnp.log(param.low)
        log_high = jnp.log(param.high)
        base = dist.TruncatedNormal(
            loc=log_low, scale=param.scale, low=log_low, high=log_high
        )
        return numpyro.sample(
            samp_name, dist.TransformedDistribution(base, ExpTransform())
        )

    raise ValueError(f"Unsupported distribution type: {param.distribution}")
