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
            samp_name, jnp.arctan2(circ_y_base, circ_x_base) % (2 * jnp.pi)
        )

    elif param.distribution == Distribution.UNIFORM:
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
    Compute the percent point function (PPF) of a truncated half-normal distribution.
    """
    # CDF at upper boundary
    standardized_upper = (upper_limit - loc) / (scale * jnp.sqrt(2.0))
    cdf_upper = erf(standardized_upper)

    # Scale the uniform sample to the truncated range
    # Since cdf_lower = 0, this is just q * cdf_upper
    scaled_q = q * cdf_upper

    # Apply inverse CDF
    return loc + scale * jnp.sqrt(2.0) * erfinv(scaled_q)


def truncnorm_ppf(q, loc, scale, lower_limit, upper_limit):
    """
    Compute the percent point function (PPF) of a truncated normal distribution.
    """
    a = (lower_limit - loc) / scale
    b = (upper_limit - loc) / scale

    # Compute CDF bounds
    cdf_a = norm.cdf(a)
    cdf_b = norm.cdf(b)

    # Compute the truncated normal PPF
    return norm.ppf(cdf_a + q * (cdf_b - cdf_a)) * scale + loc


def _sample_manual_reparam(samp_name: str, param: Parameter):
    low, high = param.low, param.high

    # if param.circular:
    #     circ_x_base = numpyro.sample(f"{samp_name}_x_base", dist.Normal(0, 1))
    #     circ_y_base = numpyro.sample(f"{samp_name}_y_base", dist.Normal(0, 1))
    #     param_samp = numpyro.deterministic(
    #         samp_name, jnp.arctan2(circ_y_base, circ_x_base) % (2 * jnp.pi)
    #     )
    #     return param_samp

    # Scalar base
    # u = numpyro.sample(f"{samp_name}_base", dist.Uniform(0.0, 1.0))
    # u = jax.nn.sigmoid(z)
    z = numpyro.sample(f"{samp_name}_base", dist.Normal(0.0, 1.0))
    u = norm.cdf(z)

    if param.distribution == Distribution.UNIFORM:
        val = low + u * (high - low)

    elif param.distribution == Distribution.LOG_UNIFORM:
        val = jnp.exp(jnp.log(low) + u * (jnp.log(high) - jnp.log(low)))

    elif param.distribution == Distribution.NORMAL:
        val = truncnorm_ppf(u, param.loc, param.scale, low, high)

    elif param.distribution == Distribution.LOG_NORMAL:
        y = truncnorm_ppf(
            u, jnp.log(param.loc), jnp.log(param.scale), jnp.log(low), jnp.log(high)
        )
        val = jnp.exp(y)

    elif param.distribution == Distribution.HALF_NORMAL:
        val = trunchalfnorm_ppf(u, loc=low, scale=param.scale, upper_limit=high)

    elif param.distribution == Distribution.LOG_HALF_NORMAL:
        mu = jnp.log(low)
        sigma = jnp.log(param.scale)
        y = trunchalfnorm_ppf(u, loc=mu, scale=sigma, upper_limit=jnp.log(high))
        val = jnp.exp(y)

    else:
        raise ValueError(f"Unsupported distribution: {param.distribution}")

    return numpyro.deterministic(samp_name, val)


def create_reparam_config(template: Template) -> dict:
    """Create reparameterization configuration for parameters."""
    reparam_config = {}

    for prof in template.disk_profiles + template.line_profiles:
        for param in prof.independent:
            samp_name = f"{prof.name}_{param.name}"

            if param.circular:
                reparam_config[f"{samp_name}_base"] = CircularReparam()
            # else:
            #     reparam_config[samp_name] = TransformReparam()

    return reparam_config


def _sample_auto_reparam(samp_name: str, param: Parameter) -> ArrayLike:
    if param.circular:
        circ_base = numpyro.sample(
            f"{samp_name}_base",
            dist.VonMises(loc=param.loc, concentration=1e-3),
        )
        param_samp = numpyro.deterministic(samp_name, circ_base % (2 * jnp.pi))

    elif param.distribution == Distribution.UNIFORM:
        param_samp = numpyro.sample(
            samp_name,
            dist.TransformedDistribution(
                dist.Uniform(0, 1),
                dist.transforms.AffineTransform(
                    loc=param.low, scale=param.high - param.low
                ),
            ),
        )

    elif param.distribution == Distribution.LOG_UNIFORM:
        param_samp = numpyro.sample(
            samp_name,
            dist.TransformedDistribution(
                dist.Uniform(0, 1),
                [
                    dist.transforms.AffineTransform(
                        loc=jnp.log(param.low),
                        scale=jnp.log(param.high) - jnp.log(param.low),
                    ),
                    dist.transforms.ExpTransform(),
                ],
            ),
        )

    elif param.distribution == Distribution.NORMAL:
        param_samp = numpyro.sample(
            samp_name,
            dist.TransformedDistribution(
                dist.Normal(0, 1),
                [
                    dist.transforms.SigmoidTransform(),
                    dist.transforms.AffineTransform(
                        loc=param.low, scale=param.high - param.low
                    ),
                ],
            ),
        )

    elif param.distribution == Distribution.LOG_NORMAL:
        param_samp = numpyro.sample(
            samp_name,
            dist.TransformedDistribution(
                dist.Normal(0, 1),
                [
                    dist.transforms.SigmoidTransform(),
                    dist.transforms.AffineTransform(
                        loc=jnp.log(param.low),
                        scale=jnp.log(param.high) - jnp.log(param.low),
                    ),
                    dist.transforms.ExpTransform(),
                ],
            ),
        )

    elif param.distribution == Distribution.HALF_NORMAL:
        param_samp = numpyro.sample(
            samp_name,
            dist.TransformedDistribution(
                dist.HalfNormal(1),
                [
                    dist.transforms.SigmoidTransform(),
                    dist.transforms.AffineTransform(
                        loc=param.low, scale=param.high - param.low
                    ),
                ],
            ),
        )

    elif param.distribution == Distribution.LOG_HALF_NORMAL:
        param_samp = numpyro.sample(
            samp_name,
            dist.TransformedDistribution(
                dist.HalfNormal(1),
                [
                    dist.transforms.SigmoidTransform(),
                    dist.transforms.AffineTransform(
                        loc=jnp.log(param.low),
                        scale=jnp.log(param.high) - jnp.log(param.low),
                    ),
                    dist.transforms.ExpTransform(),
                ],
            ),
        )

    return param_samp
