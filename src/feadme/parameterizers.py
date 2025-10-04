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

from .models.disk import quad_jax_integrate, jax_integrate
from .parser import Distribution, Template, Shape, Parameter
from .utils import truncnorm_ppf, trunchalfnorm_ppf

ERR = float(np.finfo(np.float32).tiny)
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
        # low_std = (param.low - param.loc) / param.scale
        # high_std = (param.high - param.loc) / param.scale

        param_samp = numpyro.sample(
            samp_name,
            dist.TruncatedNormal(
                param.loc, param.scale, low=param.low, high=param.high
            ),
        )

    elif param.distribution == Distribution.LOG_NORMAL:
        low_std = (jnp.log(param.low) - jnp.log(param.loc)) / jnp.log(param.scale)
        high_std = (jnp.log(param.high) - jnp.log(param.loc)) / jnp.log(param.scale)

        param_samp = numpyro.sample(
            samp_name,
            dist.TransformedDistribution(
                dist.TruncatedNormal(0, 1, low=low_std, high=high_std),
                [
                    dist.transforms.AffineTransform(
                        loc=jnp.log(param.loc), scale=jnp.log(param.scale)
                    ),
                    dist.transforms.ExpTransform(),
                ],
            ),
        )
    elif param.distribution == Distribution.HALF_NORMAL:
        # high_std = (param.high - param.low) / param.scale

        param_samp = numpyro.sample(
            samp_name,
            dist.TruncatedDistribution(
                dist.HalfNormal(param.scale), low=param.low, high=param.high
            ),
        )
    elif param.distribution == Distribution.LOG_HALF_NORMAL:
        # high_std = (jnp.log(param.high) - jnp.log(param.low)) / jnp.log(param.scale)

        param_samp = numpyro.sample(
            samp_name,
            dist.TransformedDistribution(
                dist.HalfNormal(jnp.log(param.scale)),
                dist.transforms.ExpTransform(),
            ),
        )

    return param_samp


def _sample_manual_reparam(samp_name: str, param: Parameter) -> ArrayLike:
    param_low = param.low
    param_high = param.high

    if param.circular:
        circ_x_base = numpyro.sample(f"{samp_name}_x_base", dist.Normal(0, 1))
        circ_y_base = numpyro.sample(f"{samp_name}_y_base", dist.Normal(0, 1))
        param_samp = numpyro.deterministic(
            samp_name, jnp.arctan2(circ_y_base, circ_x_base) % (2 * jnp.pi)
        )

    elif param.distribution == Distribution.UNIFORM:
        uniform_base = numpyro.sample(f"{samp_name}_base", dist.Uniform(0, 1))
        param_samp = numpyro.deterministic(
            samp_name, param_low + uniform_base * (param_high - param_low)
        )

    elif param.distribution == Distribution.LOG_UNIFORM:
        log_uniform_base = numpyro.sample(
            f"{samp_name}_base",
            dist.Uniform(0, 1),
        )
        param_samp = numpyro.deterministic(
            samp_name,
            10
            ** (
                jnp.log10(param_low)
                + log_uniform_base * (jnp.log10(param_high) - jnp.log10(param_low))
            ),
        )

    elif param.distribution == Distribution.NORMAL:
        normal_base = numpyro.sample(f"{samp_name}_base", dist.Uniform(0, 1))
        param_samp = numpyro.deterministic(
            samp_name,
            truncnorm_ppf(normal_base, param.loc, param.scale, param_low, param_high),
        )

    elif param.distribution == Distribution.LOG_NORMAL:
        log_normal_base = numpyro.sample(f"{samp_name}_base", dist.Uniform(0, 1))
        param_samp = numpyro.deterministic(
            samp_name,
            10
            ** truncnorm_ppf(
                log_normal_base,
                jnp.log10(param_low),
                jnp.log10(param.scale),
                jnp.log10(param_low),
                jnp.log10(param_high),
            ),
        )

    elif param.distribution == Distribution.HALF_NORMAL:
        half_normal_base = numpyro.sample(f"{samp_name}_base", dist.Uniform(0, 1))
        param_samp = numpyro.deterministic(
            samp_name,
            trunchalfnorm_ppf(half_normal_base, param.loc, param.scale, param_high),
        )

    elif param.distribution == Distribution.LOG_HALF_NORMAL:
        log_half_normal_base = numpyro.sample(f"{samp_name}_base", dist.Uniform(0, 1))
        param_samp = numpyro.deterministic(
            samp_name,
            10
            ** trunchalfnorm_ppf(
                log_half_normal_base,
                jnp.log10(param_low),
                jnp.log10(param.scale),
                jnp.log10(param_high),
            ),
        )

    return param_samp


def create_reparam_config(template: Template) -> dict:
    """Create reparameterization configuration for parameters."""
    reparam_config = {}

    for prof in template.disk_profiles + template.line_profiles:
        for param in prof.independent:
            samp_name = f"{prof.name}_{param.name}"

            if param.circular:
                reparam_config[f"{samp_name}_base"] = CircularReparam()
            else:
                reparam_config[samp_name] = TransformReparam()

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
        low_std = (param.low - param.loc) / param.scale
        high_std = (param.high - param.loc) / param.scale

        param_samp = numpyro.sample(
            samp_name,
            dist.TransformedDistribution(
                dist.TruncatedNormal(0, 1, low=low_std, high=high_std),
                dist.transforms.AffineTransform(loc=param.loc, scale=param.scale),
            ),
        )

    elif param.distribution == Distribution.LOG_NORMAL:
        low_std = (jnp.log(param.low) - jnp.log(param.loc)) / jnp.log(param.scale)
        high_std = (jnp.log(param.high) - jnp.log(param.loc)) / jnp.log(param.scale)

        param_samp = numpyro.sample(
            samp_name,
            dist.TransformedDistribution(
                dist.TruncatedNormal(0, 1, low=low_std, high=high_std),
                [
                    dist.transforms.AffineTransform(
                        loc=jnp.log(param.loc), scale=jnp.log(param.scale)
                    ),
                    dist.transforms.ExpTransform(),
                ],
            ),
        )

    return param_samp
