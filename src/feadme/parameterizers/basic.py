import astropy.constants as const
import astropy.units as u
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


def sample_param(samp_name: str, param: Parameter) -> ArrayLike:
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
