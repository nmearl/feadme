from typing import Any, Dict

import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from numpyro.distributions import transforms as T
from numpyro.infer.reparam import TransformReparam
from ..parser import Distribution, Parameter

TAU = 2.0 * jnp.pi


def create_reparam_config(template) -> Dict[str, Any]:
    reparam_config: Dict[str, Any] = {}

    for prof in template.disk_profiles + template.line_profiles:
        for param in prof.independent:
            name = param.qualified_name

            if param.circular:
                continue

            reparam_config[name] = TransformReparam()

    return reparam_config


def _logit(p: jnp.ndarray) -> jnp.ndarray:
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p) - jnp.log1p(-p)


def _mean_std_to_lognormal_params(
    mean: jnp.ndarray, std: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Interpret mean/std in *linear* space and convert to (mu_log, sigma_log) of log-normal.
    """
    sigma_log = jnp.sqrt(jnp.log1p((std / mean) ** 2))
    mu_log = jnp.log(mean) - 0.5 * sigma_log**2
    return mu_log, sigma_log


def _uniform_affine(low: float, high: float) -> dist.Distribution:
    # u ~ Uniform(0,1), x = low + (high-low)*u
    return dist.TransformedDistribution(
        dist.Uniform(0.0, 1.0),
        T.AffineTransform(loc=low, scale=(high - low)),
    )


def _loguniform_affine(low: float, high: float) -> dist.Distribution:
    # u ~ Uniform(0,1), z = loglow + (loghigh-loglow)*u, x = exp(z)
    loglow, loghigh = jnp.log(low), jnp.log(high)
    return dist.TransformedDistribution(
        dist.Uniform(0.0, 1.0),
        [
            T.AffineTransform(loc=loglow, scale=(loghigh - loglow)),
            T.ExpTransform(),
        ],
    )


def _bounded_normal_loc_scale(
    low: float, high: float, loc: float, scale: float
) -> dist.Distribution:
    """
    Logistic-normal on [low, high] whose typical set is approximately centered near loc
    with approximately local scale 'scale' (via sigmoid linearization).

    This avoids TwoSidedTruncatedDistribution while still respecting loc/scale *roughly*.
    """
    span = high - low
    p0 = (loc - low) / span
    z_loc = _logit(p0)

    # local slope of sigmoid at z_loc is p0*(1-p0)
    s = jnp.clip(p0 * (1.0 - p0), 1e-3, jnp.inf)
    z_scale = jnp.clip(scale / (span * s), 1e-3, 10.0)

    return dist.TransformedDistribution(
        dist.Normal(0.0, 1.0),
        [
            T.AffineTransform(loc=z_loc, scale=z_scale),
            T.SigmoidTransform(),
            T.AffineTransform(loc=low, scale=span),
        ],
    )


def _bounded_lognormal_mean_std(
    low: float, high: float, mean: float, std: float
) -> dist.Distribution:
    """
    Bounded lognormal-like prior on [low, high] using a smooth transform:
      z ~ Normal(0,1)
      y = mu_log + sigma_log * z
      u = sigmoid(y)
      t = log_low + (log_high-log_low)*u
      x = exp(t) in [low, high]

    This is not an exact truncated lognormal, but is HMC-friendly and avoids
    TwoSidedTruncatedDistribution (required for TransformReparam compatibility).
    """
    mu_log, sigma_log = _mean_std_to_lognormal_params(mean, std)

    log_low, log_high = jnp.log(low), jnp.log(high)
    span = log_high - log_low

    # Optional: guard against absurdly wide implied sigma in log-space (helps saturation).
    sigma_log = jnp.clip(sigma_log, 1e-3, 5.0)

    return dist.TransformedDistribution(
        dist.Normal(0.0, 1.0),
        [
            T.AffineTransform(loc=mu_log, scale=sigma_log),
            T.SigmoidTransform(),
            T.AffineTransform(loc=log_low, scale=span),
            T.ExpTransform(),
        ],
    )


def _unbounded_lognormal_mean_std(mean: float, std: float) -> dist.Distribution:
    """
    Unbounded lognormal with mean/std interpreted in linear space.
    """
    mu_log, sigma_log = _mean_std_to_lognormal_params(mean, std)
    sigma_log = jnp.clip(sigma_log, 1e-6, 10.0)
    return dist.TransformedDistribution(
        dist.Normal(0.0, 1.0),
        [T.AffineTransform(loc=mu_log, scale=sigma_log), T.ExpTransform()],
    )


def _sample_circular(samp_name: str, low: float, high: float) -> jnp.ndarray:
    """
    Sample a circular angle via a 2D Normal direction:
      xy ~ Normal(0,1)^2
      theta = atan2(y,x)  (uniform direction)
    Always sampleable under Predictive.
    """
    xy = numpyro.sample(f"{samp_name}_base", dist.Normal(0.0, 1.0).expand([2]))
    theta = jnp.arctan2(xy[1], xy[0])  # in (-pi, pi]
    theta = jnp.mod(theta, TAU)  # [0, 2pi)

    numpyro.deterministic(samp_name, theta)
    return theta


def sample_param(samp_name: str, param: Parameter) -> jnp.ndarray:
    if param.circular:
        return _sample_circular(samp_name, param.low, param.high)

    if param.distribution == Distribution.UNIFORM:
        return numpyro.sample(samp_name, _uniform_affine(param.low, param.high))

    if param.distribution == Distribution.LOG_UNIFORM:
        return numpyro.sample(samp_name, _loguniform_affine(param.low, param.high))

    if param.distribution == Distribution.NORMAL:
        # Requires bounds for this implementation
        return numpyro.sample(
            samp_name,
            _bounded_normal_loc_scale(param.low, param.high, param.loc, param.scale),
        )

    if param.distribution == Distribution.LOG_NORMAL:
        return numpyro.sample(
            samp_name,
            _bounded_lognormal_mean_std(param.low, param.high, param.loc, param.scale),
        )

    raise ValueError(
        f"Unsupported distribution type for auto reparam: {param.distribution!r}"
    )
