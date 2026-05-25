import jax.numpy as jnp
import jax.random as random
import numpy as np

from feadme.core.parser import Distribution, Parameter
from feadme.sampling.numpyro.model import (
    _eccentricity_apocenter_from_raw,
    _inclination_distribution_from_param,
)


def test_eccentricity_apocenter_transform_respects_template_bounds():
    z_h = jnp.array([0.0, 1.0, 10.0, -2.0])
    z_k = jnp.array([0.0, 1.0, 0.0, 3.0])
    low = 0.1
    high = 0.95

    eccentricity, apocenter, log_abs_det = _eccentricity_apocenter_from_raw(
        z_h, z_k, low, high
    )

    assert np.all(np.asarray(eccentricity) >= low)
    assert np.all(np.asarray(eccentricity) < high)
    assert np.all(np.asarray(apocenter) >= 0.0)
    assert np.all(np.asarray(apocenter) < 2.0 * np.pi)
    assert np.all(np.isfinite(np.asarray(log_abs_det)))


def test_beta_inclination_prior_is_mapped_to_cosine_interval():
    param = Parameter(
        distribution=Distribution.BETA,
        low=0.0,
        high=np.pi / 2 - 1e-3,
        loc=np.pi / 4,
        scale=np.pi / 8,
        alpha=2.0,
        beta=2.0,
    )

    prior = _inclination_distribution_from_param(param, param.low, param.high)
    samples = prior.sample(random.PRNGKey(0), sample_shape=(4096,))
    mu_min = np.cos(param.high)
    mu_max = np.cos(param.low)

    assert np.all(np.asarray(samples) >= mu_min)
    assert np.all(np.asarray(samples) <= mu_max)
    assert np.isclose(float(samples.mean()), 0.5 * (mu_min + mu_max), atol=2e-2)
