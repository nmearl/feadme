import jax.numpy as jnp
import numpy as np

from feadme.sampling.numpyro.model import _eccentricity_apocenter_from_raw


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
