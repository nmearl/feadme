import jax.numpy as jnp

from feadme.core.disk import integrand
from feadme.core.integrators import mixed_jax_integrate, quad_jax_integrate


def test_calculates_integrand_correctly():
    phi = jnp.pi / 4
    xi_tilde = 500.0
    X = jnp.array([-0.001, 0.0, 0.001])
    inc = jnp.pi / 6
    sigma = 500.0
    q = 2.0
    e = 0.1
    phi0 = jnp.pi / 3

    result = integrand(phi, xi_tilde, X, inc, sigma, q, e, phi0)

    assert result.shape == X.shape
    assert (result > 0).all()


def test_handles_integrand_edge_cases():
    phi = 0.0
    xi_tilde = 100.0
    X = jnp.array([0.0])
    inc = 0.01
    sigma = 500.0
    q = 2.0
    e = 0.0
    phi0 = 0.0

    result = integrand(phi, xi_tilde, X, inc, sigma, q, e, phi0)

    assert result.shape == X.shape
    assert jnp.isfinite(result).all()


def test_performs_quad_jax_integrate_correctly():
    xi1 = 400.0
    xi2 = 5000.0
    phi1 = 0.0
    phi2 = 2.0 * jnp.pi
    X = jnp.array([0.99, 1.0, 1.01])
    inc = jnp.pi / 6
    sigma = 500.0
    q = 2.0
    e = 0.1
    phi0 = jnp.pi / 3

    result = quad_jax_integrate(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0)

    assert result.shape == X.shape
    assert jnp.isfinite(result).all()
    assert (result >= 0).all()


def test_performs_mixed_jax_integrate_correctly():
    xi1 = 400.0
    xi2 = 5000.0
    phi1 = 0.0
    phi2 = 2.0 * jnp.pi
    X = jnp.array([0.99, 1.0, 1.01])
    inc = jnp.pi / 6
    sigma = 500.0
    q = 2.0
    e = 0.1
    phi0 = jnp.pi / 3

    result = mixed_jax_integrate(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0)

    assert result.shape == X.shape
    assert jnp.isfinite(result).all()
    assert (result >= 0).all()
