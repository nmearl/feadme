import jax.numpy as jnp

from feadme.models.disk import integrand, quad_jax_integrate, jax_integrate


def calculates_integrand_correctly():
    phi = jnp.pi / 4
    xi_tilde = 2.0
    X = jnp.array([0.1, 0.2, 0.3])
    inc = jnp.pi / 6
    sigma = 0.5
    q = 2.0
    e = 0.1
    phi0 = jnp.pi / 3
    nu0 = 1.0

    result = integrand(phi, xi_tilde, X, inc, sigma, q, e, phi0, nu0)

    assert result.shape == X.shape
    assert (result > 0).all()


def handles_integrand_edge_cases():
    phi = 0.0
    xi_tilde = 1e-10  # Very small radial coordinate
    X = jnp.array([0.0])
    inc = 0.0
    sigma = 1e-10  # Very small line broadening
    q = 0.0
    e = 0.0
    phi0 = 0.0
    nu0 = 1.0

    result = integrand(phi, xi_tilde, X, inc, sigma, q, e, phi0, nu0)

    assert result.shape == X.shape
    assert jnp.isfinite(result).all()


def performs_quad_jax_integrate_correctly():
    xi1 = 1.0
    xi2 = 2.0
    phi1 = 0.0
    phi2 = jnp.pi
    X = jnp.array([0.1, 0.2])
    inc = jnp.pi / 6
    sigma = 0.5
    q = 2.0
    e = 0.1
    phi0 = jnp.pi / 3
    nu0 = 1.0

    result = quad_jax_integrate(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0, nu0)

    assert result.shape == X.shape
    assert (result > 0).all()


def handles_quad_jax_integrate_edge_cases():
    xi1 = 1e-10  # Very small lower radial bound
    xi2 = 1e-5  # Very small upper radial bound
    phi1 = 0.0
    phi2 = jnp.pi
    X = jnp.array([0.0])
    inc = 0.0
    sigma = 1e-10  # Very small line broadening
    q = 0.0
    e = 0.0
    phi0 = 0.0
    nu0 = 1.0

    result = quad_jax_integrate(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0, nu0)

    assert result.shape == X.shape
    assert jnp.isfinite(result).all()


def performs_jax_integrate_correctly():
    xi1 = 1.0
    xi2 = 2.0
    phi1 = 0.0
    phi2 = jnp.pi
    X = jnp.array([0.1, 0.2])
    inc = jnp.pi / 6
    sigma = 0.5
    q = 2.0
    e = 0.1
    phi0 = jnp.pi / 3
    nu0 = 1.0

    result = jax_integrate(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0, nu0)

    assert result.shape == X.shape
    assert (result > 0).all()


def handles_jax_integrate_edge_cases():
    xi1 = 1e-10  # Very small lower radial bound
    xi2 = 1e-5  # Very small upper radial bound
    phi1 = 0.0
    phi2 = jnp.pi
    X = jnp.array([0.0])
    inc = 0.0
    sigma = 1e-10  # Very small line broadening
    q = 0.0
    e = 0.0
    phi0 = 0.0
    nu0 = 1.0

    result = jax_integrate(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0, nu0)

    assert result.shape == X.shape
    assert jnp.isfinite(result).all()
