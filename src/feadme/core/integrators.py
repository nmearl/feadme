from functools import partial

import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from quadax import ClenshawCurtisRule, GaussKronrodRule

from .disk import integrand, normalization_integrand

FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
ERR = 1e-5
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value

CC_RES = 32
GK_RES = 61

fixed_quad_xi = ClenshawCurtisRule(order=CC_RES).integrate
fixed_quad_phi = ClenshawCurtisRule(order=CC_RES * 4).integrate
# fixed_quad_xi = GaussKronrodRule(order=GK_RES).integrate
# fixed_quad_phi = GaussKronrodRule(order=GK_RES).integrate

N_xi, N_phi = 32, 128

LN10 = jnp.log(10.0)


@partial(jax.jit, static_argnums=(2, 3, 10))  # phi1/phi2 fixed -> helps compilation
def split_quad_jax_integrate(
    xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0, n_split: int = 4
):
    log_xi1, log_xi2 = jnp.log10(xi1), jnp.log10(xi2)

    # Split integration over phi into n_split segments
    dphi = (phi2 - phi1) / n_split

    def integrate_over_phi(log_xi):
        xi_tilde = 10.0**log_xi

        def f(phi):
            return integrand(phi, xi_tilde, X, inc, sigma, q, e, phi0)

        def body(i, acc):
            a = phi1 + i * dphi
            b = a + dphi
            return acc + fixed_quad_phi(f, a, b, args=())[0]

        v = jax.lax.fori_loop(0, n_split, body, jnp.zeros_like(X))
        return v * (xi_tilde * LN10)

    return fixed_quad_xi(integrate_over_phi, log_xi1, log_xi2, args=())[0]


@partial(jax.jit, static_argnums=(2, 3, 9))
def split_quad_scalar_integrate(
    xi1, xi2, phi1, phi2, inc, sigma, q, e, phi0, n_split: int = 4
):
    log_xi1, log_xi2 = jnp.log10(xi1), jnp.log10(xi2)
    dphi = (phi2 - phi1) / n_split

    def integrate_over_phi(log_xi):
        xi_tilde = 10.0**log_xi

        def f(phi):
            return normalization_integrand(phi, xi_tilde, inc, sigma, q, e, phi0)

        def body(i, acc):
            a = phi1 + i * dphi
            b = a + dphi
            return acc + fixed_quad_phi(f, a, b, args=())[0]

        v = jax.lax.fori_loop(0, n_split, body, jnp.asarray(0.0))
        return v * (xi_tilde * LN10)

    return fixed_quad_xi(integrate_over_phi, log_xi1, log_xi2, args=())[0]


@partial(jax.jit, static_argnums=(2, 3))  # keep if phi1/phi2 fixed
def quad_jax_integrate(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0):
    log_xi1, log_xi2 = jnp.log10(xi1), jnp.log10(xi2)

    def integrate_over_phi(log_xi):
        xi_tilde = 10.0**log_xi
        # vector-valued integrand over X
        # output shape: (n_wave,)
        val = fixed_quad_phi(
            lambda phi: integrand(phi, xi_tilde, X, inc, sigma, q, e, phi0),
            phi1,
            phi2,
            args=(),
        )[0]
        # Jacobian for dxi = xi ln(10) dlog10(xi)
        return val * (xi_tilde * LN10)

    # This integrates a vector-valued function over log_xi
    return fixed_quad_xi(integrate_over_phi, log_xi1, log_xi2, args=())[0]


@partial(jax.jit, static_argnums=(2, 3))
def quad_scalar_integrate(xi1, xi2, phi1, phi2, inc, sigma, q, e, phi0):
    log_xi1, log_xi2 = jnp.log10(xi1), jnp.log10(xi2)

    def integrate_over_phi(log_xi):
        xi_tilde = 10.0**log_xi
        val = fixed_quad_phi(
            lambda phi: normalization_integrand(phi, xi_tilde, inc, sigma, q, e, phi0),
            phi1,
            phi2,
            args=(),
        )[0]
        return val * (xi_tilde * LN10)

    return fixed_quad_xi(integrate_over_phi, log_xi1, log_xi2, args=())[0]


@partial(jax.jit, static_argnums=(2, 3))
def vmap_quad_jax_integrate(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0):
    """
    Double fixed-order quadrature with explicit wavelength vectorization.
    """

    def integrate_single_wavelength(x_val):
        """Compute integral for one wavelength"""

        def inner_quad_func(log_xi):
            xi = 10**log_xi

            def transformed_integrand(phi):
                return (
                    integrand(phi, xi, x_val, inc, sigma, q, e, phi0) * xi * jnp.log(10)
                )

            return fixed_quad_phi(transformed_integrand, phi1, phi2, args=())[0]

        return fixed_quad_xi(inner_quad_func, jnp.log10(xi1), jnp.log10(xi2), args=())[
            0
        ]

    # Vectorize over all wavelengths in parallel
    return jax.vmap(integrate_single_wavelength)(X)


@partial(jax.jit, static_argnums=(2, 3))
def trap_jax_integrate(
    xi1: float,
    xi2: float,
    phi1: float,
    phi2: float,
    X: ArrayLike,
    inc: float,
    sigma: float,
    q: float,
    e: float,
    phi0: float,
) -> ArrayLike:
    xi_log = jnp.linspace(jnp.log10(xi1), jnp.log10(xi2), N_xi)
    xi = 10.0**xi_log
    dphi = (phi2 - phi1) / N_phi
    phi = phi1 + (jnp.arange(N_phi) + 0.5) * dphi  # midpoint grid: avoids phi=0/2pi

    xi_3d = xi[:, None, None]
    phi_3d = phi[None, :, None]
    X_3d = jnp.asarray(X)[None, None, :]

    # Jacobian for dxi = xi ln(10) dlog10(xi)
    jac_3d = (xi * jnp.log(10.0))[:, None, None]

    vals = integrand(phi_3d, xi_3d, X_3d, inc, sigma, q, e, phi0) * jac_3d

    # Midpoint Riemann sum over phi: full [0, 2pi] coverage, no endpoint singularity
    inner = jnp.sum(vals * dphi, axis=1)  # (N_xi, n_wave)
    outer = jnp.trapezoid(inner, x=xi_log, axis=0)  # (n_wave,)

    return outer


@partial(jax.jit, static_argnums=(2, 3))
def trap_scalar_integrate(
    xi1: float,
    xi2: float,
    phi1: float,
    phi2: float,
    inc: float,
    sigma: float,
    q: float,
    e: float,
    phi0: float,
) -> ArrayLike:
    xi_log = jnp.linspace(jnp.log10(xi1), jnp.log10(xi2), N_xi)
    xi = 10.0**xi_log
    dphi = (phi2 - phi1) / N_phi
    phi = phi1 + (jnp.arange(N_phi) + 0.5) * dphi

    xi_2d = xi[:, None]
    phi_2d = phi[None, :]
    jac_2d = (xi * jnp.log(10.0))[:, None]

    vals = normalization_integrand(phi_2d, xi_2d, inc, sigma, q, e, phi0) * jac_2d
    inner = jnp.sum(vals * dphi, axis=1)
    outer = jnp.trapezoid(inner, x=xi_log, axis=0)
    return outer


@partial(jax.jit, static_argnums=(2, 3))
def mixed_jax_integrate(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0):
    dphi = (phi2 - phi1) / N_phi
    phi = phi1 + (jnp.arange(N_phi) + 0.5) * dphi  # midpoint grid: avoids phi=0/2pi
    log_xi1 = jnp.log10(xi1)
    log_xi2 = jnp.log10(xi2)

    def integrand_over_log_xi(log_xi):
        xi = 10.0**log_xi
        jacobian_xi = xi * jnp.log(10.0)

        vals_phi = integrand(
            phi[:, None],  # (N_phi, 1)
            xi,  # scalar
            X[None, :],  # (1, n_wave)
            inc,
            sigma,
            q,
            e,
            phi0,
        )  # -> (N_phi, n_wave)

        # Midpoint Riemann sum over phi: full [0, 2pi] coverage, no endpoint singularity
        inner_phi = jnp.sum(vals_phi * dphi, axis=0)  # (n_wave,)
        return inner_phi * jacobian_xi

    return fixed_quad_xi(
        integrand_over_log_xi,
        log_xi1,
        log_xi2,
        args=(),
    )[0]


@partial(jax.jit, static_argnums=(2, 3))
def mixed_scalar_integrate(xi1, xi2, phi1, phi2, inc, sigma, q, e, phi0):
    dphi = (phi2 - phi1) / N_phi
    phi = phi1 + (jnp.arange(N_phi) + 0.5) * dphi
    log_xi1 = jnp.log10(xi1)
    log_xi2 = jnp.log10(xi2)

    def integrand_over_log_xi(log_xi):
        xi = 10.0**log_xi
        jacobian_xi = xi * jnp.log(10.0)
        vals_phi = normalization_integrand(phi, xi, inc, sigma, q, e, phi0)
        inner_phi = jnp.sum(vals_phi * dphi, axis=0)
        return inner_phi * jacobian_xi

    return fixed_quad_xi(
        integrand_over_log_xi,
        log_xi1,
        log_xi2,
        args=(),
    )[0]


SCALAR_NORMALIZATION_INTEGRATORS = {
    quad_jax_integrate: quad_scalar_integrate,
    mixed_jax_integrate: mixed_scalar_integrate,
    trap_jax_integrate: trap_scalar_integrate,
    split_quad_jax_integrate: split_quad_scalar_integrate,
}


def get_scalar_normalization_integrator(integrator):
    return SCALAR_NORMALIZATION_INTEGRATORS.get(integrator, quad_scalar_integrate)
