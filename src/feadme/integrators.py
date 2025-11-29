import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from quadax import GaussKronrodRule, ClenshawCurtisRule, quadgk, TanhSinhRule
from .models.disk import integrand
from functools import partial

FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
ERR = 1e-5
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value

# fixed_quad_xi = ClenshawCurtisRule(order=48).integrate
# fixed_quad_phi = ClenshawCurtisRule(order=96).integrate
fixed_quad_xi = GaussKronrodRule(order=61).integrate
fixed_quad_phi = GaussKronrodRule(order=61).integrate
# fixed_quad_xi = TanhSinhRule(order=63).integrate
# fixed_quad_phi = TanhSinhRule(order=127).integrate

N_xi, N_phi = 48, 96
unit_xi = jnp.linspace(0.0, 1.0, N_xi)
unit_phi = jnp.linspace(0.0, 1.0, N_phi)
XI_u, PHI_u = jnp.meshgrid(unit_xi, unit_phi, indexing="ij")


@jax.jit
def quad_jax_integrate_vectorized(
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
    nu0: float,
) -> ArrayLike:
    """
    Perform a double integral over `xi` and `phi` using Gauss-Kronrod quadrature.
    """

    def inner_quad_func(log_xi: float, phi1, phi2, X, inc, sigma, q, e, phi0, nu0):
        xi = 10**log_xi

        def transformed_integrand(phi: float, *args) -> ArrayLike:
            return integrand(phi, *args) * xi * jnp.log(10)

        return fixed_quad_phi(
            transformed_integrand, phi1, phi2, args=(xi, X, inc, sigma, q, e, phi0, nu0)
        )[0]

    return fixed_quad_xi(
        inner_quad_func,
        jnp.log10(xi1),
        jnp.log10(xi2),
        args=(phi1, phi2, X, inc, sigma, q, e, phi0, nu0),
    )[0]


@partial(jax.jit, static_argnums=(2, 3))
def quad_jax_integrate(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0, nu0):
    """
    Double fixed-order quadrature with explicit wavelength vectorization.
    """

    def integrate_single_wavelength(x_val):
        """Compute integral for one wavelength"""

        def inner_quad_func(log_xi):
            xi = 10**log_xi

            def transformed_integrand(phi):
                return (
                    integrand(phi, xi, x_val, inc, sigma, q, e, phi0, nu0)
                    * xi
                    * jnp.log(10)
                )

            return fixed_quad_phi(transformed_integrand, phi1, phi2, args=())[0]

        return fixed_quad_xi(inner_quad_func, jnp.log10(xi1), jnp.log10(xi2), args=())[
            0
        ]

    # Vectorize over all wavelengths in parallel
    return jax.vmap(integrate_single_wavelength)(X)


@partial(jax.jit, static_argnums=(2, 3))
def quad_jax_integrate_hybrid(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0, nu0):
    """
    Trapezoid integration for outer integral over xi,
    fixed-order quadrature for inner integral over phi,
    with explicit wavelength vectorization.
    """
    # Create grid for trapezoid integration over xi
    xi_log = jnp.linspace(jnp.log10(xi1), jnp.log10(xi2), N_xi)

    def integrate_single_wavelength(x_val):
        """Compute integral for one wavelength"""

        def evaluate_at_log_xi(log_xi):
            """Evaluate inner integral at a specific log_xi point"""
            xi = 10**log_xi

            def transformed_integrand(phi):
                return (
                    integrand(phi, xi, x_val, inc, sigma, q, e, phi0, nu0)
                    * xi
                    * jnp.log(10)
                )

            # Use fixed-order quadrature for the inner integral over phi
            return fixed_quad_phi(transformed_integrand, phi1, phi2, args=())[0]

        # Evaluate inner integral at all xi_log points using vmap
        inner_results = jax.vmap(evaluate_at_log_xi)(xi_log)

        # Apply trapezoid rule for outer integral over xi
        return jnp.trapezoid(inner_results, x=xi_log)

    # Vectorize over all wavelengths in parallel
    return jax.vmap(integrate_single_wavelength)(X)


@partial(jax.jit, static_argnums=(2, 3))
def trap_jax_integrate_vectorized(
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
    nu0: float,
) -> ArrayLike:
    """
    Perform a double integral over `xi` and `phi` using trapezoidal rule.
    Uses explicit vectorization.
    """
    # Create 1D arrays for integration points
    xi_log = jnp.linspace(jnp.log10(xi1), jnp.log10(xi2), N_xi)
    phi = jnp.linspace(phi1, phi2, N_phi)

    xi = 10**xi_log
    jacobian = xi * jnp.log(10)

    # Vectorized computation over both xi and phi simultaneously
    # This broadcasts xi (N_xi,) and phi (N_phi,) to create (N_xi, N_phi) arrays
    xi_2d = xi[:, None]  # Shape: (N_xi, 1)
    phi_2d = phi[None, :]  # Shape: (1, N_phi)
    jac_2d = jacobian[:, None]  # Shape: (N_xi, 1)

    # Compute integrand for all (xi, phi) combinations at once
    # X shape: (168,), xi_2d: (N_xi, 1), phi_2d: (1, N_phi)
    # Need to add dimensions to X to broadcast with (N_xi, N_phi)
    X_expanded = X[:, None, None]  # Shape: (168, 1, 1)

    integrand_vals = (
        integrand(phi_2d, xi_2d, X_expanded, inc, sigma, q, e, phi0, nu0) * jac_2d
    )

    # Integrate using trapezoidal rule
    inner_integral = jnp.trapezoid(integrand_vals, x=phi, axis=-1)  # Shape: (168, N_xi)
    outer_integral = jnp.trapezoid(inner_integral, x=xi_log, axis=-1)  # Shape: (168,)

    return outer_integral


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
    nu0: float,
) -> ArrayLike:
    """
    Vmap over wavelengths, direct vectorization for (xi, phi) grid.
    """
    # Precompute grids (shared across wavelengths)
    xi_log = jnp.linspace(jnp.log10(xi1), jnp.log10(xi2), N_xi)
    xi = 10**xi_log
    jacobian = xi * jnp.log(10)
    phi = jnp.linspace(phi1, phi2, N_phi, endpoint=False)

    def integrate_single_wavelength(x_val):
        # Create 2D grids for this wavelength
        xi_2d = xi[:, None]  # (N_xi, 1)
        phi_2d = phi[None, :]  # (1, N_phi)
        jac_2d = jacobian[:, None]  # (N_xi, 1)

        # Compute integrand for all (xi, phi) at once - shape (N_xi, N_phi)
        integrand_vals = (
            integrand(phi_2d, xi_2d, x_val, inc, sigma, q, e, phi0, nu0) * jac_2d
        )

        # Double trapezoid integration
        # inner = jnp.trapezoid(integrand_vals, x=phi, axis=-1)  # (N_xi,)
        d_phi = (phi2 - phi1) / N_phi
        inner = d_phi * jnp.sum(integrand_vals, axis=-1)
        outer = jnp.trapezoid(inner, x=xi_log)  # scalar
        return outer

    # Vmap only over wavelengths
    return jax.vmap(integrate_single_wavelength)(X)


integrator = quad_jax_integrate
