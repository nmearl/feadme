import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from quadax import GaussKronrodRule, ClenshawCurtisRule, quadgk
from .models.disk import integrand

FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
ERR = 1e-5
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value

# fixed_quad = GaussKronrodRule(order=31).integrate
fixed_quad = ClenshawCurtisRule(order=64).integrate

N_xi, N_phi = 30, 50
unit_xi = jnp.linspace(0.0, 1.0, N_xi)
unit_phi = jnp.linspace(0.0, 1.0, N_phi)
XI_u, PHI_u = jnp.meshgrid(unit_xi, unit_phi, indexing="ij")


def _inner_trap(
    log_xi: float,
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
    Inner integral over `phi` for a fixed `xi`.
    """
    xi = 10**log_xi
    phi = jnp.linspace(phi1, phi2, 100)

    result = integrand(phi, xi, X[:, None], inc, sigma, q, e, phi0, nu0)
    return jnp.trapezoid(result, x=phi, axis=-1) * xi * jnp.log(10)


def _inner_quad(
    log_xi: float,
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
    Inner integral over `phi` for a fixed `xi`.
    """
    xi = 10**log_xi

    def transformed_integrand(phi: float, *args) -> ArrayLike:
        return integrand(phi, *args) * xi * jnp.log(10)

    return fixed_quad(
        transformed_integrand, phi1, phi2, args=(xi, X, inc, sigma, q, e, phi0, nu0)
    )[0]


@jax.jit
def quad_jax_integrate(
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
    return fixed_quad(
        _inner_quad,
        jnp.log10(xi1),
        jnp.log10(xi2),
        args=(phi1, phi2, X, inc, sigma, q, e, phi0, nu0),
    )[0]


@jax.jit
def trap_jax_integrate_vec(
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


fixed_phi_quad = ClenshawCurtisRule(order=64).integrate


@jax.jit
def jax_integrate_hybrid(
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

    xi_log = jnp.linspace(
        jnp.log10(xi1), jnp.log10(xi2), N_xi
    )  # Still need many for xi
    xi = 10**xi_log
    jacobian = xi * jnp.log(10)

    def integrate_single_wavelength(x_val):
        def compute_phi_integral(xi_val, jac_val):
            # Fixed-order quadrature on phi (no adaptivity)
            result = fixed_phi_quad(
                lambda p: integrand(p, xi_val, x_val, inc, sigma, q, e, phi0, nu0)
                * jac_val,
                phi1,
                phi2,
                args=(),
            )[0]
            return result

        inner_results = jax.vmap(compute_phi_integral)(xi, jacobian)
        return jnp.trapezoid(inner_results, x=xi_log)

    return jax.vmap(integrate_single_wavelength)(X)


@jax.jit
def trap_jax_integrate_double(
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
    Double trapezoid integration with vmap over wavelengths.
    """
    # Precompute integration grids (same for all wavelengths)
    xi_log = jnp.linspace(jnp.log10(xi1), jnp.log10(xi2), N_xi)
    xi = 10**xi_log
    jacobian = xi * jnp.log(10)
    phi = jnp.linspace(phi1, phi2, N_phi)

    def integrate_single_wavelength(x_val):
        """Compute integral for a single wavelength"""

        def compute_phi_integral(xi_val, jac_val):
            """Integrate over phi for fixed (x_val, xi_val)"""
            # Vectorize integrand evaluation over all phi
            integrand_vals = jax.vmap(
                lambda p: integrand(p, xi_val, x_val, inc, sigma, q, e, phi0, nu0)
            )(phi)

            # Integrate over phi
            return jnp.trapezoid(integrand_vals * jac_val, x=phi)

        # Integrate over phi for each xi, then integrate over xi
        inner_results = jax.vmap(compute_phi_integral)(xi, jacobian)
        return jnp.trapezoid(inner_results, x=xi_log)

    # Process all wavelengths in parallel
    return jax.vmap(integrate_single_wavelength)(X)


@jax.jit
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
    phi = jnp.linspace(phi1, phi2, N_phi)

    def integrate_single_wavelength(x_val):
        """Direct vectorization over (xi, phi) grid"""
        # Create 2D grids for this wavelength
        xi_2d = xi[:, None]  # (N_xi, 1)
        phi_2d = phi[None, :]  # (1, N_phi)
        jac_2d = jacobian[:, None]  # (N_xi, 1)

        # Compute integrand for all (xi, phi) at once - shape (N_xi, N_phi)
        integrand_vals = (
            integrand(phi_2d, xi_2d, x_val, inc, sigma, q, e, phi0, nu0) * jac_2d
        )

        # Double trapezoid integration
        inner = jnp.trapezoid(integrand_vals, x=phi, axis=-1)  # (N_xi,)
        outer = jnp.trapezoid(inner, x=xi_log)  # scalar
        return outer

    # Vmap only over wavelengths
    return jax.vmap(integrate_single_wavelength)(X)


integrator = trap_jax_integrate
