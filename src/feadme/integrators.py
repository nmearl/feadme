import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from quadax import (
    GaussKronrodRule,
    ClenshawCurtisRule,
    quadgk,
    TanhSinhRule,
    quadcc,
    trapezoid,
)
from .models.disk import integrand
from functools import partial

FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
ERR = 1e-5
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value

fixed_quad_xi = ClenshawCurtisRule(order=64).integrate
fixed_quad_phi = ClenshawCurtisRule(order=128).integrate
# fixed_quad_xi = GaussKronrodRule(order=61).integrate
# fixed_quad_phi = GaussKronrodRule(order=61).integrate
# fixed_quad_xi = TanhSinhRule(order=63).integrate
# fixed_quad_phi = TanhSinhRule(order=127).integrate

N_xi, N_phi = 64, 128
unit_xi = jnp.linspace(0.0, 1.0, N_xi)
unit_phi = jnp.linspace(0.0, 1.0, N_phi)
XI_u, PHI_u = jnp.meshgrid(unit_xi, unit_phi, indexing="ij")


@partial(jax.jit, static_argnums=(2, 3))
def quad_jax_integrate(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0, nu0):
    """
    Double fixed-order quadrature with explicit wavelength vectorization.
    Uses fixed [0, 1] limits for outer integral via linear transformation.
    """
    log_xi1 = jnp.log10(xi1)
    log_xi2 = jnp.log10(xi2)
    delta_log_xi = log_xi2 - log_xi1

    def integrate_single_wavelength(x_val):
        """Compute integral for one wavelength"""

        def inner_quad_func(t):
            # Transform t ∈ [0, 1] to log_xi ∈ [log_xi1, log_xi2]
            log_xi = log_xi1 + t * delta_log_xi
            xi = 10**log_xi

            def transformed_integrand(phi):
                return (
                    integrand(phi, xi, x_val, inc, sigma, q, e, phi0, nu0)
                    * xi
                    * jnp.log(10)
                )

            # Include Jacobian from transformation: d(log_xi) = delta_log_xi * dt
            return (
                fixed_quad_phi(transformed_integrand, phi1, phi2, args=())[0]
                * delta_log_xi
            )

        return fixed_quad_xi(inner_quad_func, 0.0, 1.0, args=())[0]

    return jax.vmap(integrate_single_wavelength)(X)


@partial(jax.jit, static_argnums=(2, 3))
def __quad_jax_integrate(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0, nu0):
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
    phi = jnp.linspace(phi1, phi2, N_phi, endpoint=True)

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
        inner = trapezoid(integrand_vals, x=phi, axis=-1)  # (N_xi,)
        # d_phi = (phi2 - phi1) / N_phi
        # inner = d_phi * jnp.sum(integrand_vals, axis=-1)
        outer = trapezoid(inner, x=xi_log)  # scalar
        return outer

    # Vmap only over wavelengths
    return jax.vmap(integrate_single_wavelength)(X)


PHI_GRID = jnp.linspace(0.0, 2 * jnp.pi, N_phi, endpoint=True)


@partial(jax.jit, static_argnums=(2, 3))
def mixed_jax_integrate(
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
    Double integral over xi and phi with:
      - Clenshaw–Curtis (fixed_quad_xi) over log10(xi)
      - Trapezoid rule over phi

    Parameters
    ----------
    xi1, xi2 : float
        Inner and outer radii (in xi units).
    phi1, phi2 : float
        Azimuthal limits [rad].
    X : ArrayLike
        Wavelength (or frequency) array. Shape (N_lambda,).
    inc, sigma, q, e, phi0, nu0 : float
        Disk / line profile parameters, passed through to `integrand`.

    Returns
    -------
    ArrayLike
        Integrated flux at each wavelength. Shape (N_lambda,).
    """
    log_xi1 = jnp.log10(xi1)
    log_xi2 = jnp.log10(xi2)
    delta_log_xi = log_xi2 - log_xi1

    def integrate_single_wavelength(x_val: float) -> float:
        """Compute the double integral for a single wavelength x_val."""

        def integrand_over_t(t: float) -> float:
            """
            Outer integrand as a function of t ∈ [0, 1], after
            integrating over phi with a trapezoid rule.
            """
            # Transform t ∈ [0, 1] to log_xi ∈ [log_xi1, log_xi2]
            log_xi = log_xi1 + t * delta_log_xi
            xi = 10.0**log_xi
            jacobian_xi = xi * jnp.log(10.0)

            # Evaluate model integrand for all phi at this xi, x_val
            vals_phi = integrand(PHI_GRID, xi, x_val, inc, sigma, q, e, phi0, nu0)

            # Trapezoid over phi
            inner_phi = jnp.trapezoid(vals_phi, x=PHI_GRID)

            # Include both Jacobians: d(log_xi)/dt and d(xi)/d(log_xi)
            return inner_phi * jacobian_xi * delta_log_xi

        # Clenshaw–Curtis over fixed [0, 1] interval
        result = fixed_quad_xi(integrand_over_t, 0.0, 1.0, args=())[0]

        return result

    # Vectorize over wavelengths
    return jax.vmap(integrate_single_wavelength)(X)


integrator = mixed_jax_integrate
