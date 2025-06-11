import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
from jax.typing import ArrayLike
from quadax import GaussKronrodRule, quadgk

FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
ERR = 1e-5
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value

fixed_quadgk51 = GaussKronrodRule(order=51).integrate
fixed_quadgk31 = GaussKronrodRule(order=31).integrate

N_xi, N_phi = 30, 50
unit_xi = jnp.linspace(0.0, 1.0, N_xi)
unit_phi = jnp.linspace(0.0, 1.0, N_phi)
XI_u, PHI_u = jnp.meshgrid(unit_xi, unit_phi, indexing="ij")


def doppler_factor(
    xi: float, phi: float, inc: float, e: float, phi0: float
) -> jnp.ndarray:
    """
    Calculate the Doppler factor for a given `xi`, `phi`, `inc`, `e`, and `phi0`.
    """
    sini = jnp.sin(inc)
    sinphi = jnp.sin(phi)
    cosphi = jnp.cos(phi)
    sinphiphinot = jnp.sin(phi - phi0)
    cosphiphinot = jnp.cos(phi - phi0)
    scale = 1 - 2 / xi

    # Eracleous et al, eq 6
    b_div_r = jnp.sqrt(1 - sini**2 * cosphi**2) * (
        1 + (1 / xi) * (1 - sini * cosphi) / (1 + sini * cosphi)
    )

    # Eracleous et al, eq 16
    gamma = (
        1
        - (e**2 * sinphiphinot**2 + scale * (1 - e * cosphiphinot) ** 2)
        / (xi * scale**2 * (1 - e * cosphiphinot))
    ) ** -0.5

    term_binner = 1 - b_div_r**2 * scale
    term_binner = jnp.where(term_binner < 0, 0, term_binner)

    # Eracleous et al, eq 15
    da = scale**-0.5
    db = term_binner**0.5 * e * sinphiphinot
    dc = xi**0.5 * scale ** (3 / 2) * (1 - e * cosphiphinot) ** 0.5
    dd = b_div_r * (1 - e * cosphiphinot) ** 0.5 * sini * sinphi
    de = xi**0.5 * scale**0.5 * (1 - sini**2 * cosphi**2) ** 0.5
    de = jnp.where(de == 0.0, FLOAT_EPSILON, de)

    inv_dop = gamma * (da - db / dc + dd / de)

    return inv_dop**-1


def intensity(
    xi: float, X: ArrayLike, D: float, sigma: float, q: float, nu0: float
) -> jnp.ndarray:
    """
    `I_nu` function for the disk model, as defined in Eracleous et al. (1995).
    """
    # Eracleous et al, eq 18; returned units are erg / cm^2
    # exponent = -((1 + X - D) ** 2) / (2 * D**2) * (c_cgs / sigma) ** 2
    exponent = -((1 + X - D) ** 2 * nu0**2) / (2 * D**2 * sigma**2)
    exponent = jnp.clip(exponent, min=-37)

    # res = (xi**-q * c_cgs) / (jnp.sqrt(2 * jnp.pi) * sigma) * jnp.exp(exponent)
    res = (xi**-q) / (jnp.sqrt(2 * jnp.pi) * sigma) * jnp.exp(exponent)

    return res


def Psi(xi: float, phi: float, inc: float) -> jnp.ndarray:
    """
    `Psi` function for the disk model, as defined in Eracleous et al. (1995).
    """
    # Eracleous et al, eq 8
    return 1 + (1 / xi) * (
        (1 - jnp.sin(inc) * jnp.cos(phi)) / (1 + jnp.sin(inc) * jnp.cos(phi))
    )


def integrand(
    phi: float,
    xi_tilde: float,
    X: jnp.ndarray | np.ndarray,
    inc: float,
    sigma: float,
    q: float,
    e: float,
    phi0: float,
    nu0: float,
) -> jnp.ndarray:
    """
    Integrand for the double integral over `xi` and `phi`.
    """
    # Eracleous et al, eq 10
    trans_fac = (1 + e) / (1 - e * jnp.cos(phi - phi0))
    xi = xi_tilde * trans_fac

    D = doppler_factor(xi, phi, inc, e, phi0)
    I_nu = intensity(xi, X, D, sigma, q, nu0)

    # Eracleous et al, eq 7
    res = xi * I_nu * D**3 * Psi(xi, phi, inc) * trans_fac

    return res


def _inner_quad(
    xi: float,
    phi1: float,
    phi2: float,
    X: jnp.ndarray | np.ndarray,
    inc: float,
    sigma: float,
    q: float,
    e: float,
    phi0: float,
    nu0: float,
) -> jnp.ndarray:
    """
    Inner integral over `phi` for a fixed `xi`.
    """
    return fixed_quadgk51(
        integrand, phi1, phi2, args=(xi, X, inc, sigma, q, e, phi0, nu0)
    )[0]


@jax.jit
def quad_jax_integrate(
    xi1: float,
    xi2: float,
    phi1: float,
    phi2: float,
    X: jnp.ndarray | np.ndarray,
    inc: float,
    sigma: float,
    q: float,
    e: float,
    phi0: float,
    nu0: float,
) -> jnp.ndarray:
    """
    Perform a double integral over `xi` and `phi` using Gauss-Kronrod quadrature.
    """
    return fixed_quadgk31(
        _inner_quad, xi1, xi2, args=(phi1, phi2, X, inc, sigma, q, e, phi0, nu0)
    )[0]


@jax.jit
def jax_integrate(
    xi1: float,
    xi2: float,
    phi1: float,
    phi2: float,
    X: jnp.ndarray | np.ndarray,
    inc: float,
    sigma: float,
    q: float,
    e: float,
    phi0: float,
    nu0: float,
) -> jnp.ndarray:
    """
    Perform a double integral over `xi` and `phi` using trapezoidal rule.
    """
    xi = jnp.exp(jnp.log(xi1) + (jnp.log(xi2) - jnp.log(xi1)) * XI_u)
    phi = phi1 + (phi2 - phi1) * PHI_u

    res = jax.vmap(
        lambda phi_arr, xi_arr: integrand(
            phi_arr, xi_arr, X[:, None], inc, sigma, q, e, phi0, nu0
        ),
        in_axes=(0, 0),
    )(phi, xi)

    inner_integral = jnp.trapezoid(res, x=phi[0], axis=2)
    outer_integral = jnp.trapezoid(inner_integral, x=xi[:, 0], axis=0)

    return outer_integral
