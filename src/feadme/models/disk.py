import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from quadax import GaussKronrodRule, ClenshawCurtisRule, quadgk

FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
ERR = 1e-5
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value


def doppler_factor(
    xi: float, phi: float, inc: float, e: float, phi0: float
) -> ArrayLike:
    """
    Calculate the Doppler factor for a given `xi`, `phi`, `inc`, `e`, and `phi0`.
    """
    sini = jnp.sin(inc)
    sinphi = jnp.sin(phi)
    cosphi = jnp.cos(phi)
    sinphiphinot = jnp.sin(phi - phi0)
    cosphiphinot = jnp.cos(phi - phi0)
    scale = 1 - 2 / xi
    one_minus_sinisq_cosphisq = 1 - sini**2 * cosphi**2
    one_minus_sinisq_cosphisq = jnp.where(
        one_minus_sinisq_cosphisq < FLOAT_EPSILON,
        FLOAT_EPSILON,
        one_minus_sinisq_cosphisq,
    )
    one_minus_e_cosphiphinot = 1 - e * cosphiphinot
    one_minus_e_cosphiphinot = jnp.where(
        one_minus_e_cosphiphinot < FLOAT_EPSILON,
        FLOAT_EPSILON,
        one_minus_e_cosphiphinot,
    )

    # Eracleous et al, eq 6
    b_div_r = jnp.sqrt(one_minus_sinisq_cosphisq) * (
        1 + (1 / xi) * (1 - sini * cosphi) / (1 + sini * cosphi)
    )

    # Eracleous et al, eq 16
    gamma = (
        1
        - (e**2 * sinphiphinot**2 + scale * one_minus_e_cosphiphinot**2)
        / (xi * scale**2 * one_minus_e_cosphiphinot)
    ) ** -0.5

    term_binner = 1 - b_div_r**2 * scale
    term_binner = jnp.where(term_binner < 0, 0, term_binner)

    # Eracleous et al, eq 15
    da = scale**-0.5
    db = term_binner**0.5 * e * sinphiphinot
    dc = xi**0.5 * scale ** (3 / 2) * one_minus_e_cosphiphinot**0.5
    dd = b_div_r * one_minus_e_cosphiphinot**0.5 * sini * sinphi
    de = xi**0.5 * scale**0.5 * one_minus_sinisq_cosphisq**0.5

    # dc = jnp.where(dc < FLOAT_EPSILON, FLOAT_EPSILON, dc)
    # de = jnp.where(de < FLOAT_EPSILON, FLOAT_EPSILON, de)

    dbc = db / dc
    dde = dd / de

    # dbc = jnp.where(jnp.isnan(dbc), 0.0, dbc)
    # dde = jnp.where(jnp.isnan(dde), 0.0, dde)

    inv_dop = gamma * (da - dbc + dde)
    dop = inv_dop**-1

    return dop


def intensity(
    xi: float, X: ArrayLike, D: float, sigma: float, q: float, nu0: float
) -> ArrayLike:
    """
    `I_nu` function for the disk model, as defined in Eracleous et al. (1995).
    """
    # Eracleous et al, eq 18; returned units are erg / cm^2
    # exponent = -((1 + X - D) ** 2) / (2 * D**2) * (c_cgs / sigma) ** 2
    exponent = -((1 + X - D) ** 2 * nu0**2) / (2 * D**2 * sigma**2)
    exponent = jnp.maximum(exponent, -37.0)

    # res = (xi**-q * c_cgs) / (jnp.sqrt(2 * jnp.pi) * sigma) * jnp.exp(exponent)
    res = (xi**-q) / (jnp.sqrt(2 * jnp.pi) * sigma) * jnp.exp(exponent)

    return res


def Psi(xi: float, phi: float, inc: float) -> ArrayLike:
    """
    `Psi` function for the disk model, as defined in Eracleous et al. (1995).
    """
    # Eracleous et al, eq 8
    numerator = 1 - jnp.sin(inc) * jnp.cos(phi)
    denominator = 1 + jnp.sin(inc) * jnp.cos(phi)
    denominator = jnp.where(denominator < FLOAT_EPSILON, FLOAT_EPSILON, denominator)

    return 1 + (1 / xi) * (numerator / denominator)


def integrand(
    phi: float,
    xi_tilde: float,
    X: ArrayLike,
    inc: float,
    sigma: float,
    q: float,
    e: float,
    phi0: float,
    nu0: float,
) -> ArrayLike:
    """
    Integrand for the double integral over `xi` and `phi`.
    """
    # Eracleous et al, eq 10
    trans_fac_denominator = 1 - e * jnp.cos(phi - phi0)
    trans_fac_denominator = jnp.where(
        trans_fac_denominator < FLOAT_EPSILON, FLOAT_EPSILON, trans_fac_denominator
    )
    trans_fac = (1 + e) / trans_fac_denominator
    xi = xi_tilde * trans_fac

    D = doppler_factor(xi, phi, inc, e, phi0)
    I_nu = intensity(xi, X, D, sigma, q, nu0)
    Psi_ = Psi(xi, phi, inc)

    # jax.debug.print("{} {} {}", jnp.mean(D), jnp.mean(I_nu), jnp.mean(Psi_))

    # Eracleous et al, eq 7
    res = xi * I_nu * D**3 * Psi_ * trans_fac

    # res = jnp.where(jnp.isnan(res) | jnp.isinf(res), 0.0, res)

    return res
