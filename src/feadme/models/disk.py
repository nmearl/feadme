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
    # one_minus_sinisq_cosphisq = jnp.maximum(one_minus_sinisq_cosphisq, 1e-15)
    one_minus_e_cosphiphinot = 1 - e * cosphiphinot
    # one_minus_e_cosphiphinot = jnp.maximum(one_minus_e_cosphiphinot, 1e-15)
    one_plus_sini_cosphi = 1 + sini * cosphi
    # one_plus_sini_cosphi = jnp.maximum(one_plus_sini_cosphi, 1e-15)

    # Eracleous et al, eq 6
    b_div_r = jnp.sqrt(one_minus_sinisq_cosphisq) * (
        1 + (1 / xi) * (1 - sini * cosphi) / one_plus_sini_cosphi
    )

    # Eracleous et al, eq 16
    gamma = (
        1
        - (e**2 * sinphiphinot**2 + scale * one_minus_e_cosphiphinot**2)
        / (xi * scale**2 * one_minus_e_cosphiphinot)
    ) ** -0.5

    term_binner = 1 - b_div_r**2 * scale
    term_binner = jnp.maximum(term_binner, 0.0)

    # Eracleous et al, eq 15
    da = scale**-0.5
    db = term_binner**0.5 * e * sinphiphinot
    dc = xi**0.5 * scale ** (3 / 2) * one_minus_e_cosphiphinot**0.5
    dd = b_div_r * one_minus_e_cosphiphinot**0.5 * sini * sinphi
    de = xi**0.5 * scale**0.5 * one_minus_sinisq_cosphisq**0.5

    # dc = jnp.sign(dc) * jnp.maximum(dc, 1e-15)
    # de = jnp.sign(de) * jnp.maximum(de, 1e-15)

    inv_dop = gamma * (da - db / dc + dd / de)
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
    # denominator = jnp.sign(denominator) * jnp.maximum(denominator, 1e-15)

    return 1 + (1 / xi) * (numerator / denominator)


def _integrand(
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
    # trans_fac_denominator = jnp.sign(trans_fac_denominator) * jnp.maximum(
    #     trans_fac_denominator, 1e-15
    # )
    trans_fac = (1 + e) / trans_fac_denominator
    xi = xi_tilde * trans_fac

    D = doppler_factor(xi, phi, inc, e, phi0)
    I_nu = intensity(xi, X, D, sigma, q, nu0)
    Psi_ = Psi(xi, phi, inc)

    # Eracleous et al, eq 7
    res = xi * I_nu * D**3 * Psi_ * trans_fac

    return res


@jax.jit
def integrand(
    phi: ArrayLike | float,
    xi_tilde: ArrayLike | float,
    X: ArrayLike,
    inc: float,
    sigma: float,
    q: float,
    e: float,
    phi0: float,
    nu0: float,
) -> ArrayLike:
    """
    Ultra-optimized version with algebraic simplifications and numerical safeguards.
    """
    EPS = 1e-15  # Small epsilon for numerical stability

    # Trigonometric pre-computation
    sini = jnp.sin(inc)
    sinphi = jnp.sin(phi)
    cosphi = jnp.cos(phi)

    phi_diff = phi - phi0
    sinphiphinot = jnp.sin(phi_diff)
    cosphiphinot = jnp.cos(phi_diff)

    # Common terms
    sini_cosphi = sini * cosphi
    sini_cosphi_sq = sini_cosphi * sini_cosphi

    one_minus_sinisq_cosphisq = 1.0 - sini_cosphi_sq
    one_minus_e_cosphiphinot = jnp.maximum(1.0 - e * cosphiphinot, EPS)

    # Transform
    trans_fac = (1.0 + e) / one_minus_e_cosphiphinot
    xi = xi_tilde * trans_fac

    # Powers and reciprocals
    xi_recip = 1.0 / jnp.maximum(xi, EPS)
    sqrt_xi = jnp.sqrt(xi)

    scale = 1.0 - 2.0 * xi_recip
    sqrt_scale = jnp.sqrt(jnp.maximum(scale, EPS))

    # Compute these once
    sqrt_one_minus_e_cosphiphinot = jnp.sqrt(one_minus_e_cosphiphinot)
    sqrt_one_minus_sinisq_cosphisq = jnp.sqrt(one_minus_sinisq_cosphisq)

    # b/r
    one_plus_sini_cosphi = jnp.maximum(1.0 + sini_cosphi, EPS)
    one_minus_sini_cosphi = 1.0 - sini_cosphi

    b_div_r = sqrt_one_minus_sinisq_cosphisq * (
        1.0 + xi_recip * one_minus_sini_cosphi / one_plus_sini_cosphi
    )

    # Gamma - safeguard against negative denominator
    e_sq_sin_sq = e * e * sinphiphinot * sinphiphinot
    scale_sq = scale * scale
    one_minus_e_cosphiphinot_sq = one_minus_e_cosphiphinot * one_minus_e_cosphiphinot

    gamma_denom = 1.0 - (e_sq_sin_sq + scale * one_minus_e_cosphiphinot_sq) / (
        xi * scale_sq * one_minus_e_cosphiphinot + EPS
    )

    # If gamma_denom <= 0, the geometry is unphysical - return 0
    gamma = jnp.where(gamma_denom > EPS, jnp.sqrt(1.0 / gamma_denom), 0.0)

    # Doppler components
    b_div_r_sq_scale = b_div_r * b_div_r * scale
    term_binner = jnp.maximum(1.0 - b_div_r_sq_scale, 0.0)

    # Optimize division chains
    inv_sqrt_scale = 1.0 / sqrt_scale

    da = inv_sqrt_scale

    # Numerator and denominator for db/dc
    db_num = jnp.sqrt(term_binner) * e * sinphiphinot
    dc_val = jnp.maximum(
        sqrt_xi * scale * sqrt_scale * sqrt_one_minus_e_cosphiphinot, EPS
    )

    # Numerator for dd/de
    dd_num = b_div_r * sqrt_one_minus_e_cosphiphinot * sini * sinphi
    de_val = jnp.maximum(sqrt_xi * sqrt_scale * sqrt_one_minus_sinisq_cosphisq, EPS)

    inv_dop = gamma * (da - db_num / dc_val + dd_num / de_val)
    D = jnp.where(jnp.abs(inv_dop) > EPS, 1.0 / inv_dop, 0.0)

    # Intensity - optimize exponent computation
    D_sq = D * D
    one_plus_X_minus_D_sq = (1.0 + X - D) ** 2

    exponent = (
        -one_plus_X_minus_D_sq
        * (nu0 * nu0)
        / (2.0 * jnp.maximum(D_sq * sigma * sigma, EPS))
    )
    exponent = jnp.maximum(exponent, -37.0)

    # Pre-compute constant
    const = 1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma)
    I_nu = jnp.power(xi, -q) * const * jnp.exp(exponent)

    # Psi
    Psi_ = 1.0 + xi_recip * one_minus_sini_cosphi / one_plus_sini_cosphi

    # Final - avoid repeated multiplication
    D_cubed = D_sq * D
    res = xi * I_nu * D_cubed * Psi_ * trans_fac

    # Zero out result if gamma was invalid
    res = jnp.where(gamma_denom > EPS, res, 0.0)

    return res
