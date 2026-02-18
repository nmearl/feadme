import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
ERR = 1e-5
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value


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
    one_minus_e_cosphiphinot = 1.0 - e * cosphiphinot

    # Transform
    trans_fac = (1.0 + e) / one_minus_e_cosphiphinot
    xi = xi_tilde * trans_fac

    # Powers and reciprocals
    xi_recip = 1.0 / xi
    sqrt_xi = jnp.sqrt(xi)

    scale = 1.0 - 2.0 * xi_recip
    sqrt_scale = jnp.sqrt(scale)

    # Compute these once
    sqrt_one_minus_e_cosphiphinot = jnp.sqrt(one_minus_e_cosphiphinot)
    sqrt_one_minus_sinisq_cosphisq = jnp.sqrt(one_minus_sinisq_cosphisq)

    # b/r
    one_plus_sini_cosphi = 1.0 + sini_cosphi
    # When i → π/2 (edge-on) and φ → π (far side of disk), the denominator
    # 1 + sin(i)cos(φ) → 0, causing Ψ to diverge. Physically, this represents
    # the infinite path length through an infinitesimally thin disk when
    # viewing along the disk plane. Real disks have finite thickness that
    # regulates this singularity. We mitigate this issue by imposing a safeguard.
    one_plus_sini_cosphi = 0.5 * (
        one_plus_sini_cosphi + jnp.sqrt(one_plus_sini_cosphi**2 + 1e-5)
    )
    one_minus_sini_cosphi = 1.0 - sini_cosphi

    b_div_r = sqrt_one_minus_sinisq_cosphisq * (
        1.0 + xi_recip * one_minus_sini_cosphi / one_plus_sini_cosphi
    )

    # Gamma - use reciprocal form
    e_sq_sin_sq = e * e * sinphiphinot * sinphiphinot
    scale_sq = scale * scale
    one_minus_e_cosphiphinot_sq = one_minus_e_cosphiphinot * one_minus_e_cosphiphinot

    gamma_denom = 1.0 - (e_sq_sin_sq + scale * one_minus_e_cosphiphinot_sq) / (
        xi * scale_sq * one_minus_e_cosphiphinot
    )
    gamma_denom = 0.5 * (gamma_denom + jnp.sqrt(gamma_denom**2 + EPS))
    gamma = jnp.sqrt(1.0 / gamma_denom)

    # Doppler components
    b_div_r_sq_scale = b_div_r * b_div_r * scale
    one_minus_b_div_r_sq_scale = 1.0 - b_div_r_sq_scale
    term_binner = 0.5 * (
        one_minus_b_div_r_sq_scale + jnp.sqrt(one_minus_b_div_r_sq_scale**2 + EPS)
    )
    # term_binner = jnp.maximum(1.0 - b_div_r_sq_scale, 0.0)
    # term_binner = one_minus_b_div_r_sq_scale

    # Optimize division chains
    inv_sqrt_scale = 1.0 / sqrt_scale

    da = inv_sqrt_scale

    # Numerator and denominator for db/dc
    db_num = jnp.sqrt(term_binner) * e * sinphiphinot
    dc_val = sqrt_xi * scale * sqrt_scale * sqrt_one_minus_e_cosphiphinot

    # Numerator for dd/de
    dd_num = b_div_r * sqrt_one_minus_e_cosphiphinot * sini * sinphi
    de_val = sqrt_xi * sqrt_scale * sqrt_one_minus_sinisq_cosphisq

    inv_dop = gamma * (da - db_num / dc_val + dd_num / de_val)
    D = 1.0 / inv_dop

    # Intensity - optimize exponent computation
    D_sq = D * D
    one_plus_X_minus_D_sq = (1.0 + X - D) ** 2

    # exponent = -one_plus_X_minus_D_sq * (nu0 * nu0) / (2.0 * D_sq * sigma * sigma)
    exponent = -one_plus_X_minus_D_sq / (2 * D**2) * (c_kms / sigma) ** 2
    # exponent = jnp.maximum(exponent, -37.0)

    # Pre-compute constant
    # cc = 1.0 / (jnp.sqrt(2.0 * jnp.pi) * sigma)
    cc = c_kms / (jnp.sqrt(2.0 * jnp.pi) * sigma)
    I_nu = jnp.power(xi, -q) * cc * jnp.exp(exponent)

    # Psi
    Psi_ = 1.0 + xi_recip * one_minus_sini_cosphi / one_plus_sini_cosphi

    # Final - avoid repeated multiplication
    D_cubed = D_sq * D
    res = xi * I_nu * D_cubed * Psi_ * trans_fac

    return res
