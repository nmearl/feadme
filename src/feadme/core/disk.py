import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

c_kms = const.c.to(u.km / u.s).value

# Small numerical floor for the explicitly guarded singular denominator.
_EPS_POS: float = 1e-12

# Minimal retained guard: keep the Doppler radicand slightly positive so
# sqrt(term_raw) remains defined in the small pathological region where
# the raw Eracleous/Hung algebra crosses zero.
_TERM_FLOOR: float = 1e-8
_TERM_WIDTH: float = 1e-6


def softplus_floor(x: ArrayLike, floor: float, width: float) -> ArrayLike:
    """
    Smooth lower bound using softplus.

    Returns approximately max(x, floor) with a transition scale controlled
    by ``width``.

    Parameters
    ----------
    x:
        Input array.
    floor:
        The asymptotic lower bound.
    width:
        Transition half-width.  Larger values smooth gradients more
        aggressively; smaller values approach a hard clamp.
    """
    z = (x - floor) / width
    return floor + width * jax.nn.softplus(z)


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
    Near-literal Eracleous/Hung disk integrand with minimal numerical guards.

    Notes
    -----
    Regularization strategy
    ~~~~~~~~~~~~~~~~~~~~~~~
    * Most of the algebra is left unguarded so failure surfaces in the raw
      Eracleous/Hung kernel are preserved.
    * ``1 + sin(i) cos(phi)`` is clamped only to avoid the exact razor-thin
      denominator singularity at the edge-on limit.
    * ``term_raw`` retains a soft floor so ``sqrt(term_raw)`` stays defined;
      this was the dominant practical initialization failure mode in ad hoc
      scans of representative disk regimes.
    * No guards are applied to ``gamma_raw``, ``inv_dop_raw``, ``scale``, or
      the Gaussian exponent in this minimal version.
    """
    # ------------------------------------------------------------------
    # Trig
    # ------------------------------------------------------------------
    sini = jnp.sin(inc)
    sinphi = jnp.sin(phi)
    cosphi = jnp.cos(phi)

    phi_diff = phi - phi0
    sinphiphinot = jnp.sin(phi_diff)
    cosphiphinot = jnp.cos(phi_diff)

    # ------------------------------------------------------------------
    # Common factors
    # ------------------------------------------------------------------
    sini_cosphi = sini * cosphi
    sini_cosphi_sq = sini_cosphi * sini_cosphi

    one_minus_sinisq_cosphisq = 1.0 - sini_cosphi_sq
    one_minus_e_cosphiphinot = 1.0 - e * cosphiphinot

    # ------------------------------------------------------------------
    # Transform xi_tilde -> xi
    # ------------------------------------------------------------------
    trans_fac = (1.0 + e) / one_minus_e_cosphiphinot
    xi = xi_tilde * trans_fac

    xi_recip = 1.0 / xi
    sqrt_xi = jnp.sqrt(xi)

    # ------------------------------------------------------------------
    # Schwarzschild-like factor
    # ------------------------------------------------------------------
    scale = 1.0 - 2.0 * xi_recip
    sqrt_scale = jnp.sqrt(scale)

    sqrt_one_minus_e_cosphiphinot = jnp.sqrt(one_minus_e_cosphiphinot)
    sqrt_one_minus_sinisq_cosphisq = jnp.sqrt(one_minus_sinisq_cosphisq)

    # ------------------------------------------------------------------
    # Psi geometry denominator.
    # Using the half-angle form makes the edge-on singular surface explicit;
    # the hard floor only prevents exact division by zero.
    # ------------------------------------------------------------------
    one_plus_sini_cosphi_raw = (1.0 - sini) + 2.0 * sini * jnp.cos(phi / 2.0) ** 2
    one_plus_sini_cosphi = jnp.maximum(one_plus_sini_cosphi_raw, _EPS_POS)
    one_minus_sini_cosphi = 1.0 - sini_cosphi

    # Factor shared by both b/r and Psi so both use the same (regularized)
    # denominator — avoids the silent inconsistency in the previous version.
    psi_geom = one_minus_sini_cosphi / one_plus_sini_cosphi

    b_div_r = sqrt_one_minus_sinisq_cosphisq * (1.0 + xi_recip * psi_geom)

    # ------------------------------------------------------------------
    # Gamma factor (unguarded in the minimal implementation)
    # ------------------------------------------------------------------
    e_sq_sin_sq = e * e * sinphiphinot * sinphiphinot
    scale_sq = scale * scale
    one_minus_e_cosphiphinot_sq = one_minus_e_cosphiphinot * one_minus_e_cosphiphinot

    gamma_raw = 1.0 - (e_sq_sin_sq + scale * one_minus_e_cosphiphinot_sq) / (
        xi * scale_sq * one_minus_e_cosphiphinot
    )
    gamma_denom = gamma_raw
    gamma = jnp.sqrt(1.0 / gamma_denom)

    # ------------------------------------------------------------------
    # Doppler term radicand. This is the only retained smooth guard because
    # term_raw <= 0 was the leading observed raw-algebra failure surface.
    # ------------------------------------------------------------------
    term_raw = 1.0 - b_div_r * b_div_r * scale
    term_binner = softplus_floor(term_raw, _TERM_FLOOR, _TERM_WIDTH)

    inv_sqrt_scale = 1.0 / sqrt_scale

    db_num = jnp.sqrt(term_binner) * e * sinphiphinot
    dc_val = sqrt_xi * scale * sqrt_scale * sqrt_one_minus_e_cosphiphinot

    dd_num = b_div_r * sqrt_one_minus_e_cosphiphinot * sini * sinphi
    de_val = sqrt_xi * sqrt_scale * sqrt_one_minus_sinisq_cosphisq

    inv_dop_raw = gamma * (inv_sqrt_scale - db_num / dc_val + dd_num / de_val)
    inv_dop = inv_dop_raw
    D = 1.0 / inv_dop

    # ------------------------------------------------------------------
    # Intensity
    # ------------------------------------------------------------------
    D_sq = D * D
    exponent = -((1.0 + X - D) ** 2) / (2.0 * D_sq) * (c_kms / sigma) ** 2

    cc = c_kms / (jnp.sqrt(2.0 * jnp.pi) * sigma)
    I_nu = jnp.power(xi, -q) * cc * jnp.exp(exponent)

    # ------------------------------------------------------------------
    # Psi  (reuses psi_geom, same regularized denominator as b/r)
    # ------------------------------------------------------------------
    Psi_ = 1.0 + xi_recip * psi_geom

    # ------------------------------------------------------------------
    # Final flux result
    # ------------------------------------------------------------------
    D_cubed = D_sq * D
    return xi * I_nu * D_cubed * Psi_ * trans_fac
