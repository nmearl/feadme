import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

c_kms = const.c.to(u.km / u.s).value

# Floor for quantities that must be strictly positive but have no
# natural physical scale (e.g. intermediate trig terms, xi).
_EPS_POS: float = 1e-12

# Numerical floor for the near-edge-on singularity in Psi.
# The sqrt-norm regularization is symmetric near zero, so _DENOM_FLOOR is
# not a literal disk scale height — it is a numerical hyperparameter with
# a thickness-like physical interpretation.  It primarily affects the tiny
# pathological corner near (i -> pi/2, phi -> pi).
_DENOM_FLOOR: float = 3e-4

# Softplus parameters for the gamma denominator.  The floor prevents
# gamma from blowing up if the denominator crosses zero; the width sets
# the gradient-smoothing scale for NUTS.
_GAMMA_FLOOR: float = 1e-8
_GAMMA_WIDTH: float = 1e-6

# Radicand floor for the Doppler term sqrt(1 - (b/r)^2 * scale).
# A strictly positive floor ensures the sqrt stays away from zero near
# the photon capture radius, preventing a downstream 1/inv_dop divergence.
# Using softplus with this floor (rather than 0.0) gives a genuine lower
# bound on term_binner.
_TERM_FLOOR: float = 1e-8
_TERM_WIDTH: float = 1e-6

# Lower-only exponent clamp for the Gaussian intensity term.
# The exponent is -(...)^2 / (2D^2) * (c/sigma)^2, which is strictly
# non-positive by construction.  A large positive value therefore signals
# upstream numerical breakage rather than a physical state, so only the
# lower bound is enforced here.  We can add an upper clip too if we observe
# genuine positive excursions (e.g. from sigma -> 0 or D -> 0 edge cases).
_EXPONENT_MIN: float = -37.0


def smooth_floor(x: ArrayLike, floor: float) -> ArrayLike:
    """
    Smooth approximation to max(x, floor) via a sqrt-based softplus.

    For x >> floor returns ~x; for x << floor returns ~floor.
    The transition half-width scales with ``floor``, so very small floors
    produce correspondingly sharp transitions, nearly identical to a hard
    clamp.  Choose ``floor`` with the gradient-smoothness requirement in
    mind, not just numerical stability.
    """
    y = x - floor
    return floor + 0.5 * (y + jnp.sqrt(y * y + floor * floor))


def softplus_floor(x: ArrayLike, floor: float, width: float) -> ArrayLike:
    """
    Smooth lower bound using softplus.

    Returns approximately max(x, floor) with a transition scale controlled
    by ``width``.  Produces a broader, gentler gradient than the sqrt-based
    ``smooth_floor``, which is generally preferable for NUTS samplers.

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
    Eracleous disk integrand with smooth, physically motivated regularization.

    Notes
    -----
    Regularization strategy
    ~~~~~~~~~~~~~~~~~~~~~~~
    * Near-zero trig/algebraic quantities use ``smooth_floor`` with
      ``_EPS_POS`` as a purely numerical guard.
    * The Psi denominator (1 + sin i cos phi) is regularized with a
      finite-thickness floor ``_DENOM_FLOOR`` via the manifestly non-negative
      half-angle form, which avoids the thin-disk singularity in a physically
      interpretable way.
    * The gamma denominator and Doppler radicand use ``softplus_floor`` for
      broader, NUTS-friendly gradient transitions.
    * The Gaussian exponent has a lower clamp at ``_EXPONENT_MIN`` to prevent
      silent underflow/NaN propagation in gradient chains.
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

    one_minus_sinisq_cosphisq = smooth_floor(1.0 - sini_cosphi_sq, _EPS_POS)
    one_minus_e_cosphiphinot = smooth_floor(1.0 - e * cosphiphinot, _EPS_POS)

    # ------------------------------------------------------------------
    # Transform xi_tilde -> xi
    # ------------------------------------------------------------------
    trans_fac = (1.0 + e) / one_minus_e_cosphiphinot
    xi = smooth_floor(xi_tilde * trans_fac, _EPS_POS)

    xi_recip = 1.0 / xi
    sqrt_xi = jnp.sqrt(xi)

    # ------------------------------------------------------------------
    # Schwarzschild-like factor
    # ------------------------------------------------------------------
    scale = smooth_floor(1.0 - 2.0 * xi_recip, _EPS_POS)
    sqrt_scale = jnp.sqrt(scale)

    sqrt_one_minus_e_cosphiphinot = jnp.sqrt(one_minus_e_cosphiphinot)
    sqrt_one_minus_sinisq_cosphisq = jnp.sqrt(one_minus_sinisq_cosphisq)

    # ------------------------------------------------------------------
    # Psi geometry denominator
    #
    # 1 + sin(i)cos(phi) is rewritten via the half-angle identity as
    #   (1 - sin i) + 2 sin i cos^2(phi/2)
    # The sqrt-norm regularization is symmetric near zero, so _DENOM_FLOOR
    # should be read as a numerical hyperparameter with a thickness-like
    # physical interpretation, not a literal geometric scale height.
    # ------------------------------------------------------------------
    one_plus_sini_cosphi_raw = (1.0 - sini) + 2.0 * sini * jnp.cos(phi / 2.0) ** 2
    one_plus_sini_cosphi = jnp.sqrt(one_plus_sini_cosphi_raw**2 + _DENOM_FLOOR**2)
    one_minus_sini_cosphi = 1.0 - sini_cosphi

    # Factor shared by both b/r and Psi so both use the same (regularized)
    # denominator — avoids the silent inconsistency in the previous version.
    psi_geom = one_minus_sini_cosphi / one_plus_sini_cosphi

    b_div_r = sqrt_one_minus_sinisq_cosphisq * (1.0 + xi_recip * psi_geom)

    # ------------------------------------------------------------------
    # Gamma factor  (softplus floor for NUTS-friendly gradients)
    # ------------------------------------------------------------------
    e_sq_sin_sq = e * e * sinphiphinot * sinphiphinot
    scale_sq = scale * scale
    one_minus_e_cosphiphinot_sq = one_minus_e_cosphiphinot * one_minus_e_cosphiphinot

    gamma_raw = 1.0 - (e_sq_sin_sq + scale * one_minus_e_cosphiphinot_sq) / (
        xi * scale_sq * one_minus_e_cosphiphinot
    )
    gamma_denom = softplus_floor(gamma_raw, _GAMMA_FLOOR, _GAMMA_WIDTH)
    gamma = jnp.sqrt(1.0 / gamma_denom)

    # ------------------------------------------------------------------
    # Doppler term radicand
    #
    # A strictly positive _TERM_FLOOR (rather than 0.0) ensures sqrt stays
    # away from zero near the photon capture radius, preventing a downstream
    # 1/inv_dop divergence.
    # ------------------------------------------------------------------
    term_raw = 1.0 - b_div_r * b_div_r * scale
    term_binner = softplus_floor(term_raw, _TERM_FLOOR, _TERM_WIDTH)

    inv_sqrt_scale = 1.0 / sqrt_scale

    db_num = jnp.sqrt(term_binner) * e * sinphiphinot
    dc_val = smooth_floor(
        sqrt_xi * scale * sqrt_scale * sqrt_one_minus_e_cosphiphinot, _EPS_POS
    )

    dd_num = b_div_r * sqrt_one_minus_e_cosphiphinot * sini * sinphi
    de_val = smooth_floor(
        sqrt_xi * sqrt_scale * sqrt_one_minus_sinisq_cosphisq, _EPS_POS
    )

    # NOTE: flooring inv_dop changes the sign structure of the Doppler
    # factor directly.  We are unsure if it is appropriate, since it depends
    # on whether inv_dop crossing zero is a numerical artifact or part of the
    # model's physical support boundary.
    # inv_dop = smooth_floor(
    #     gamma * (inv_sqrt_scale - db_num / dc_val + dd_num / de_val),
    #     _EPS_POS,
    # )
    inv_dop = gamma * (inv_sqrt_scale - db_num / dc_val + dd_num / de_val)
    D = 1.0 / inv_dop

    # ------------------------------------------------------------------
    # Intensity
    #
    # Exponent is clamped to prevent silent NaN propagation through NUTS
    # gradient chains on overflow/underflow.
    # ------------------------------------------------------------------
    D_sq = D * D
    exponent = -((1.0 + X - D) ** 2) / (2.0 * D_sq) * (c_kms / sigma) ** 2
    exponent = jnp.maximum(exponent, _EXPONENT_MIN)

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
