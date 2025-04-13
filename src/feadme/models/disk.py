from functools import partial

import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from quadax import quadgk, GaussKronrodRule, trapezoid, quadcc

from jax import Array
from jax.typing import ArrayLike

FLOAT_EPSILON = float(np.finfo(np.float32).tiny)
ERR = 1e-5
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value

fixed_quadgk = GaussKronrodRule(order=41).integrate


@jax.jit
def doppler_factor(
    xi: ArrayLike, phi: ArrayLike, inc: float, e: float, phi0: float
) -> Array:
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
    inv_dop = gamma * (
        scale**-0.5
        # - (1 - b_div_r**2 * scale) ** 0.5
        - term_binner**0.5
        * e
        * sinphiphinot
        / (xi**0.5 * scale ** (3 / 2) * (1 - e * cosphiphinot) ** 0.5)
        + b_div_r
        * (1 - e * cosphiphinot) ** 0.5
        * sini
        * sinphi
        / (xi**0.5 * scale**0.5 * (1 - sini**2 * cosphi**2) ** 0.5)
    )

    return inv_dop**-1


@jax.jit
def intensity(xi: ArrayLike, X: ArrayLike, D: float, sigma: float, q: float) -> Array:
    # Eracleous et al, eq 18; returned units are erg / cm^2
    exponent = -((1 + X - D) ** 2) / (2 * D**2) * (c_cgs / sigma) ** 2

    res = (xi**-q * c_cgs) / (jnp.sqrt(2 * jnp.pi) * sigma) * jnp.exp(exponent)
    res = jnp.where(exponent < -37, 0.0, res)
    # res[exponent < -37] = 0.0
    # if exponent <= -37:
    #     return 0.0

    return res


@jax.jit
def Psi(xi: ArrayLike, phi: ArrayLike, inc: float) -> Array:
    # Eracleous et al, eq 8
    return 1 + (1 / xi) * (
        (1 - jnp.sin(inc) * jnp.cos(phi)) / (1 + jnp.sin(inc) * jnp.cos(phi))
    )


@jax.jit
def integrand(
    phi: ArrayLike,
    xi_tilde: ArrayLike,
    X: ArrayLike,
    inc: float,
    sigma: float,
    q: float,
    e: float,
    phi0: float,
) -> Array:
    # Eracleous et al, eq 10
    xi = xi_tilde * (1 + e) / (1 - e * jnp.cos(phi - phi0))

    D = doppler_factor(xi, phi, inc, e, phi0)
    I_nu = intensity(xi, X, D, sigma, q)

    # Eracleous et al, eq 7
    res = xi * I_nu * D**3 * Psi(xi, phi, inc)

    return res


@jax.jit
def _inner_quad(
    xi: ArrayLike,
    phi1: float,
    phi2: float,
    X: ArrayLike,
    inc: float,
    sigma: float,
    q: float,
    e: float,
    phi0: float,
) -> Array:
    return fixed_quadgk(integrand, phi1, phi2, args=(xi, X, inc, sigma, q, e, phi0))[0]
    # return quadgk(
    #     integrand,
    #     [phi1, phi2],
    #     args=(xi, X, inc, sigma, q, e, phi0),
    #     order=15,
    # )[0]
    # return quadcc(
    #     integrand,
    #     [phi1, phi2],
    #     args=(xi, X, inc, sigma, q, e, phi0),
    #     order=4,
    # )[0]


@jax.jit
def jax_integrate(
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
) -> Array:
    return fixed_quadgk(
        _inner_quad, xi1, xi2, args=(phi1, phi2, X, inc, sigma, q, e, phi0)
    )[0]
    # return quadgk(
    #     _inner_quad,
    #     [xi1, xi2],
    #     args=(phi1, phi2, X, inc, sigma, q, e, phi0),
    #     order=15,
    # )[0]
    # return quadcc(
    #     _inner_quad,
    #     [xi1, xi2],
    #     args=(phi1, phi2, X, inc, sigma, q, e, phi0),
    #     order=4,
    # )[0]


@jax.jit
def _jax_integrate(
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
) -> Array:
    N_xi, N_phi = 50, 50

    # xi = jax.lax.cond(
    #     xi2 / xi1 > 10,
    #     lambda _: jnp.logspace(jnp.log10(xi1), jnp.log10(xi2), N_xi),
    #     lambda _: jnp.linspace(xi1, xi2, N_xi),
    #     operand=None,
    # )

    xi = jnp.logspace(jnp.log10(xi1), jnp.log10(xi2), N_xi)
    phi = jnp.linspace(phi1, phi2, N_phi)

    # DISP, XI, PHI = jnp.meshgrid(
    #     X,
    #     xi,
    #     phi,
    #     indexing="ij",
    # )
    #
    # res = jax.vmap(integrand, in_axes=(0, 0, 0, None, None, None, None, None))(
    #     PHI, XI, DISP, inc, sigma, q, e, phi0
    # )
    #
    # inner_integral = jnp.trapezoid(res, x=phi, axis=2)
    # outer_integral = jnp.trapezoid(inner_integral, x=xi, axis=1)

    # jax.debug.print("{}", res.shape)

    XI, PHI = jnp.meshgrid(
        xi,
        phi,
        indexing="ij",
    )

    res = jax.vmap(integrand, in_axes=(0, 0, None, None, None, None, None, None))(
        PHI, XI, X[:, None], inc, sigma, q, e, phi0
    )

    inner_integral = jnp.trapezoid(res, x=phi, axis=2)
    outer_integral = jnp.trapezoid(inner_integral, x=xi, axis=0)

    # jax.debug.print("{}", res.shape)

    # res = jax.vmap(integrand, in_axes=(None, 0, None, None, None, None, None, None))(
    #     phi[:, None], xi, X[None, :], inc, sigma, q, e, phi0
    # )
    #
    # inner_integral = jnp.trapezoid(res, x=phi, axis=1)
    # outer_integral = jnp.trapezoid(inner_integral, x=xi, axis=0)

    # jax.debug.print("{}", res.shape)

    # res = integrand(
    #     phi[:, None, None], xi[None, :, None], X[None, None, :], inc, sigma, q, e, phi0
    # )
    #
    # inner_integral = jnp.trapezoid(res, x=phi, axis=0)
    # outer_integral = jnp.trapezoid(inner_integral, x=xi, axis=0)

    return outer_integral
