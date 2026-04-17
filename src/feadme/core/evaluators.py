from functools import partial
from typing import Callable, Dict, Tuple

import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike

from .integrators import quad_jax_integrate
from .parser import Template, Shape

ERR = float(np.finfo(np.float32).tiny)
EPS = 1e-6
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value


def compose_param_arrays(
    template: Template, param_mods: Dict[str, float], redshift: float
) -> Tuple[Dict[str, ArrayLike], Dict[str, ArrayLike]]:
    disk_arrays = {}
    line_arrays = {}

    if template.disk_profiles:
        for param_name in [
            "inner_radius",
            "outer_radius",
            "sigma",
            "inclination",
            "q",
            "eccentricity",
            "apocenter",
            "area",
            "offset",
        ]:
            disk_arrays[param_name] = jnp.stack(
                [
                    param_mods[f"{prof.name}_{param_name}"]
                    for prof in template.disk_profiles
                ]
            )

        disk_arrays["center"] = jnp.stack(
            [
                param_mods[f"{prof.name}_center"] * (1 + redshift)
                for prof in template.disk_profiles
            ]
        )

        # Precomputed per-disk X-grids: shape (n_disks, n_disk_samples)
        disk_arrays["x_grids"] = jnp.asarray(template.x_grids)

    if template.line_profiles:
        for param_name in ["offset", "vel_width", "area"]:
            line_arrays[param_name] = jnp.stack(
                [
                    param_mods[f"{prof.name}_{param_name}"]
                    for prof in template.line_profiles
                ]
            )

        line_arrays["center"] = jnp.stack(
            [prof.center * (1 + redshift) for prof in template.line_profiles]
        )

        line_arrays["shape"] = jnp.array(
            [
                (
                    param_mods[f"{prof.name}_shape"]
                    if f"{prof.name}_shape" in param_mods
                    else prof.shape == Shape.GAUSSIAN
                )
                for prof in template.line_profiles
            ]
        )

    return disk_arrays, line_arrays


def evaluate_model(
    template: Template,
    wave: ArrayLike,
    param_mods: Dict[str, float],
    redshift: float,
    integrator: Callable = quad_jax_integrate,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    disk_arrays, line_arrays = compose_param_arrays(template, param_mods, redshift)

    # Build arrays for ALL disk profiles at once (not in a loop)
    if disk_arrays:
        total_disk_flux = _compute_disk_flux_vectorized(
            wave,
            **disk_arrays,
            integrator=integrator,
        )
    else:
        total_disk_flux = jnp.zeros_like(wave)

    if line_arrays:
        # Call vectorized function once with all line profiles
        total_line_flux = _compute_line_flux_vectorized(wave, **line_arrays)
    else:
        total_line_flux = jnp.zeros_like(wave)

    # Combine fluxes
    total_flux = total_disk_flux + total_line_flux

    return total_flux, total_disk_flux, total_line_flux


@jax.jit
def _compute_line_flux_vectorized(
    wave,
    center,
    offset,
    vel_width,
    area,
    shape,
):
    """
    Vectorized line profiles (Gaussian or Lorentzian) with a *flux/area*
    parameterization to reduce amplitude-width degeneracy.

    Parameters
    ----------
    wave : ArrayLike
        Wavelength array (n_wave,)
    center : ArrayLike
        Rest-frame line centers in wavelength units (n_lines,)
    offset : ArrayLike
        Velocity offsets in km/s (n_lines,)
    vel_width : ArrayLike
        Velocity dispersion (sigma_v) in km/s (n_lines,)
    area : ArrayLike
        Integrated line flux (area under profile in wavelength units) (n_lines,)
        - Gaussian: integral over lambda equals area
        - Lorentzian: integral over lambda equals area
    shape : ArrayLike
        Boolean array: True for Gaussian, False for Lorentzian (n_lines,)

    Returns
    -------
    ArrayLike
        Total line flux density evaluated on wave grid (n_wave,)
    """
    if len(center) == 0:
        return jnp.zeros_like(wave)

    wave_bc = wave[:, None]  # (n_wave, 1)
    centers_bc = center[None, :]  # (1, n_lines)
    offsets_bc = offset[None, :]  # (1, n_lines)
    vel_widths_bc = vel_width[None, :]  # (1, n_lines)
    areas_bc = area[None, :]  # (1, n_lines)
    shapes_bc = shape[None, :]  # (1, n_lines)

    # apply velocity offset to center (non-relativistic Doppler)
    centers_bc = centers_bc * (1.0 + offsets_bc / c_kms)

    delta_lamb = wave_bc - centers_bc

    sigma_lambda = vel_widths_bc / c_kms * centers_bc

    # Gaussian: area = amp * sqrt(2pi) * sigma_lambda  => amp = area / (sqrt(2pi)*sigma_lambda)
    amp_g = areas_bc / (jnp.sqrt(2.0 * jnp.pi) * sigma_lambda)
    gau = amp_g * jnp.exp(-0.5 * (delta_lamb / sigma_lambda) ** 2)

    # Lorentzian: L = amp * gamma / (x^2 + gamma^2), integral = amp * pi  => amp = area / pi
    fwhm_lambda = 2.35482 * sigma_lambda
    gamma = 0.5 * fwhm_lambda  # HWHM
    amp_l = areas_bc / jnp.pi
    lor = amp_l * gamma / (delta_lamb**2 + gamma**2)

    line_fluxes = jnp.where(shapes_bc, gau, lor)
    return jnp.sum(line_fluxes, axis=1)


@partial(jax.jit, static_argnames=["integrator"])
def _compute_disk_flux_vectorized(
    wave,
    center,
    inner_radius,
    outer_radius,
    sigma,
    inclination,
    q,
    eccentricity,
    apocenter,
    area,
    offset,
    x_grids,
    integrator: Callable,
):
    """
    Compute disk flux for multiple disk profiles using a precomputed per-disk
    X-grid for integration and normalization, then interpolate onto the
    observed wavelength grid.

    Parameters
    ----------
    wave : ArrayLike
        Observed wavelength array (n_wave,)
    center : ArrayLike
        Observed-frame line center wavelengths (n_disks,)
    inner_radius, outer_radius, sigma, inclination, q, eccentricity, apocenter : ArrayLike
        Disk parameters (n_disks,)
    area : ArrayLike
        Integrated disk flux (n_disks,)
    offset : ArrayLike
        Additive continuum offset (n_disks,)
    x_grids : ArrayLike
        Precomputed X-grids derived from mask bounds, shape (n_disks, n_samples).
        Each row is an ascending linspace in X = (center - lambda) / center.
    integrator : Callable
        Signature: (rin, rout, phi_min, phi_max, X, inc, sigma, q, ecc, apo) -> res_X

    Returns
    -------
    ArrayLike
        Summed disk flux density evaluated on the observed wavelength grid (n_wave,)
    """

    def _compute_single(
        center_i, inner_i, outer_i, sigma_i, inc_i, q_i, ecc_i, apo_i, area_i, offset_i, x_grid_i
    ):
        # Integrate on the fixed, evenly-spaced X-grid (ascending)
        res_X = integrator(
            inner_i,
            outer_i,
            0.0,
            2.0 * jnp.pi,
            x_grid_i,
            inc_i,
            sigma_i,
            q_i,
            ecc_i,
            apo_i,
        )

        # Normalize on the fixed grid: stable, data-independent, no gap issues.
        # x_grid_i is ascending so trapezoid sign is correct.
        norm = center_i * jnp.trapezoid(res_X, x_grid_i)
        norm = jnp.maximum(norm, ERR)
        profile_X = res_X / norm

        # Interpolate onto observed wavelengths. x_grid_i is a uniform linspace
        # so we can compute the fractional index directly — no searchsorted needed.
        # This keeps the XLA graph simple and gives correct gradients through x_wave.
        x_wave = (center_i - wave) / center_i
        n = profile_X.shape[0]
        x_lo = x_grid_i[0]
        x_hi = x_grid_i[-1]
        idx_f = jnp.clip((x_wave - x_lo) / (x_hi - x_lo) * (n - 1), 0.0, n - 1.0)
        idx_lo = jnp.floor(idx_f).astype(jnp.int32)
        idx_hi = jnp.minimum(idx_lo + 1, n - 1)
        frac = idx_f - idx_lo
        profile_wave = profile_X[idx_lo] * (1.0 - frac) + profile_X[idx_hi] * frac

        return profile_wave * area_i + offset_i

    prof_disk_flux = jax.vmap(
        _compute_single,
        in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )(
        center,
        inner_radius,
        outer_radius,
        sigma,
        inclination,
        q,
        eccentricity,
        apocenter,
        area,
        offset,
        x_grids,
    )

    return jnp.sum(prof_disk_flux, axis=0)
