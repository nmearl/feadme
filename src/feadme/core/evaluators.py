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
    template: Template, param_mods: Dict[str, float]
) -> Tuple[Dict[str, ArrayLike], Dict[str, ArrayLike]]:
    disk_arrays = {}
    line_arrays = {}

    if template.disk_profiles:
        for param_type in [
            "center",
            "inner_radius",
            "outer_radius",
            "sigma",
            "inclination",
            "q",
            "eccentricity",
            "apocenter",
            "scale",
            "offset",
        ]:
            disk_arrays[param_type] = jnp.array(
                [
                    param_mods.get(f"{prof.name}_{param_type}", jnp.nan)
                    for prof in template.disk_profiles
                ]
            )

    if template.line_profiles:
        line_arrays = {
            "center": jnp.array([prof.center for prof in template.line_profiles]),
            "offset": jnp.array(
                [param_mods[f"{prof.name}_offset"] for prof in template.line_profiles]
            ),
            "vel_width": jnp.array(
                [
                    param_mods[f"{prof.name}_vel_width"]
                    for prof in template.line_profiles
                ]
            ),
            "amplitude": jnp.array(
                [
                    param_mods[f"{prof.name}_amplitude"]
                    for prof in template.line_profiles
                ]
            ),
            "shape": jnp.array(
                [
                    (
                        param_mods[f"{prof.name}_shape"]
                        if f"{prof.name}_shape" in param_mods
                        else prof.shape == Shape.GAUSSIAN
                    )
                    for prof in template.line_profiles
                ]
            ),
        }

    return disk_arrays, line_arrays


def evaluate_model(
    template: Template,
    wave: ArrayLike,
    param_mods: Dict[str, float],
    integrator: Callable = quad_jax_integrate,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    disk_arrays, line_arrays = compose_param_arrays(template, param_mods)

    # Build arrays for ALL disk profiles at once (not in a loop)
    if disk_arrays:
        # Call vectorized function once with all disk profiles
        total_disk_flux = _compute_disk_flux_vectorized(
            wave, **disk_arrays, integrator=integrator
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
    wave: ArrayLike,
    center: ArrayLike,
    offset: ArrayLike,
    vel_width: ArrayLike,
    amplitude: ArrayLike,
    shape: ArrayLike,
) -> ArrayLike:
    """
    Compute the line flux for multiple spectral lines in a vectorized manner.

    Parameters
    ----------
    wave : ArrayLike
        Wavelength array
    center : ArrayLike
        Line centers in wavelength units
    offset : ArrayLike
        Velocity offsets in km/s (to be converted to wavelength shift)
    vel_width : ArrayLike
        Velocity dispersion (sigma_v) in km/s
    amplitude : ArrayLike
        Line amplitudes
    shape : ArrayLike
        Boolean array: True for Gaussian, False for Lorentzian
    """
    if len(center) == 0:
        return jnp.zeros_like(wave)

    # Broadcast for vectorized computation: (n_wave, n_lines)
    wave_bc = wave[:, None]
    centers_bc = center[None, :]
    offsets_bc = offset[None, :]
    vel_widths_bc = vel_width[None, :]  # This is sigma_v, not FWHM
    amplitudes_bc = amplitude[None, :]
    shapes_bc = shape[None, :]

    # Adjust centers for velocity offsets
    centers_bc += offsets_bc / c_kms * centers_bc

    # Compute wavelength offset
    delta_lamb = wave_bc - centers_bc

    # Convert velocity dispersion (sigma_v) to wavelength dispersion (sigma_lambda)
    # sigma_lambda = (sigma_v / c) * lambda_0
    sigma_lambda = vel_widths_bc / c_kms * centers_bc

    # Gaussian profile: exp[-(x - x0)^2 / (2 * sigma^2)]
    gau_exp = -0.5 * (delta_lamb / sigma_lambda) ** 2
    gau = amplitudes_bc * jnp.exp(gau_exp)

    # Lorentzian profile
    # For Lorentzian, HWHM is more natural, but we have sigma
    # Convert sigma to FWHM, then to HWHM
    # FWHM_gaussian = 2.35482 * sigma
    # For Lorentzian with same width perception, use same FWHM
    fwhm_lambda = 2.35482 * sigma_lambda
    hwhm_lambda = 0.5 * fwhm_lambda
    lor = amplitudes_bc * hwhm_lambda / (delta_lamb**2 + hwhm_lambda**2)

    # Select based on shape (True = Gaussian, False = Lorentzian)
    line_fluxes = jnp.where(shapes_bc, gau, lor)

    # Sum over all lines
    return jnp.sum(line_fluxes, axis=1)


@partial(jax.jit, static_argnames=["integrator"])
def _compute_disk_flux_vectorized(
    wave: ArrayLike,
    center: ArrayLike,
    inner_radius: ArrayLike,
    outer_radius: ArrayLike,
    sigma: ArrayLike,
    inclination: ArrayLike,
    q: ArrayLike,
    eccentricity: ArrayLike,
    apocenter: ArrayLike,
    scale: ArrayLike,
    offset: ArrayLike,
    integrator: Callable,
) -> ArrayLike:
    """
    Compute the disk flux for multiple disk profiles in a vectorized manner.
    """

    def _compute_single(
        center_i, inner_i, outer_i, sigma_i, inc_i, q_i, ecc_i, apo_i, scale_i, offset_i
    ):
        # nu = c_cgs / (wave * 1e-8)
        # nu0 = c_cgs / (center_i * 1e-8)
        # X = nu / nu0 - 1
        # local_sigma = sigma_i * 1e5 * nu0 / c_cgs

        velocity = (wave - center_i) / center_i * c_kms
        X = -velocity / c_kms

        res_X = integrator(
            inner_i,
            outer_i,
            0.0,
            2 * jnp.pi,
            jnp.asarray(X),
            inc_i,
            sigma_i,
            q_i,
            ecc_i,
            apo_i,
        )

        # Check for invalid results
        normalized_res = res_X / jnp.sqrt(jnp.mean(res_X**2))
        return normalized_res * scale_i + offset_i

        # norm = jax.lax.stop_gradient(jnp.sqrt(jnp.mean(res_X**2)))
        # return (res_X / norm) * scale_i + offset_i

    prof_disk_flux = jax.vmap(_compute_single, in_axes=(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))(
        center,
        inner_radius,
        outer_radius,
        sigma,
        inclination,
        q,
        eccentricity,
        apocenter,
        scale,
        offset,
    )

    return jnp.sum(prof_disk_flux, axis=0)
