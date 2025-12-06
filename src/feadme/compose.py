from typing import Dict, Optional

import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.infer.reparam import CircularReparam, TransformReparam
from numpyro.handlers import reparam
from jax.typing import ArrayLike

from .integrators import integrator
from .parser import Distribution, Template, Shape, Parameter
from .parameterizers import _sample_manual_reparam as sample_reparam
from .parameterizers import create_reparam_config

ERR = float(np.finfo(np.float32).tiny)
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value


@jax.jit
def _compute_line_flux_vectorized(
    wave: ArrayLike,
    center: ArrayLike,
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
    vel_widths_bc = vel_width[None, :]  # This is sigma_v, not FWHM
    amplitudes_bc = amplitude[None, :]
    shapes_bc = shape[None, :]

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


@jax.jit
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
    delta_radius: ArrayLike = jnp.array([]),
    radius_ratio: ArrayLike = jnp.array([]),
    radius_scale: ArrayLike = jnp.array([]),
) -> ArrayLike:
    """
    Compute the disk flux for multiple disk profiles in a vectorized manner.
    """
    if len(center) == 0:
        return jnp.zeros_like(wave)

    def _compute_single(
        center_i, inner_i, outer_i, sigma_i, inc_i, q_i, ecc_i, apo_i, scale_i, offset_i
    ):
        nu = c_cgs / (wave * 1e-8)
        nu0 = c_cgs / (center_i * 1e-8)
        X = nu / nu0 - 1
        local_sigma = sigma_i * 1e5 * nu0 / c_cgs

        # ecc_i = jnp.minimum(ecc_i, 1.0 - 1e-3)
        # inc_i = jnp.minimum(inc_i, jnp.pi / 2 - 1e-5)

        res_X = integrator(
            inner_i,
            outer_i,
            0.0,
            2 * jnp.pi,
            jnp.asarray(X),
            inc_i,
            local_sigma,
            q_i,
            ecc_i,
            apo_i,
            nu0,
        )

        # Convert from X to wavelength space
        # X = nu / nu0 - 1 = lambda_0 / lambda - 1
        # F_lambda = F_X * |dX/dlambda|
        # dX/dlambda = - lambda_0 / lambda^2
        # F_lambda = F_X * lambda_0 / lambda^2
        # jac = center_i / wave**2
        # res_lambda = res_X * jac

        # Convert from X to frequency space
        # X = nu / nu0 - 1
        # F_nu = F_X * |dX/dnu|
        # dX/dnu = 1 / nu0
        # F_nu = F_X / nu0
        jac = nu0
        res_nu = res_X / jac

        max_res = jnp.max(res_nu)
        normalized_res = res_nu / jnp.maximum(max_res, ERR)

        return normalized_res * scale_i + offset_i

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


def disk_model(
    template: Template,
    wave: ArrayLike,
    flux: Optional[ArrayLike] = None,
    flux_err: Optional[ArrayLike] = None,
):
    """
    Main disk model function that computes the disk and line fluxes.
    """
    # Dictionary to store all sampled parameters
    param_mods = {}

    # Disk model array initialization
    disk_arrs = {
        k: jnp.zeros(len(template.disk_profiles))
        for k in [
            "center",
            "inner_radius",
            "outer_radius",
            "radius_scale",
            "sigma",
            "inclination",
            "q",
            "eccentricity",
            "apocenter",
            "scale",
            "offset",
        ]
    }

    line_arrs = {
        k: jnp.zeros(len(template.line_profiles))
        for k in ["center", "vel_width", "amplitude"]
    }

    # Define line profile shapes
    line_arrs["shape"] = jnp.array(
        [prof.shape == Shape.GAUSSIAN for prof in template.line_profiles]
    )

    for i, prof in enumerate(template.disk_profiles):
        # Sample independent parameters
        for param in prof.independent:
            samp_name = param.qualified_name
            param_samp = sample_reparam(samp_name, param)
            param_mods[samp_name] = param_samp
            disk_arrs[param.name] = disk_arrs[param.name].at[i].set(param_samp)

        # Add fixed parameters
        for param in prof.fixed:
            samp_name = param.qualified_name
            param_samp = numpyro.deterministic(samp_name, param.value)
            param_mods[samp_name] = param_samp
            disk_arrs[param.name] = disk_arrs[param.name].at[i].set(param_samp)

        # Define outer radius from inner radius and ratio
        samp_name = f"{prof.name}_outer_radius"
        inner_radius = disk_arrs["inner_radius"][i]
        radius_scale = disk_arrs["radius_scale"][i]

        param_samp = numpyro.deterministic(
            samp_name,
            10
            ** (
                jnp.log10(inner_radius)
                + (jnp.log10(5e4) - jnp.log10(inner_radius)) * radius_scale
            ),
        )
        param_mods[samp_name] = param_samp
        disk_arrs["outer_radius"] = disk_arrs["outer_radius"].at[i].set(param_samp)

    for i, prof in enumerate(template.line_profiles):
        # Sample independent parameters
        for param in prof.independent:
            samp_name = param.qualified_name
            param_samp = sample_reparam(samp_name, param)
            param_mods[samp_name] = param_samp
            line_arrs[param.name] = line_arrs[param.name].at[i].set(param_samp)

        # Add fixed parameters
        for param in prof.fixed:
            samp_name = param.qualified_name
            param_samp = numpyro.deterministic(samp_name, param.value)
            param_mods[samp_name] = param_samp
            line_arrs[param.name] = line_arrs[param.name].at[i].set(param_samp)

    # Add shared parameters
    for i, prof in enumerate(template.disk_profiles):
        for param in prof.shared:
            samp_name = param.qualified_name
            param_samp = numpyro.deterministic(
                samp_name, param_mods[f"{param.shared}_{param.name}"]
            )
            param_mods[samp_name] = param_samp
            disk_arrs[param.name] = disk_arrs[param.name].at[i].set(param_samp)

    for i, prof in enumerate(template.line_profiles):
        for param in prof.shared:
            samp_name = param.qualified_name
            param_samp = numpyro.deterministic(
                samp_name, param_mods[f"{param.shared}_{param.name}"]
            )
            param_mods[samp_name] = param_samp
            line_arrs[param.name] = line_arrs[param.name].at[i].set(param_samp)

    # Convert lists to jax arrays
    # disk_arrs = {k: jnp.asarray(v) for k, v in disk_arrs.items()}
    # line_arrs = {k: jnp.asarray(v) for k, v in line_arrs.items()}

    # Sample white noise with better bounds
    if template.white_noise.fixed:
        white_noise = numpyro.deterministic("white_noise", template.white_noise.value)
    else:
        white_noise = sample_reparam("white_noise", template.white_noise)

    # Sample redshift
    if template.redshift.fixed:
        redshift = numpyro.deterministic("redshift", template.redshift.value)
    else:
        redshift = sample_reparam("redshift", template.redshift)

    rest_wave = wave / (1 + redshift)

    total_disk_flux = _compute_disk_flux_vectorized(rest_wave, **disk_arrs)
    total_line_flux = _compute_line_flux_vectorized(rest_wave, **line_arrs)
    total_flux = total_disk_flux + total_line_flux

    total_error = jnp.sqrt(flux_err**2 + total_flux**2 * jnp.exp(2 * white_noise))

    with numpyro.plate("data", wave.shape[0]):
        numpyro.deterministic("disk_flux", total_disk_flux)
        numpyro.deterministic("line_flux", total_line_flux)

        numpyro.sample("total_flux", dist.Normal(total_flux, total_error), obs=flux)


def construct_model(
    template: Template, auto_reparam: bool = False, circ_only: bool = False
):
    if auto_reparam:
        reparam_config = create_reparam_config(template, circ_only=circ_only)
        return reparam(disk_model, config=reparam_config)
    else:
        return disk_model


def evaluate_model(template: Template, wave: ArrayLike, param_mods: Dict[str, float]):
    # Build arrays for ALL disk profiles at once (not in a loop)
    if template.disk_profiles:
        disk_arrays = {}

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
                    param_mods[f"{prof.name}_{param_type}"]
                    for prof in template.disk_profiles
                    if f"{prof.name}_{param_type}" in param_mods
                ]
            )

        # Call vectorized function once with all disk profiles
        total_disk_flux = _compute_disk_flux_vectorized(wave, **disk_arrays)
    else:
        total_disk_flux = jnp.zeros_like(wave)

    # Build arrays for ALL line profiles at once
    if template.line_profiles:
        line_arrays = {
            "center": jnp.array(
                [
                    param_mods[f"{prof.name}_center"]
                    for prof in template.line_profiles
                    if f"{prof.name}_center" in param_mods
                ]
            ),
            "vel_width": jnp.array(
                [
                    param_mods[f"{prof.name}_vel_width"]
                    for prof in template.line_profiles
                    if f"{prof.name}_vel_width" in param_mods
                ]
            ),
            "amplitude": jnp.array(
                [
                    param_mods[f"{prof.name}_amplitude"]
                    for prof in template.line_profiles
                    if f"{prof.name}_amplitude" in param_mods
                ]
            ),
            "shape": jnp.array(
                [prof.shape == Shape.GAUSSIAN for prof in template.line_profiles]
            ),
        }

        # Call vectorized function once with all line profiles
        total_line_flux = _compute_line_flux_vectorized(wave, **line_arrays)
    else:
        total_line_flux = jnp.zeros_like(wave)

    # Combine fluxes
    total_flux = total_disk_flux + total_line_flux
    # total_flux = jnp.where(jnp.isfinite(total_flux), total_flux, 0.0)

    return total_flux, total_disk_flux, total_line_flux
