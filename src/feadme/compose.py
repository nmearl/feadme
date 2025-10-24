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
    """
    if len(center) == 0:
        return jnp.zeros_like(wave)

    # Pre-compute constants to avoid repeated calculations
    fwhm_factor = 1.0 / 2.35482  # Gaussian FWHM to sigma conversion
    lorentz_factor = 0.5

    # Broadcast for vectorized computation: (n_wave, n_lines)
    wave_bc = wave[:, None]
    centers_bc = center[None, :]
    vel_widths_bc = vel_width[None, :]
    amplitudes_bc = amplitude[None, :]
    shapes_bc = shape[None, :]

    # Compute delta_lamb and fwhm once
    delta_lamb = wave_bc - centers_bc
    fwhm = vel_widths_bc / c_kms * centers_bc

    # More efficient Gaussian computation
    sigma = fwhm * fwhm_factor
    gau_exp = -0.5 * (delta_lamb / sigma) ** 2
    gau = amplitudes_bc * jnp.exp(gau_exp)

    # More efficient Lorentzian computation
    hwhm = fwhm * lorentz_factor
    lor = amplitudes_bc * hwhm / (delta_lamb**2 + hwhm**2)

    # Select based on shape
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

        res = integrator(
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

        normalized_res = res / jnp.max(res)
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
        k: []
        for k in [
            "center",
            "inner_radius",
            "radius_ratio",
            "outer_radius",
            "sigma",
            "inclination",
            "q",
            "eccentricity",
            "apocenter",
            "scale",
            "offset",
        ]
    }

    line_arrs = {k: [] for k in ["center", "vel_width", "amplitude", "shape"]}

    # Define line profile shapes
    line_arrs["shape"] = [
        prof.shape == Shape.GAUSSIAN for prof in template.line_profiles
    ]

    for i, prof in enumerate(template.disk_profiles):
        # Sample independent parameters
        for param in prof.independent:
            samp_name = param.qualified_name
            param_samp = sample_reparam(samp_name, param)
            param_mods[samp_name] = param_samp
            disk_arrs[param.name].append(param_samp)

        # Add fixed parameters
        for param in prof.fixed:
            samp_name = param.qualified_name
            param_samp = numpyro.deterministic(samp_name, param.value)
            param_mods[samp_name] = param_samp
            disk_arrs[param.name].append(param_samp)

        # Define outer radius from inner radius and ratio
        samp_name = f"{prof.name}_outer_radius"
        param_samp = numpyro.deterministic(
            samp_name,
            param_mods[f"{prof.name}_inner_radius"]
            * param_mods[f"{prof.name}_radius_ratio"],
        )
        param_mods[samp_name] = param_samp
        disk_arrs["outer_radius"].append(param_samp)

    for i, prof in enumerate(template.line_profiles):
        # Sample independent parameters
        for param in prof.independent:
            samp_name = param.qualified_name
            param_samp = sample_reparam(samp_name, param)
            param_mods[samp_name] = param_samp
            line_arrs[param.name].append(param_samp)

        # Add fixed parameters
        for param in prof.fixed:
            samp_name = param.qualified_name
            param_samp = numpyro.deterministic(samp_name, param.value)
            param_mods[samp_name] = param_samp
            line_arrs[param.name].append(param_samp)

    # Add shared parameters
    for i, prof in enumerate(template.disk_profiles):
        for param in prof.shared:
            samp_name = param.qualified_name
            param_samp = numpyro.deterministic(
                samp_name, param_mods[f"{param.shared}_{param.name}"]
            )
            param_mods[samp_name] = param_samp
            disk_arrs[param.name].append(param_samp)

    for i, prof in enumerate(template.line_profiles):
        for param in prof.shared:
            samp_name = param.qualified_name
            param_samp = numpyro.deterministic(
                samp_name, param_mods[f"{param.shared}_{param.name}"]
            )
            param_mods[samp_name] = param_samp
            line_arrs[param.name].append(param_samp)

    # Convert lists to jax arrays
    disk_arrs = {k: jnp.array(v) for k, v in disk_arrs.items()}
    line_arrs = {k: jnp.array(v) for k, v in line_arrs.items()}

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


def construct_model(template: Template, auto_reparam: bool = False):
    if auto_reparam:
        reparam_config = create_reparam_config(template)
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

    return total_flux, total_disk_flux, total_line_flux
