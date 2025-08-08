from typing import Dict, Optional

import astropy.constants as const
import astropy.units as u
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer.reparam import CircularReparam

from .models.disk import quad_jax_integrate
from .parser import Distribution, Template, Shape

ERR = float(np.finfo(np.float32).tiny)
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value


@jax.jit
def _compute_line_flux_vectorized(
    wave: jnp.ndarray,
    center: jnp.ndarray,
    vel_width: jnp.ndarray,
    amplitude: jnp.ndarray,
    shape: jnp.ndarray,
) -> jnp.ndarray:
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
    wave: jnp.ndarray,
    center: jnp.ndarray,
    inner_radius: jnp.ndarray,
    outer_radius: jnp.ndarray,
    sigma: jnp.ndarray,
    inclination: jnp.ndarray,
    q: jnp.ndarray,
    eccentricity: jnp.ndarray,
    apocenter: jnp.ndarray,
    scale: jnp.ndarray,
    offset: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the disk flux for multiple disk profiles in a vectorized manner.
    """
    if len(center) == 0:
        return jnp.zeros_like(wave)

    # Use vmap for disk computation with more efficient batching
    prof_disk_flux = jax.vmap(
        _compute_disk_flux_single, in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    )(
        wave,
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


@jax.jit
def _compute_disk_flux_single(
    wave: jnp.ndarray,
    center: float,
    inner_radius: float,
    outer_radius: float,
    sigma: float,
    inclination: float,
    q: float,
    eccentricity: float,
    apocenter: float,
    scale: float = 1.0,
    offset: float = 0.0,
) -> jnp.ndarray:
    """
    Compute the flux for a single disk profile.
    """
    # Pre-compute frequency conversions
    nu = c_cgs / (wave * 1e-8)
    nu0 = c_cgs / (center * 1e-8)
    X = nu / nu0 - 1

    local_sigma = sigma * 1e5 * nu0 / c_cgs

    res = quad_jax_integrate(
        inner_radius,
        outer_radius,
        0.0,
        2 * jnp.pi - 1e-6,
        jnp.asarray(X),
        inclination,
        local_sigma,
        q,
        eccentricity,
        apocenter,
        nu0,
    )

    max_res = jnp.max(res)
    normalized_res = jnp.where(max_res > 0, res / max_res, res)

    return normalized_res * scale + offset


def create_reparam_config(template: Template) -> dict:
    """Create reparameterization configuration for circular parameters."""
    reparam_config = {}

    for prof in template.disk_profiles + template.line_profiles:
        for param in prof.independent:
            if param.circular:
                samp_name = f"{prof.name}_{param.name}_base"
                reparam_config[samp_name] = CircularReparam()

    return reparam_config


def disk_model(
    template: Template,
    wave: jnp.ndarray,
    flux: Optional[jnp.ndarray] = None,
    flux_err: Optional[jnp.ndarray] = None,
):
    """
    Main disk model function that computes the disk and line fluxes.
    """
    reparam_config = create_reparam_config(template)

    with reparam(config=reparam_config):
        # Sample white noise with better bounds
        white_noise = numpyro.sample(
            "white_noise",
            dist.Uniform(
                template.white_noise.low,
                template.white_noise.high,
            ),
        )

        # Dictionary to store all sampled parameters
        param_mods = {}

        # Sample independent parameters for all profiles
        for prof in template.disk_profiles + template.line_profiles:
            for param in prof.independent:
                samp_name = f"{prof.name}_{param.name}"

                if param.circular:
                    circ_base = numpyro.sample(
                        f"{samp_name}_base",
                        dist.VonMises(concentration=1e-3, loc=0.0),
                    )
                    param_mods[samp_name] = numpyro.deterministic(
                        samp_name, (circ_base + jnp.pi)
                    )

                elif param.distribution == Distribution.UNIFORM:
                    param_mods[samp_name] = numpyro.sample(
                        samp_name, dist.Uniform(param.low, param.high)
                    )
                elif param.distribution == Distribution.LOG_UNIFORM:
                    param_mods[f"{samp_name}_base"] = numpyro.sample(
                        f"{samp_name}_base",
                        # dist.TransformedDistribution(
                        #     dist.Uniform(jnp.log(param.low), jnp.log(param.high)),
                        #     ExpTransform(),
                        # ),
                        dist.Uniform(jnp.log10(param.low), jnp.log10(param.high)),
                    )
                    param_mods[samp_name] = numpyro.deterministic(
                        samp_name,
                        10 ** param_mods[f"{samp_name}_base"],
                    )
                elif param.distribution == Distribution.NORMAL:
                    param_mods[samp_name] = numpyro.sample(
                        samp_name,
                        dist.TruncatedNormal(
                            loc=param.loc,
                            scale=param.scale,
                            low=param.low,
                            high=param.high,
                        ),
                    )
                elif param.distribution == Distribution.LOG_NORMAL:
                    param_mods[f"{samp_name}_base"] = numpyro.sample(
                        f"{samp_name}_base",
                        dist.TruncatedNormal(
                            loc=jnp.log10(param.loc),
                            scale=jnp.log10(param.scale),
                            low=jnp.log10(param.low),
                            high=jnp.log10(param.high),
                        ),
                    )
                    param_mods[samp_name] = numpyro.deterministic(
                        samp_name,
                        10 ** param_mods[f"{samp_name}_base"],
                    )

    # Add fixed parameters
    for prof in template.disk_profiles + template.line_profiles:
        for param in prof.fixed:
            samp_name = f"{prof.name}_{param.name}"
            param_mods[samp_name] = numpyro.deterministic(samp_name, param.value)

    # Add shared parameters
    for prof in template.disk_profiles + template.line_profiles:
        for param in prof.shared:
            samp_name = f"{prof.name}_{param.name}"
            param_mods[samp_name] = numpyro.deterministic(
                samp_name, param_mods[f"{param.shared}_{param.name}"]
            )

    # Compute outer radius for disk profiles
    for prof in template.disk_profiles:
        samp_name = f"{prof.name}_outer_radius"
        param_mods[samp_name] = numpyro.deterministic(
            samp_name,
            param_mods[f"{prof.name}_inner_radius"]
            + param_mods[f"{prof.name}_delta_radius"],
        )

    total_flux, total_disk_flux, total_line_flux = evaluate_model(
        template, wave, param_mods
    )

    total_error = jnp.sqrt(flux_err**2 + total_flux**2 * jnp.exp(2 * white_noise))

    with numpyro.plate("data", wave.shape[0]):
        numpyro.deterministic("disk_flux", total_disk_flux)
        numpyro.deterministic("line_flux", total_line_flux)

        numpyro.sample("total_flux", dist.Normal(total_flux, total_error), obs=flux)


def evaluate_model(
    template: Template, wave: jnp.ndarray | np.ndarray, param_mods: Dict[str, float]
):
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
