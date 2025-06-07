import astropy.constants as const
import astropy.units as u
import numpyro

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro.distributions as dist

from .models.disk import jax_integrate, quad_jax_integrate
from .parser import Distribution, Template
from .utils import truncnorm_ppf
from numpyro.infer.reparam import TransformReparam, LocScaleReparam, CircularReparam


ERR = float(np.finfo(np.float32).tiny)
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value


def _compute_line_flux(
    wave: jnp.ndarray,
    center: float,
    vel_width: float,
    amplitude: float,
    is_gauss: bool = True,
) -> jnp.ndarray:
    fwhm = vel_width / c_kms * center
    delta_lamb = wave - center

    gau = amplitude * jnp.exp(-0.5 * (delta_lamb / (fwhm / 2.35482)) ** 2)
    lor = amplitude * ((fwhm * 0.5) / (delta_lamb**2 + (fwhm * 0.5) ** 2))

    return jnp.where(is_gauss, gau, lor)


def _compute_disk_flux(
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
    use_quad: bool = False,
) -> jnp.ndarray:
    nu = c_cgs / (wave * 1e-8)
    nu0 = c_cgs / (center * 1e-8)
    X = nu / nu0 - 1

    local_sigma = sigma * 1e5 * nu0 / c_cgs

    res = quad_jax_integrate(
        inner_radius,
        outer_radius,
        0,
        2 * jnp.pi - 1e-6,
        jnp.asarray(X),
        inclination,
        local_sigma,
        q,
        eccentricity,
        apocenter,
        nu0,
    )

    return res / jnp.max(res) * scale + offset  # Normalize and apply scale/offset


@jax.jit
def _evaluate_disk_model(
    wave: jnp.ndarray, param_mods: dict, use_quad: bool = False
):
    total_disk_flux = jnp.zeros_like(wave)
    total_line_flux = jnp.zeros_like(wave)

    # Compute disk flux
    if "disk_profiles" in param_mods:
        centers = jnp.array(param_mods["disk_profiles"]["centers"])
        inner_radii = jnp.array(param_mods["disk_profiles"]["inner_radii"])
        outer_radii = jnp.array(param_mods["disk_profiles"]["outer_radii"])
        sigmas = jnp.array(param_mods["disk_profiles"]["sigmas"])
        inclinations = jnp.array(param_mods["disk_profiles"]["inclinations"])
        qs = jnp.array(param_mods["disk_profiles"]["qs"])
        eccentricities = jnp.array(param_mods["disk_profiles"]["eccentricities"])
        apocenters = jnp.array(param_mods["disk_profiles"]["apocenters"])
        scales = jnp.array(param_mods["disk_profiles"]["scales"])
        offsets = jnp.array(param_mods["disk_profiles"]["offsets"])

        prof_disk_flux = jax.vmap(
            _compute_disk_flux,
            in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )(wave, centers, inner_radii, outer_radii, sigmas, inclinations, qs,
            eccentricities, apocenters, scales, offsets)
        
        total_disk_flux = jnp.sum(prof_disk_flux, axis=0)

    # Compute line flux
    if "line_profiles" in param_mods:
        centers = jnp.array(param_mods["line_profiles"]["centers"])
        vel_widths = jnp.array(param_mods["line_profiles"]["vel_widths"])
        amplitudes = jnp.array(param_mods["line_profiles"]["amplitudes"])
        is_gauss = jnp.array(param_mods["line_profiles"]["shapes"])

        prof_line_flux = jax.vmap(
            _compute_line_flux,
            in_axes=(None, 0, 0, 0, 0),
        )(wave, centers, vel_widths, amplitudes, is_gauss)

        total_line_flux = jnp.sum(prof_line_flux, axis=0)

    total_flux = total_disk_flux + total_line_flux

    return total_flux, total_disk_flux, total_line_flux


def evaluate_disk_model(
    template: Template,
    wave: jnp.ndarray,
    param_mods: dict,
    use_quad: bool = False
):
    
    disk_profile_mods = {}
    line_profile_mods = {}

    for prof in template.disk_profiles:
        disk_profile_mods.setdefault("centers", []).append(param_mods[f"{prof.name}_center"])
        disk_profile_mods.setdefault("inner_radii", []).append(param_mods[f"{prof.name}_inner_radius"])
        disk_profile_mods.setdefault("outer_radii", []).append(param_mods[f"{prof.name}_outer_radius"])
        disk_profile_mods.setdefault("sigmas", []).append(param_mods[f"{prof.name}_sigma"])
        disk_profile_mods.setdefault("inclinations", []).append(param_mods[f"{prof.name}_inclination"])
        disk_profile_mods.setdefault("qs", []).append(param_mods[f"{prof.name}_q"])
        disk_profile_mods.setdefault("eccentricities", []).append(param_mods[f"{prof.name}_eccentricity"])
        disk_profile_mods.setdefault("apocenters", []).append(param_mods[f"{prof.name}_apocenter"])
        disk_profile_mods.setdefault("scales", []).append(param_mods.get(f"{prof.name}_scale", 1.0))
        disk_profile_mods.setdefault("offsets", []).append(param_mods.get(f"{prof.name}_offset", 0.0))

    for prof in template.line_profiles:
        line_profile_mods.setdefault("centers", []).append(param_mods[f"{prof.name}_center"])
        line_profile_mods.setdefault("vel_widths", []).append(param_mods[f"{prof.name}_vel_width"])
        line_profile_mods.setdefault("amplitudes", []).append(param_mods[f"{prof.name}_amplitude"])
        line_profile_mods.setdefault("shapes", []).append(prof.shape == 'gaussian')

    trans_param_mods = {}

    if len(disk_profile_mods) > 0:
        trans_param_mods["disk_profiles"] = disk_profile_mods

    if len(line_profile_mods) > 0:
        trans_param_mods["line_profiles"] = line_profile_mods
    
    return _evaluate_disk_model(wave, trans_param_mods, use_quad)


def disk_model(
    template: Template,
    wave: jnp.ndarray,
    flux: jnp.ndarray | None = None,
    flux_err: jnp.ndarray | None = None,
    use_quad: bool = False,
):
    param_mods = {}

    # Build re-parameterization configuration more efficiently
    # reparam_config = {}  # "white_noise": TransformReparam()}

    # for prof in template.all_profiles:
    #     for param in prof.independent:
    #         samp_name = f"{prof.name}_{param.name}"

    #         if param.circular:
    #             # reparam_config[f"{samp_name}_wrap"] = CircularReparam()
    #             pass
    #         else:
    #             reparam_config[samp_name] = TransformReparam()

    # Find shared profiles whose parent doesn't exist; make them independent
    _shared_orphans = {
        prof.name: [
            param
            for param in prof._shared()
            if param.shared not in [p.name for p in template.all_profiles]
        ]
        for prof in template.all_profiles
    }

    # Pre-compute all profiles to iterate over
    for prof in template.all_profiles:
        for param in prof.independent + _shared_orphans[prof.name]:
            samp_name = f"{prof.name}_{param.name}"

            if param.distribution == Distribution.uniform:
                if param.circular:
                    param_mods[f"{samp_name}_x"] = numpyro.sample(
                        f"{samp_name}_x", dist.Normal(0, 1)
                    )
                    param_mods[f"{samp_name}_y"] = numpyro.sample(
                        f"{samp_name}_y", dist.Normal(0, 1)
                    )
                else:
                    base_dist = dist.Uniform(0, 1)
                    param_mods[f"{samp_name}_base"] = numpyro.sample(
                        f"{samp_name}_base", base_dist
                    )
                    param_mods[samp_name] = numpyro.deterministic(
                        samp_name,
                        jsp.stats.uniform.ppf(
                            param_mods[f"{samp_name}_base"],
                            loc=param.low,
                            scale=param.high - param.low,
                        ),
                    )
            elif param.distribution == Distribution.log_uniform:
                base_dist = dist.Uniform(0, 1)
                param_mods[f"{samp_name}_base"] = numpyro.sample(
                    f"{samp_name}_base", base_dist
                )
                param_mods[samp_name] = numpyro.deterministic(
                    samp_name,
                    10
                    ** jsp.stats.uniform.ppf(
                        param_mods[f"{samp_name}_base"],
                        loc=jnp.log10(param.low),
                        scale=jnp.log10(param.high) - jnp.log10(param.low),
                    ),
                )
            elif param.distribution == Distribution.normal:
                if param.circular:
                    param_mods[f"{samp_name}_x"] = numpyro.sample(
                        f"{samp_name}_x", dist.Normal(jnp.cos(param.loc), 1)
                    )
                    param_mods[f"{samp_name}_y"] = numpyro.sample(
                        f"{samp_name}_y", dist.Normal(jnp.sin(param.loc), 1)
                    )
                else:
                    base_dist = dist.Uniform(0, 1)
                    param_mods[f"{samp_name}_base"] = numpyro.sample(
                        f"{samp_name}_base", base_dist
                    )
                    param_mods[samp_name] = numpyro.deterministic(
                        samp_name,
                        truncnorm_ppf(
                            param_mods[f"{samp_name}_base"],
                            loc=param.loc,
                            scale=param.scale,
                            lower_limit=param.low,
                            upper_limit=param.high,
                        ),
                    )
            elif param.distribution == Distribution.log_normal:
                base_dist = dist.Uniform(0, 1)
                param_mods[f"{samp_name}_base"] = numpyro.sample(
                    f"{samp_name}_base", base_dist
                )
                param_mods[samp_name] = numpyro.deterministic(
                    samp_name,
                    10
                    ** truncnorm_ppf(
                        param_mods[f"{samp_name}_base"],
                        loc=jnp.log10(param.loc),
                        scale=jnp.log10(param.scale),
                        lower_limit=jnp.log10(param.low),
                        upper_limit=jnp.log10(param.high),
                    ),
                )
            else:
                raise ValueError(
                    f"Invalid distribution: {param.distribution} for parameter {param.name}"
                )

    white_noise = numpyro.sample(
        "white_noise", dist.Uniform(template.white_noise.low, template.white_noise.high)
    )

    # Include fixed fields
    for prof in template.all_profiles:
        for param in prof.fixed:
            param_mods[f"{prof.name}_{param.name}"] = numpyro.deterministic(
                f"{prof.name}_{param.name}", param.value
            )

        # Sample all shared parameters
        for param in [x for x in prof.shared if x not in _shared_orphans[prof.name]]:
            samp_name = f"{prof.name}_{param.name}"
            param_mods[samp_name] = numpyro.deterministic(
                samp_name, param_mods[f"{param.shared}_{param.name}"]
            )

    # Define outer radius for each disk profile
    for prof in template.disk_profiles:
        param_name = f"{prof.name}_outer_radius"
        ir = param_mods[f"{prof.name}_inner_radius"]
        dr = param_mods[f"{prof.name}_delta_radius"]
        param_mods[param_name] = numpyro.deterministic(param_name, ir + dr)

    # Unwrap apocenter to the proper range
    for prof in template.disk_profiles:
        for param in prof.independent:
            if param.circular:
                param_name = f"{prof.name}_{param.name}"
                x = param_mods[f"{param_name}_x"]
                y = param_mods[f"{param_name}_y"]
                r = jnp.sqrt(x**2 + y**2) + 1e-6
                param_mods[param_name] = numpyro.deterministic(
                    param_name, jnp.arctan2(y / r, x / r) % (2 * jnp.pi)
                )

    total_flux, total_disk_flux, total_line_flux = evaluate_disk_model(
        template, wave, param_mods, use_quad
    )

    # Construct total error
    flux_err = flux_err if flux_err is not None else jnp.zeros_like(wave)
    total_error = jnp.sqrt(flux_err**2 + total_flux**2 * jnp.exp(2 * white_noise))

    with numpyro.plate("data", wave.shape[0]):
        numpyro.deterministic("disk_flux", total_disk_flux)
        numpyro.deterministic("line_flux", total_line_flux)
        numpyro.sample("total_flux", dist.Normal(total_flux, total_error), obs=flux)
