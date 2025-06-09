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
) -> jnp.ndarray:
    nu = c_cgs / (wave * 1e-8)
    nu0 = c_cgs / (center * 1e-8)
    X = nu / nu0 - 1

    local_sigma = sigma * 1e5 * nu0 / c_cgs

    res = quad_jax_integrate(
        inner_radius.squeeze(),
        outer_radius.squeeze(),
        0.0,
        2 * jnp.pi - 1e-6,
        jnp.asarray(X),
        inclination.squeeze(),
        local_sigma.squeeze(),
        q.squeeze(),
        eccentricity.squeeze(),
        apocenter.squeeze(),
        nu0.squeeze(),
    )

    return res / jnp.max(res) * scale + offset  # Normalize and apply scale/offset


@jax.jit
def _evaluate_disk_model(wave: jnp.ndarray, disk_params: dict, line_params: dict):
    # Compute disk flux
    prof_disk_flux = jax.vmap(
        _compute_disk_flux,
        in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
    )(
        wave,
        disk_params["centers"],
        disk_params["inner_radii"],
        disk_params["outer_radii"],
        disk_params["sigmas"],
        disk_params["inclinations"],
        disk_params["qs"],
        disk_params["eccentricities"],
        disk_params["apocenters"],
        disk_params["scales"],
        disk_params["offsets"],
    )

    total_disk_flux = jnp.sum(prof_disk_flux, axis=0)

    # Compute line flux
    prof_line_flux = jax.vmap(
        _compute_line_flux,
        in_axes=(None, 0, 0, 0, 0),
    )(
        wave,
        line_params["centers"],
        line_params["vel_widths"],
        line_params["amplitudes"],
        line_params["shapes"],
    )

    total_line_flux = jnp.sum(prof_line_flux, axis=0)

    total_flux = total_disk_flux + total_line_flux

    return total_flux, total_disk_flux, total_line_flux


def evaluate_disk_model(template: Template, wave: jnp.ndarray, param_mods: dict):
    disk_names = template.disk_names
    line_names = template.line_names

    disk_params = {
        "centers": jnp.array([param_mods[f"{name}_center"] for name in disk_names]),
        "inner_radii": jnp.array(
            [param_mods[f"{name}_inner_radius"] for name in disk_names]
        ),
        "outer_radii": jnp.array(
            [param_mods[f"{name}_outer_radius"] for name in disk_names]
        ),
        "sigmas": jnp.array([param_mods[f"{name}_sigma"] for name in disk_names]),
        "inclinations": jnp.array(
            [param_mods[f"{name}_inclination"] for name in disk_names]
        ),
        "qs": jnp.array([param_mods[f"{name}_q"] for name in disk_names]),
        "eccentricities": jnp.array(
            [param_mods[f"{name}_eccentricity"] for name in disk_names]
        ),
        "apocenters": jnp.array(
            [param_mods[f"{name}_apocenter"] for name in disk_names]
        ),
        "scales": jnp.array([param_mods[f"{name}_scale"] for name in disk_names]),
        "offsets": jnp.array([param_mods[f"{name}_offset"] for name in disk_names]),
    }

    line_params = {
        "centers": jnp.array([param_mods[f"{name}_center"] for name in line_names]),
        "vel_widths": jnp.array(
            [param_mods[f"{name}_vel_width"] for name in line_names]
        ),
        "amplitudes": jnp.array(
            [param_mods[f"{name}_amplitude"] for name in line_names]
        ),
        "shapes": jnp.array(
            [prof.shape == "gaussian" for prof in template.line_profiles]
        ),
    }

    return _evaluate_disk_model(wave, disk_params, line_params)


def disk_model(
    template: Template,
    wave: jnp.ndarray,
    flux: jnp.ndarray | None = None,
    flux_err: jnp.ndarray | None = None,
):
    param_mods = {}

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
    for prof in template.all_profiles:
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
        template, wave, param_mods
    )

    # Construct total error
    flux_err = flux_err if flux_err is not None else jnp.zeros_like(wave)
    total_error = jnp.sqrt(flux_err**2 + total_flux**2 * jnp.exp(2 * white_noise))

    with numpyro.plate("data", wave.shape[0]):
        numpyro.deterministic("disk_flux", total_disk_flux)
        numpyro.deterministic("line_flux", total_line_flux)
        numpyro.sample("total_flux", dist.Normal(total_flux, total_error), obs=flux)
