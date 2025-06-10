import astropy.constants as const
import astropy.units as u
import numpyro

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro.distributions as dist

from .models.disk import jax_integrate, quad_jax_integrate
from .parser import Distribution, Template, Shape
from .utils import truncnorm_ppf


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

    return res / jnp.max(res) * scale + offset


@jax.jit
def _compute_model_fluxes(
    wave: jnp.ndarray,
    disk_centers: jnp.ndarray,
    disk_inner_radii: jnp.ndarray,
    disk_outer_radii: jnp.ndarray,
    disk_sigmas: jnp.ndarray,
    disk_inclinations: jnp.ndarray,
    disk_qs: jnp.ndarray,
    disk_eccentricities: jnp.ndarray,
    disk_apocenters: jnp.ndarray,
    disk_scales: jnp.ndarray,
    disk_offsets: jnp.ndarray,
    line_centers: jnp.ndarray,
    line_vel_widths: jnp.ndarray,
    line_amplitudes: jnp.ndarray,
    line_shapes: jnp.ndarray,
):
    """Compute disk and line fluxes directly from parameter arrays."""
    # Compute disk flux only if we have disk profiles
    total_disk_flux = jnp.zeros_like(wave)
    if len(disk_centers) > 0:
        prof_disk_flux = jax.vmap(
            _compute_disk_flux,
            in_axes=(None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0),
        )(
            wave,
            disk_centers,
            disk_inner_radii,
            disk_outer_radii,
            disk_sigmas,
            disk_inclinations,
            disk_qs,
            disk_eccentricities,
            disk_apocenters,
            disk_scales,
            disk_offsets,
        )
        total_disk_flux = jnp.sum(prof_disk_flux, axis=0)

    # Compute line flux only if we have line profiles
    total_line_flux = jnp.zeros_like(wave)
    if len(line_centers) > 0:
        prof_line_flux = jax.vmap(
            _compute_line_flux,
            in_axes=(None, 0, 0, 0, 0),
        )(
            wave,
            line_centers,
            line_vel_widths,
            line_amplitudes,
            line_shapes,
        )
        total_line_flux = jnp.sum(prof_line_flux, axis=0)

    total_flux = total_disk_flux + total_line_flux
    return total_flux, total_disk_flux, total_line_flux


def _sample_parameter_batch(param_batch, profile_names, base_name):
    """Sample a batch of parameters with the same distribution type."""
    param_mods = {}

    if not param_batch:
        return param_mods

    # Group by distribution type for vectorized sampling
    uniform_params = []
    log_uniform_params = []
    normal_params = []
    log_normal_params = []
    circular_params = []

    for prof_name, param in param_batch:
        if param.circular:
            circular_params.append((prof_name, param))
        elif param.distribution == Distribution.UNIFORM:
            uniform_params.append((prof_name, param))
        elif param.distribution == Distribution.LOG_UNIFORM:
            log_uniform_params.append((prof_name, param))
        elif param.distribution == Distribution.NORMAL:
            normal_params.append((prof_name, param))
        elif param.distribution == Distribution.LOG_NORMAL:
            log_normal_params.append((prof_name, param))

    # Vectorized sampling for each distribution type
    if uniform_params:
        n_uniform = len(uniform_params)
        uniform_bases = numpyro.sample(
            f"{base_name}_uniform_base", dist.Uniform(0, 1).expand([n_uniform])
        )

        for i, (prof_name, param) in enumerate(uniform_params):
            samp_name = f"{prof_name}_{param.name}"
            param_mods[samp_name] = numpyro.deterministic(
                samp_name,
                jsp.stats.uniform.ppf(
                    uniform_bases[i],
                    loc=param.low,
                    scale=param.high - param.low,
                ),
            )

    if log_uniform_params:
        n_log_uniform = len(log_uniform_params)
        log_uniform_bases = numpyro.sample(
            f"{base_name}_log_uniform_base", dist.Uniform(0, 1).expand([n_log_uniform])
        )

        for i, (prof_name, param) in enumerate(log_uniform_params):
            samp_name = f"{prof_name}_{param.name}"
            param_mods[samp_name] = numpyro.deterministic(
                samp_name,
                10
                ** jsp.stats.uniform.ppf(
                    log_uniform_bases[i],
                    loc=jnp.log10(param.low),
                    scale=jnp.log10(param.high) - jnp.log10(param.low),
                ),
            )

    if normal_params:
        n_normal = len(normal_params)
        normal_bases = numpyro.sample(
            f"{base_name}_normal_base", dist.Uniform(0, 1).expand([n_normal])
        )

        for i, (prof_name, param) in enumerate(normal_params):
            samp_name = f"{prof_name}_{param.name}"
            param_mods[samp_name] = numpyro.deterministic(
                samp_name,
                truncnorm_ppf(
                    normal_bases[i],
                    loc=param.loc,
                    scale=param.scale,
                    lower_limit=param.low,
                    upper_limit=param.high,
                ),
            )

    if log_normal_params:
        n_log_normal = len(log_normal_params)
        log_normal_bases = numpyro.sample(
            f"{base_name}_log_normal_base", dist.Uniform(0, 1).expand([n_log_normal])
        )

        for i, (prof_name, param) in enumerate(log_normal_params):
            samp_name = f"{prof_name}_{param.name}"
            param_mods[samp_name] = numpyro.deterministic(
                samp_name,
                10
                ** truncnorm_ppf(
                    log_normal_bases[i],
                    loc=jnp.log10(param.loc),
                    scale=jnp.log10(param.scale),
                    lower_limit=jnp.log10(param.low),
                    upper_limit=jnp.log10(param.high),
                ),
            )

    # Handle circular parameters
    if circular_params:
        n_circular = len(circular_params)
        circular_x = numpyro.sample(
            f"{base_name}_circular_x", dist.Normal(0, 1).expand([n_circular])
        )
        circular_y = numpyro.sample(
            f"{base_name}_circular_y", dist.Normal(0, 1).expand([n_circular])
        )

        for i, (prof_name, param) in enumerate(circular_params):
            samp_name = f"{prof_name}_{param.name}"
            x = circular_x[i]
            y = circular_y[i]

            if param.distribution == Distribution.NORMAL:
                x = x + jnp.cos(param.loc)
                y = y + jnp.sin(param.loc)

            r = jnp.sqrt(x**2 + y**2) + 1e-6
            param_mods[samp_name] = numpyro.deterministic(
                samp_name, jnp.arctan2(y / r, x / r) % (2 * jnp.pi)
            )

    return param_mods


def disk_model(
    template: Template,
    wave: jnp.ndarray,
    flux: jnp.ndarray | None = None,
    flux_err: jnp.ndarray | None = None,
):
    param_mods = {}

    # Collect all independent parameters by type for batch processing
    all_profiles = template.disk_profiles + template.line_profiles

    # Group parameters by name for vectorized sampling
    param_groups = {}
    for prof in all_profiles:
        for param in prof.independent:
            if param.name not in param_groups:
                param_groups[param.name] = []
            param_groups[param.name].append((prof.name, param))

    # Sample parameters in batches by parameter name
    for param_name, param_batch in param_groups.items():
        batch_params = _sample_parameter_batch(
            param_batch, [prof.name for prof in all_profiles], param_name
        )
        param_mods.update(batch_params)

    # Sample white noise
    white_noise = numpyro.sample(
        "white_noise", dist.Uniform(template.white_noise.low, template.white_noise.high)
    )

    # Add fixed parameters
    for prof in all_profiles:
        for param in prof.fixed:
            param_mods[f"{prof.name}_{param.name}"] = numpyro.deterministic(
                f"{prof.name}_{param.name}", param.value
            )

    # Add shared parameters
    for prof in all_profiles:
        for param in prof.shared:
            samp_name = f"{prof.name}_{param.name}"
            param_mods[samp_name] = numpyro.deterministic(
                samp_name, param_mods[f"{param.shared}_{param.name}"]
            )

    # Compute outer radius for disk profiles
    for prof in template.disk_profiles:
        param_name = f"{prof.name}_outer_radius"
        param_mods[param_name] = numpyro.deterministic(
            param_name,
            param_mods[f"{prof.name}_inner_radius"]
            + param_mods[f"{prof.name}_delta_radius"],
        )

    # Collect parameter arrays directly for model evaluation
    disk_names = [prof.name for prof in template.disk_profiles]
    line_names = [prof.name for prof in template.line_profiles]

    # Create parameter arrays directly - avoid dictionary lookups in JIT
    disk_centers = (
        jnp.array([param_mods[f"{name}_center"] for name in disk_names])
        if disk_names
        else jnp.array([])
    )
    disk_inner_radii = (
        jnp.array([param_mods[f"{name}_inner_radius"] for name in disk_names])
        if disk_names
        else jnp.array([])
    )
    disk_outer_radii = (
        jnp.array([param_mods[f"{name}_outer_radius"] for name in disk_names])
        if disk_names
        else jnp.array([])
    )
    disk_sigmas = (
        jnp.array([param_mods[f"{name}_sigma"] for name in disk_names])
        if disk_names
        else jnp.array([])
    )
    disk_inclinations = (
        jnp.array([param_mods[f"{name}_inclination"] for name in disk_names])
        if disk_names
        else jnp.array([])
    )
    disk_qs = (
        jnp.array([param_mods[f"{name}_q"] for name in disk_names])
        if disk_names
        else jnp.array([])
    )
    disk_eccentricities = (
        jnp.array([param_mods[f"{name}_eccentricity"] for name in disk_names])
        if disk_names
        else jnp.array([])
    )
    disk_apocenters = (
        jnp.array([param_mods[f"{name}_apocenter"] for name in disk_names])
        if disk_names
        else jnp.array([])
    )
    disk_scales = (
        jnp.array([param_mods[f"{name}_scale"] for name in disk_names])
        if disk_names
        else jnp.array([])
    )
    disk_offsets = (
        jnp.array([param_mods[f"{name}_offset"] for name in disk_names])
        if disk_names
        else jnp.array([])
    )

    line_centers = (
        jnp.array([param_mods[f"{name}_center"] for name in line_names])
        if line_names
        else jnp.array([])
    )
    line_vel_widths = (
        jnp.array([param_mods[f"{name}_vel_width"] for name in line_names])
        if line_names
        else jnp.array([])
    )
    line_amplitudes = (
        jnp.array([param_mods[f"{name}_amplitude"] for name in line_names])
        if line_names
        else jnp.array([])
    )
    line_shapes = (
        jnp.array([prof.shape == Shape.GAUSSIAN for prof in template.line_profiles])
        if line_names
        else jnp.array([])
    )

    # Evaluate model directly
    total_flux, total_disk_flux, total_line_flux = _compute_model_fluxes(
        wave,
        disk_centers,
        disk_inner_radii,
        disk_outer_radii,
        disk_sigmas,
        disk_inclinations,
        disk_qs,
        disk_eccentricities,
        disk_apocenters,
        disk_scales,
        disk_offsets,
        line_centers,
        line_vel_widths,
        line_amplitudes,
        line_shapes,
    )

    # Construct total error
    flux_err = flux_err if flux_err is not None else jnp.zeros_like(wave)
    total_error = jnp.sqrt(flux_err**2 + total_flux**2 * jnp.exp(2 * white_noise))

    with numpyro.plate("data", wave.shape[0]):
        numpyro.deterministic("disk_flux", total_disk_flux)
        numpyro.deterministic("line_flux", total_line_flux)
        numpyro.sample("total_flux", dist.Normal(total_flux, total_error), obs=flux)
