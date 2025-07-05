import astropy.constants as const
import astropy.units as u
import numpyro

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro.distributions as dist
from typing import Dict, List, Tuple, Optional

import flax

from .models.disk import jax_integrate, quad_jax_integrate
from .parser import Distribution, Template, Shape, Parameter
from .utils import truncnorm_ppf


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

    OPTIMIZATION: Use more efficient computation and avoid redundant operations.
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
    # Clamp to avoid overflow in exp
    gau_exp = jnp.clip(gau_exp, -50.0, 0.0)
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

    OPTIMIZATION: Use more efficient vmapping and avoid redundant computations.
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

    OPTIMIZATION: Pre-compute constants and use more efficient operations.
    """
    # Pre-compute frequency conversions
    nu = c_cgs / (wave * 1e-8)
    nu0 = c_cgs / (center * 1e-8)
    X = nu / nu0 - 1

    local_sigma = sigma * 1e5 * nu0 / c_cgs

    res = jax_integrate(
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


@flax.struct.dataclass
class ParameterCache:
    """
    Enhanced cache with pre-computed arrays for better performance.
    """

    disk_names: List[str]
    line_names: List[str]
    n_disks: int
    n_lines: int
    param_groups: Dict[str, List[Tuple[str, Parameter]]]
    fixed_params: List[Tuple[str, Parameter]]
    shared_params: List[Tuple[str, Parameter]]
    line_shapes: jnp.ndarray
    disk_param_indices: Dict[str, jnp.ndarray]
    line_param_indices: Dict[str, jnp.ndarray]

    @classmethod
    def create(cls, template: Template):
        """Enhanced cache creation with parameter indexing."""
        disk_names = [prof.name for prof in template.disk_profiles]
        line_names = [prof.name for prof in template.line_profiles]
        n_disks = len(disk_names)
        n_lines = len(line_names)

        param_groups = cls._compute_param_groups(template)
        fixed_params = cls._collect_fixed_params(template)
        shared_params = cls._collect_shared_params(template)

        line_shapes = (
            jnp.array([prof.shape == Shape.GAUSSIAN for prof in template.line_profiles])
            if line_names
            else jnp.array([])
        )

        # Pre-compute parameter indices
        disk_param_indices = {}
        line_param_indices = {}

        for i, name in enumerate(disk_names):
            disk_param_indices[name] = i

        for i, name in enumerate(line_names):
            line_param_indices[name] = i

        return cls(
            disk_names=disk_names,
            line_names=line_names,
            n_disks=n_disks,
            n_lines=n_lines,
            param_groups=param_groups,
            fixed_params=fixed_params,
            shared_params=shared_params,
            line_shapes=line_shapes,
            disk_param_indices=disk_param_indices,
            line_param_indices=line_param_indices,
        )

    @staticmethod
    def _compute_param_groups(
        template: Template,
    ) -> Dict[str, List[Tuple[str, Parameter]]]:
        """Pre-compute parameter groups for batch sampling."""
        param_groups = {}
        all_profiles = template.disk_profiles + template.line_profiles

        for prof in all_profiles:
            for param in prof.independent:
                if param.name not in param_groups:
                    param_groups[param.name] = []
                param_groups[param.name].append((prof.name, param))

        return param_groups

    @staticmethod
    def _collect_fixed_params(template: Template) -> List[Tuple[str, Parameter]]:
        """Collect all fixed parameters."""
        fixed_params = []
        all_profiles = template.disk_profiles + template.line_profiles

        for prof in all_profiles:
            for param in prof.fixed:
                fixed_params.append((prof.name, param))

        return fixed_params

    @staticmethod
    def _collect_shared_params(template: Template) -> List[Tuple[str, Parameter]]:
        """Collect all shared parameters."""
        shared_params = []
        all_profiles = template.disk_profiles + template.line_profiles

        for prof in all_profiles:
            for param in prof.shared:
                shared_params.append((prof.name, param))

        return shared_params


def _sample_parameter_batch_optimized(
    param_batch: List[Tuple[str, Parameter]], base_name: str
) -> Dict[str, jnp.ndarray]:
    """
    Optimized parameter batch sampling with reduced overhead.
    """
    param_mods = {}

    if not param_batch:
        return param_mods

    for prof_name, param in param_batch:
        samp_name = f"{prof_name}_{param.name}"

        if param.circular:
            circular_x = numpyro.sample(f"{samp_name}_circ_x_base", dist.Normal(0, 1))
            circular_y = numpyro.sample(f"{samp_name}_circ_y_base", dist.Normal(0, 1))

            if param.distribution == Distribution.NORMAL:
                circular_x = circular_x * param.scale + jnp.cos(param.loc)
                circular_y = circular_y * param.scale + jnp.sin(param.loc)

            r = jnp.sqrt(circular_x**2 + circular_y**2) + 1e-6
            value = jnp.arctan2(circular_y / r, circular_x / r) % (2 * jnp.pi)
            param_mods[samp_name] = numpyro.deterministic(samp_name, value)

        elif param.distribution == Distribution.UNIFORM:
            param_mods[samp_name] = numpyro.sample(
                samp_name, dist.Uniform(param.low, param.high)
            )

        elif param.distribution == Distribution.LOG_UNIFORM:
            log_low = jnp.log(param.low)
            log_high = jnp.log(param.high)
            base_log_uniform = numpyro.sample(
                f"{samp_name}_base",
                dist.Uniform(log_low, log_high),
            )
            param_mods[samp_name] = numpyro.deterministic(
                samp_name, jnp.exp(base_log_uniform)
            )

        elif param.distribution == Distribution.NORMAL:
            param_mods[samp_name] = numpyro.sample(
                samp_name,
                dist.TruncatedNormal(
                    loc=param.loc, scale=param.scale, low=param.low, high=param.high
                ),
            )

        elif param.distribution == Distribution.LOG_NORMAL:
            log_loc = jnp.log(param.loc)
            log_scale = jnp.log(param.scale)
            log_low = jnp.log(param.low)
            log_high = jnp.log(param.high)

            base_log_normal = numpyro.sample(
                f"{samp_name}_base",
                dist.TruncatedNormal(
                    loc=log_loc,
                    scale=log_scale,
                    low=log_low,
                    high=log_high,
                ),
            )
            param_mods[samp_name] = numpyro.deterministic(
                samp_name, jnp.exp(base_log_normal)
            )

        else:
            raise ValueError(f"Unsupported distribution type: {param.distribution}")

    return param_mods


def evaluate_model(
    template: Template,
    wave: jnp.ndarray | np.ndarray,
    param_mods: Dict[str, float],
    cache: Optional[ParameterCache] = None,
):
    if cache is None:
        cache = ParameterCache.create(template)

    total_disk_flux = jnp.zeros_like(wave)
    total_line_flux = jnp.zeros_like(wave)

    if cache.n_disks > 0:
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
                [param_mods[f"{name}_{param_type}"] for name in cache.disk_names]
            )

        total_disk_flux = _compute_disk_flux_vectorized(wave, **disk_arrays)

    if cache.n_lines > 0:
        # Use more efficient array construction
        line_arrays = {
            "center": jnp.array(
                [param_mods[f"{name}_center"] for name in cache.line_names]
            ),
            "vel_width": jnp.array(
                [param_mods[f"{name}_vel_width"] for name in cache.line_names]
            ),
            "amplitude": jnp.array(
                [param_mods[f"{name}_amplitude"] for name in cache.line_names]
            ),
            "shape": cache.line_shapes,
        }

        total_line_flux = _compute_line_flux_vectorized(wave, **line_arrays)

    total_flux = total_disk_flux + total_line_flux

    return total_flux, total_disk_flux, total_line_flux


def disk_model_optimized(
    template: Template,
    wave: jnp.ndarray,
    flux: Optional[jnp.ndarray] = None,
    flux_err: Optional[jnp.ndarray] = None,
    cache: Optional[ParameterCache] = None,
):
    """
    Optimized disk model with caching and vectorized operations.
    """
    if cache is None:
        cache = ParameterCache.create(template)

    param_mods = {}

    # Sample parameters in optimized batches
    for param_name, param_batch in cache.param_groups.items():
        batch_params = _sample_parameter_batch_optimized(param_batch, param_name)
        param_mods.update(batch_params)

    # Sample white noise with better bounds
    white_noise = numpyro.sample(
        "white_noise",
        dist.Uniform(
            template.white_noise.low,
            template.white_noise.high,
        ),
    )

    # Add fixed parameters
    for prof_name, param in cache.fixed_params:
        samp_name = f"{prof_name}_{param.name}"
        param_mods[samp_name] = numpyro.deterministic(samp_name, param.value)

    # Add shared parameters
    for prof_name, param in cache.shared_params:
        samp_name = f"{prof_name}_{param.name}"
        param_mods[samp_name] = numpyro.deterministic(
            samp_name, param_mods[f"{param.shared}_{param.name}"]
        )

    # Compute outer radius for disk profiles
    for prof_name in cache.disk_names:
        samp_name = f"{prof_name}_outer_radius"
        param_mods[samp_name] = numpyro.deterministic(
            samp_name,
            param_mods[f"{prof_name}_inner_radius"]
            + param_mods[f"{prof_name}_delta_radius"],
        )

    total_flux, total_disk_flux, total_line_flux = evaluate_model(
        template, wave, param_mods, cache
    )

    if flux_err is not None:
        base_error_sq = flux_err**2
    else:
        base_error_sq = jnp.zeros_like(wave)

    noise_factor = jnp.exp(2 * white_noise)
    total_error = jnp.sqrt(base_error_sq + total_flux**2 * noise_factor)

    with numpyro.plate("data", wave.shape[0]):
        numpyro.deterministic("disk_flux", total_disk_flux)
        numpyro.deterministic("line_flux", total_line_flux)

        numpyro.sample("total_flux", dist.Normal(total_flux, total_error), obs=flux)


def create_optimized_model(template: Template):
    """
    Factory function to create optimized model with cached metadata.
    """
    cache = ParameterCache.create(template)

    def model(wave, flux=None, flux_err=None):
        return disk_model_optimized(template, wave, flux, flux_err, cache)

    return model
