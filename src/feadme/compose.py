import astropy.constants as const
import astropy.units as u
import numpyro

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np
import numpyro.distributions as dist
from typing import Dict, List, Tuple

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
    centers: jnp.ndarray,
    vel_widths: jnp.ndarray,
    amplitudes: jnp.ndarray,
    shapes: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the line flux for multiple spectral lines in a vectorized manner.

    Parameters
    ----------
    wave : jnp.ndarray
        Wavelength array (in Angstroms).
    centers : jnp.ndarray
        Central wavelengths of the spectral lines (in Angstroms).
    vel_widths : jnp.ndarray
        Velocity widths of the spectral lines (in km/s).
    amplitudes : jnp.ndarray
        Amplitudes of the spectral lines.
    shapes : jnp.ndarray
        Boolean array indicating the shape of each line:
        `True` for Gaussian, `False` for Lorentzian.

    Returns
    -------
    jnp.ndarray
        The computed line flux for each wavelength in the input array.
    """
    if len(centers) == 0:
        return jnp.zeros_like(wave)

    # Broadcast for vectorized computation: (n_wave, n_lines)
    wave_bc = wave[:, None]
    centers_bc = centers[None, :]
    vel_widths_bc = vel_widths[None, :]
    amplitudes_bc = amplitudes[None, :]
    shapes_bc = shapes[None, :]

    fwhm = vel_widths_bc / c_kms * centers_bc
    delta_lamb = wave_bc - centers_bc

    # Gaussian profiles
    gau = amplitudes_bc * jnp.exp(-0.5 * (delta_lamb / (fwhm / 2.35482)) ** 2)

    # Lorentzian profiles
    lor = amplitudes_bc * ((fwhm * 0.5) / (delta_lamb**2 + (fwhm * 0.5) ** 2))

    # Select based on shape
    line_fluxes = jnp.where(shapes_bc, gau, lor)

    # Sum over all lines
    return jnp.sum(line_fluxes, axis=1)


@jax.jit
def _compute_disk_flux_vectorized(
    wave: jnp.ndarray,
    centers: jnp.ndarray,
    inner_radii: jnp.ndarray,
    outer_radii: jnp.ndarray,
    sigmas: jnp.ndarray,
    inclinations: jnp.ndarray,
    qs: jnp.ndarray,
    eccentricities: jnp.ndarray,
    apocenters: jnp.ndarray,
    scales: jnp.ndarray,
    offsets: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute the disk flux for multiple disk profiles in a vectorized manner.

    Parameters
    ----------
    wave : jnp.ndarray
        Wavelength array (in Angstroms).
    centers : jnp.ndarray
        Central wavelengths of the disk profiles (in Angstroms).
    inner_radii : jnp.ndarray
        Inner radii of the disk profiles (in arbitrary units).
    outer_radii : jnp.ndarray
        Outer radii of the disk profiles (in arbitrary units).
    sigmas : jnp.ndarray
        Line broadening parameters for the disk profiles.
    inclinations : jnp.ndarray
        Inclination angles of the disks (in radians).
    qs : jnp.ndarray
        Power-law indices for the radial intensity profiles.
    eccentricities : jnp.ndarray
        Eccentricities of the disk profiles.
    apocenters : jnp.ndarray
        Apocenter angles of the disk profiles (in radians).
    scales : jnp.ndarray
        Scaling factors for the disk profiles.
    offsets : jnp.ndarray
        Offset values for the disk profiles.

    Returns
    -------
    jnp.ndarray
        The computed disk flux for each wavelength in the input array.
    """
    if len(centers) == 0:
        return jnp.zeros_like(wave)

    # Use vmap for disk computation (already optimized in original)
    prof_disk_flux = jax.vmap(
        lambda c, ir, or_, s, i, q, e, a, sc, o: _compute_disk_flux_single(
            wave, c, ir, or_, s, i, q, e, a, sc, o
        )
    )(
        centers,
        inner_radii,
        outer_radii,
        sigmas,
        inclinations,
        qs,
        eccentricities,
        apocenters,
        scales,
        offsets,
    )

    return jnp.sum(prof_disk_flux, axis=0)


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

    This function calculates the flux contribution of a single disk profile
    based on its parameters, using numerical integration over the disk's
    radial and azimuthal extent.

    Parameters
    ----------
    wave : jnp.ndarray
        Wavelength array (in Angstroms).
    center : float
        Central wavelength of the disk profile (in Angstroms).
    inner_radius : float
        Inner radius of the disk (in arbitrary units).
    outer_radius : float
        Outer radius of the disk (in arbitrary units).
    sigma : float
        Line broadening parameter for the disk profile.
    inclination : float
        Inclination angle of the disk (in radians).
    q : float
        Power-law index for the radial intensity profile.
    eccentricity : float
        Eccentricity of the disk profile.
    apocenter : float
        Apocenter angle of the disk profile (in radians).
    scale : float, optional
        Scaling factor for the disk profile, by default 1.0.
    offset : float, optional
        Offset value for the disk profile, by default 0.0.

    Returns
    -------
    jnp.ndarray
        The computed flux for the disk profile at each wavelength in the input array.
    """
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

    return res / jnp.max(res) * scale + offset


def evaluate_model(
    template: Template,
    wave: jnp.ndarray | np.ndarray,
    param_mods: Dict[str, float],
):
    """
    Evaluate the model fluxes for a given template and parameter modifications.

    Parameters
    ----------
    template : Template
        The model template containing disk and line profile definitions.
    wave : jnp.ndarray or np.ndarray
        Wavelength array (in Angstroms) at which to evaluate the model.
    param_mods : dict of str to float
        Dictionary mapping parameter names to their modified values.

    Returns
    -------
    tuple of jnp.ndarray
        A tuple containing:
        - total_flux : jnp.ndarray
            The sum of disk and line fluxes at each wavelength.
        - total_disk_flux : jnp.ndarray
            The disk flux at each wavelength.
        - total_line_flux : jnp.ndarray
            The line flux at each wavelength.
    """
    total_disk_flux = jnp.zeros_like(wave)
    total_line_flux = jnp.zeros_like(wave)

    disk_params = {
        "centers": jnp.array(
            [
                param_mods[f"{prof.name}_center"]
                for prof in template.disk_profiles
                if f"{prof.name}_center" in param_mods
            ]
        ),
        "inner_radii": jnp.array(
            [
                param_mods[f"{prof.name}_inner_radius"]
                for prof in template.disk_profiles
                if f"{prof.name}_inner_radius" in param_mods
            ]
        ),
        "outer_radii": jnp.array(
            [
                param_mods[f"{prof.name}_outer_radius"]
                for prof in template.disk_profiles
                if f"{prof.name}_outer_radius" in param_mods
            ]
        ),
        "sigmas": jnp.array(
            [
                param_mods[f"{prof.name}_sigma"]
                for prof in template.disk_profiles
                if f"{prof.name}_sigma" in param_mods
            ]
        ),
        "inclinations": jnp.array(
            [
                param_mods[f"{prof.name}_inclination"]
                for prof in template.disk_profiles
                if f"{prof.name}_inclination" in param_mods
            ]
        ),
        "qs": jnp.array(
            [
                param_mods[f"{prof.name}_q"]
                for prof in template.disk_profiles
                if f"{prof.name}_q" in param_mods
            ]
        ),
        "eccentricities": jnp.array(
            [
                param_mods[f"{prof.name}_eccentricity"]
                for prof in template.disk_profiles
                if f"{prof.name}_eccentricity" in param_mods
            ]
        ),
        "apocenters": jnp.array(
            [
                param_mods[f"{prof.name}_apocenter"]
                for prof in template.disk_profiles
                if f"{prof.name}_apocenter" in param_mods
            ]
        ),
        "scales": jnp.array(
            [
                param_mods[f"{prof.name}_scale"]
                for prof in template.disk_profiles
                if f"{prof.name}_scale" in param_mods
            ]
        ),
        "offsets": jnp.array(
            [
                param_mods[f"{prof.name}_offset"]
                for prof in template.disk_profiles
                if f"{prof.name}_offset" in param_mods
            ]
        ),
    }

    if len(disk_params["centers"]) > 0:
        total_disk_flux = _compute_disk_flux_vectorized(wave, **disk_params)

        if jnp.any(jnp.isnan(total_disk_flux)):
            import pprint

            pprint.pprint(disk_params)
            raise ValueError()

    line_params = {
        "centers": jnp.array(
            [
                param_mods[f"{prof.name}_center"]
                for prof in template.line_profiles
                if f"{prof.name}_center" in param_mods
            ]
        ),
        "vel_widths": jnp.array(
            [
                param_mods[f"{prof.name}_vel_width"]
                for prof in template.line_profiles
                if f"{prof.name}_vel_width" in param_mods
            ]
        ),
        "amplitudes": jnp.array(
            [
                param_mods[f"{prof.name}_amplitude"]
                for prof in template.line_profiles
                if f"{prof.name}_amplitude" in param_mods
            ]
        ),
        "shapes": jnp.array(
            [prof.shape == Shape.GAUSSIAN for prof in template.line_profiles]
        ),
    }

    if len(line_params["centers"]) > 0:
        total_line_flux = _compute_line_flux_vectorized(wave, **line_params)

    total_flux = total_disk_flux + total_line_flux

    return total_flux, total_disk_flux, total_line_flux


@flax.struct.dataclass
class ParameterCache:
    """
    Cache for pre-computed parameter metadata to avoid repeated processing.

    Attributes
    ----------
    disk_names : List[str]
        Names of disk profiles in the template.
    line_names : List[str]
        Names of line profiles in the template.
    n_disks : int
        Number of disk profiles.
    n_lines : int
        Number of line profiles.
    param_groups : Dict[str, List[Tuple[str, Parameter]]]
        Grouped parameters for batch sampling, keyed by parameter name.
    fixed_params : List[Tuple[str, Parameter]]
        List of fixed parameters for all profiles.
    shared_params : List[Tuple[str, Parameter]]
        List of shared parameters for all profiles.
    line_shapes : jnp.ndarray
        Boolean array indicating the shape of each line profile.
    """

    disk_names: List[str]
    line_names: List[str]
    n_disks: int
    n_lines: int
    param_groups: Dict[str, List[Tuple[str, Parameter]]]
    fixed_params: List[Tuple[str, Parameter]]
    shared_params: List[Tuple[str, Parameter]]
    line_shapes: jnp.ndarray

    @classmethod
    def create(cls, template: Template):
        """
        Create a `ParameterCache` from a `Template`. Extract and pre-compute
        necessary parameter metadata to optimize model evaluation and sampling.

        Parameters
        ----------
        template : Template
            The model template containing disk and line profile definitions.

        Returns
        -------
        ParameterCache
            An instance of ParameterCache containing pre-computed parameter metadata.
        """
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
        return cls(
            disk_names=disk_names,
            line_names=line_names,
            n_disks=n_disks,
            n_lines=n_lines,
            param_groups=param_groups,
            fixed_params=fixed_params,
            shared_params=shared_params,
            line_shapes=line_shapes,
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
    """Optimized parameter batch sampling with reduced branching."""
    param_mods = {}

    if not param_batch:
        return param_mods

    # Pre-sort parameters by type to minimize branching
    param_types = {
        "uniform": [],
        "log_uniform": [],
        "normal": [],
        "log_normal": [],
        "circular": [],
    }

    for prof_name, param in param_batch:
        if param.circular:
            param_types["circular"].append((prof_name, param))
        elif param.distribution == Distribution.UNIFORM:
            param_types["uniform"].append((prof_name, param))
        elif param.distribution == Distribution.LOG_UNIFORM:
            param_types["log_uniform"].append((prof_name, param))
        elif param.distribution == Distribution.NORMAL:
            param_types["normal"].append((prof_name, param))
        elif param.distribution == Distribution.LOG_NORMAL:
            param_types["log_normal"].append((prof_name, param))

    # Process each type with vectorized operations
    for dist_type, params in param_types.items():
        if not params:
            continue

        n_params = len(params)

        if dist_type == "uniform":
            # Pre-compute bounds arrays
            lows = jnp.array([param.low for _, param in params])
            highs = jnp.array([param.high for _, param in params])
            scales = highs - lows

            uniform_bases = numpyro.sample(
                f"{base_name}_uniform_base", dist.Uniform(0, 1).expand([n_params])
            )

            values = lows + uniform_bases * scales

            for i, (prof_name, param) in enumerate(params):
                samp_name = f"{prof_name}_{param.name}"
                param_mods[samp_name] = numpyro.deterministic(samp_name, values[i])

        elif dist_type == "log_uniform":
            log_lows = jnp.array([jnp.log10(param.low) for _, param in params])
            log_highs = jnp.array([jnp.log10(param.high) for _, param in params])
            log_scales = log_highs - log_lows

            log_uniform_bases = numpyro.sample(
                f"{base_name}_log_uniform_base", dist.Uniform(0, 1).expand([n_params])
            )

            log_values = log_lows + log_uniform_bases * log_scales
            values = 10**log_values

            for i, (prof_name, param) in enumerate(params):
                samp_name = f"{prof_name}_{param.name}"
                param_mods[samp_name] = numpyro.deterministic(samp_name, values[i])

        elif dist_type == "normal":
            normal_bases = numpyro.sample(
                f"{base_name}_normal_base", dist.Uniform(0, 1).expand([n_params])
            )

            for i, (prof_name, param) in enumerate(params):
                samp_name = f"{prof_name}_{param.name}"
                value = truncnorm_ppf(
                    normal_bases[i],
                    loc=param.loc,
                    scale=param.scale,
                    lower_limit=param.low,
                    upper_limit=param.high,
                )
                param_mods[samp_name] = numpyro.deterministic(samp_name, value)

        elif dist_type == "log_normal":
            log_normal_bases = numpyro.sample(
                f"{base_name}_log_normal_base", dist.Uniform(0, 1).expand([n_params])
            )

            for i, (prof_name, param) in enumerate(params):
                samp_name = f"{prof_name}_{param.name}"
                log_value = truncnorm_ppf(
                    log_normal_bases[i],
                    loc=jnp.log10(param.loc),
                    scale=jnp.log10(param.scale),
                    lower_limit=jnp.log10(param.low),
                    upper_limit=jnp.log10(param.high),
                )
                param_mods[samp_name] = numpyro.deterministic(samp_name, 10**log_value)

        elif dist_type == "circular":
            for i, (prof_name, param) in enumerate(params):
                samp_name = f"{prof_name}_{param.name}"

                circular_x = numpyro.sample(
                    f"{samp_name}_circ_x_base", dist.Normal(0, 1).expand([n_params])
                )
                circular_y = numpyro.sample(
                    f"{samp_name}_circ_y_base", dist.Normal(0, 1).expand([n_params])
                )

                x = circular_x[i]
                y = circular_y[i]

                if param.distribution == Distribution.NORMAL:
                    x = x + jnp.cos(param.loc)
                    y = y + jnp.sin(param.loc)

                r = jnp.sqrt(x**2 + y**2) + 1e-6
                value = jnp.arctan2(y / r, x / r) % (2 * jnp.pi)

                param_mods[samp_name] = numpyro.deterministic(samp_name, value)

    return param_mods


def disk_model_optimized(
    template: Template,
    wave: jnp.ndarray,
    flux: jnp.ndarray | None = None,
    flux_err: jnp.ndarray | None = None,
    cache: ParameterCache | None = None,
):
    """
    Optimized disk model with caching and vectorized operations.

    Parameters
    ----------
    template : Template
        The model template containing disk and line profile definitions.
    wave : jnp.ndarray
        Wavelength array (in Angstroms) at which to evaluate the model.
    flux : jnp.ndarray or None, optional
        Observed flux values at each wavelength, by default None.
    flux_err : jnp.ndarray or None, optional
        Observational uncertainties for each flux value, by default None.
    cache : ParameterCache or None, optional
        Precomputed parameter cache for efficiency, by default None.
    """

    # Use cache or create new one
    if cache is None:
        cache = ParameterCache.create(template)

    param_mods = {}

    # Sample parameters in optimized batches
    for param_name, param_batch in cache.param_groups.items():
        batch_params = _sample_parameter_batch_optimized(param_batch, param_name)
        param_mods.update(batch_params)

    # Sample white noise
    white_noise = numpyro.sample(
        "white_noise", dist.Uniform(template.white_noise.low, template.white_noise.high)
    )

    # Add fixed parameters (pre-computed)
    for prof_name, param in cache.fixed_params:
        param_mods[f"{prof_name}_{param.name}"] = numpyro.deterministic(
            f"{prof_name}_{param.name}", param.value
        )

    # Add shared parameters (pre-computed)
    for prof_name, param in cache.shared_params:
        samp_name = f"{prof_name}_{param.name}"
        param_mods[samp_name] = numpyro.deterministic(
            samp_name, param_mods[f"{param.shared}_{param.name}"]
        )

    # Compute outer radius for disk profiles
    for prof_name in cache.disk_names:
        param_name = f"{prof_name}_outer_radius"
        param_mods[param_name] = numpyro.deterministic(
            param_name,
            param_mods[f"{prof_name}_inner_radius"]
            + param_mods[f"{prof_name}_delta_radius"],
        )

    # Pre-allocate and fill parameter arrays efficiently
    if cache.n_disks > 0:
        disk_centers = jnp.stack(
            [param_mods[f"{name}_center"] for name in cache.disk_names]
        )
        disk_inner_radii = jnp.stack(
            [param_mods[f"{name}_inner_radius"] for name in cache.disk_names]
        )
        disk_outer_radii = jnp.stack(
            [param_mods[f"{name}_outer_radius"] for name in cache.disk_names]
        )
        disk_sigmas = jnp.stack(
            [param_mods[f"{name}_sigma"] for name in cache.disk_names]
        )
        disk_inclinations = jnp.stack(
            [param_mods[f"{name}_inclination"] for name in cache.disk_names]
        )
        disk_qs = jnp.stack([param_mods[f"{name}_q"] for name in cache.disk_names])
        disk_eccentricities = jnp.stack(
            [param_mods[f"{name}_eccentricity"] for name in cache.disk_names]
        )
        disk_apocenters = jnp.stack(
            [param_mods[f"{name}_apocenter"] for name in cache.disk_names]
        )
        disk_scales = jnp.stack(
            [param_mods[f"{name}_scale"] for name in cache.disk_names]
        )
        disk_offsets = jnp.stack(
            [param_mods[f"{name}_offset"] for name in cache.disk_names]
        )
    else:
        # Use empty arrays with proper shape
        disk_centers = jnp.array([])
        disk_inner_radii = jnp.array([])
        disk_outer_radii = jnp.array([])
        disk_sigmas = jnp.array([])
        disk_inclinations = jnp.array([])
        disk_qs = jnp.array([])
        disk_eccentricities = jnp.array([])
        disk_apocenters = jnp.array([])
        disk_scales = jnp.array([])
        disk_offsets = jnp.array([])

    if cache.n_lines > 0:
        line_centers = jnp.stack(
            [param_mods[f"{name}_center"] for name in cache.line_names]
        )
        line_vel_widths = jnp.stack(
            [param_mods[f"{name}_vel_width"] for name in cache.line_names]
        )
        line_amplitudes = jnp.stack(
            [param_mods[f"{name}_amplitude"] for name in cache.line_names]
        )
    else:
        line_centers = jnp.array([])
        line_vel_widths = jnp.array([])
        line_amplitudes = jnp.array([])

    # Compute fluxes with optimized functions
    total_disk_flux = _compute_disk_flux_vectorized(
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

    total_line_flux = _compute_line_flux_vectorized(
        wave, line_centers, line_vel_widths, line_amplitudes, cache.line_shapes
    )

    total_flux = total_disk_flux + total_line_flux

    # Construct total error (optimized)
    flux_err = flux_err if flux_err is not None else jnp.zeros_like(wave)
    total_error = jnp.sqrt(flux_err**2 + total_flux**2 * jnp.exp(2 * white_noise))

    with numpyro.plate("data", wave.shape[0]):
        numpyro.deterministic("disk_flux", total_disk_flux)
        numpyro.deterministic("line_flux", total_line_flux)
        numpyro.sample("total_flux", dist.Normal(total_flux, total_error), obs=flux)


# Usage example:
def create_optimized_model(template: Template):
    """Factory function to create optimized model with cached metadata."""
    cache = ParameterCache.create(template)

    def model(wave, flux=None, flux_err=None):
        return disk_model_optimized(template, wave, flux, flux_err, cache)

    return model
