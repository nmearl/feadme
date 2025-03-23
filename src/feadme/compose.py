import astropy.constants as const
import astropy.units as u
import numpyro
from numpyro.infer.reparam import TransformReparam, LocScaleReparam
from .utils import TruncatedAffineTransform, truncnorm_ppf
import jax.scipy.stats as stats

numpyro.set_host_device_count(2)
numpyro.enable_x64()

import numpyro.distributions as dist
import jax
import jax.numpy as jnp
import numpy as np

from .models.disk import (
    jax_integrate,
    _jax_integrate,
    jax_integrate_single,
)
from .parser import Template, Distribution

numpyro.set_platform("cpu")

print(jax.local_device_count())

ERR = float(np.finfo(np.float32).tiny)
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value


def evaluate_disk_model(template, wave, param_mods):
    total_disk_flux = jnp.zeros_like(wave)
    total_line_flux = jnp.zeros_like(wave)

    # Compute disk flux
    for prof in template.disk_profiles:
        nu = c_cgs / (wave * 1e-8)
        nu0 = c_cgs / (param_mods[f"{prof.name}_center"] * 1e-8)
        X = nu / nu0 - 1

        local_sigma = param_mods[f"{prof.name}_sigma"] * 1e5

        xi1 = param_mods[f"{prof.name}_inner_radius"]
        xi2 = param_mods[f"{prof.name}_outer_radius"]
        phi1 = -jnp.pi * 0.5
        phi2 = jnp.pi * 0.5

        res = jax_integrate(
            xi1,
            xi2,
            phi1,
            phi2,
            jnp.asarray(X),
            param_mods[f"{prof.name}_inclination"],
            local_sigma,
            param_mods[f"{prof.name}_q"],
            param_mods[f"{prof.name}_eccentricity"],
            param_mods[f"{prof.name}_apocenter"],
        )

        total_disk_flux += (
            res / jnp.max(res) * param_mods[f"{prof.name}_scale"]
            + param_mods[f"{prof.name}_offset"]
        )

    # Compute line flux
    for prof in template.line_profiles:
        fwhm = (
            param_mods[f"{prof.name}_vel_width"]
            / c_kms
            * param_mods[f"{prof.name}_center"]
        )

        if prof.shape.name == "lorentzian":
            line_flux = param_mods[f"{prof.name}_amplitude"] * (
                (fwhm * 0.5)
                / ((wave - param_mods[f"{prof.name}_center"]) ** 2 + (fwhm * 0.5) ** 2)
            )
        elif prof.shape.name == "gaussian":
            stddev = fwhm / 2.355
            line_flux = param_mods[f"{prof.name}_amplitude"] * jnp.exp(
                -0.5 * ((wave - param_mods[f"{prof.name}_center"]) / stddev) ** 2
            )
        else:
            raise ValueError(f"Invalid line profile shape: {prof.shape}")

        total_line_flux += line_flux

    total_flux = total_disk_flux + total_line_flux

    return total_flux, total_disk_flux, total_line_flux


def disk_model(
    template: Template,
    wave: jnp.ndarray,
    flux: jnp.ndarray | None = None,
    flux_err: jnp.ndarray | None = None,
    masks: dict[str, jnp.ndarray] | None = None,
):
    param_mods = {}

    # Pre-compute all profiles to iterate over
    all_profiles = template.disk_profiles + template.line_profiles

    # Build re-parameterization configuration more efficiently
    # reparam_config = {}
    #
    # for prof in all_profiles:
    #     for param in prof.independent:
    #         samp_name = f"{prof.name}_{param.name}"
    #
    #         if param.distribution in [
    #             Distribution.uniform,
    #             Distribution.log_uniform,
    #         ]:
    #             reparam_config[samp_name] = TransformReparam()
    #         elif param.distribution in [Distribution.normal, Distribution.log_normal]:
    #             reparam_config[samp_name] = LocScaleReparam(centered=0)

    # Sample parameters with optimized reparam
    # with numpyro.handlers.reparam(config=reparam_config):
    for prof in all_profiles:
        # Sample all shared parameters
        for param in prof.shared:
            samp_name = f"{prof.name}_{param.name}"
            param_mods[samp_name] = numpyro.deterministic(
                samp_name, param_mods[f"{param.shared}_{param.name}"]
            )

        # Include fixed fields
        for param in prof.fixed:
            param_mods[f"{prof.name}_{param.name}"] = numpyro.deterministic(
                f"{prof.name}_{param.name}", param.value
            )

        for param in prof.independent:
            samp_name = f"{prof.name}_{param.name}"

            if param.distribution == Distribution.uniform:
                base_value = numpyro.sample(samp_name + "_base", dist.Uniform(0, 1))
                param_mods[samp_name] = numpyro.deterministic(
                    samp_name,
                    param.low + base_value * (param.high - param.low),
                )
            elif param.distribution == Distribution.log_uniform:
                base_value = numpyro.sample(samp_name + "_base", dist.Uniform(0, 1))
                param_mods[samp_name] = numpyro.deterministic(
                    samp_name,
                    10
                    ** (
                        jnp.log10(param.low)
                        + base_value * (jnp.log10(param.high) - jnp.log10(param.low))
                    ),
                )
            elif param.distribution == Distribution.normal:
                base_value = numpyro.sample(samp_name + "_base", dist.Uniform(0, 1))
                param_mods[samp_name] = numpyro.deterministic(
                    samp_name,
                    truncnorm_ppf(
                        base_value, param.loc, param.scale, param.low, param.high
                    ),
                )

                # if param.distribution == Distribution.uniform:
                #     base_dist = dist.Uniform(0, 1)
                #     transforms = [
                #         dist.transforms.AffineTransform(param.low, param.high)
                #     ]
                #     unif_param_dist = dist.TransformedDistribution(
                #         base_dist, transforms
                #     )
                #     param_mods[samp_name] = numpyro.sample(samp_name, unif_param_dist)
                # elif param.distribution == Distribution.log_uniform:
                #     # Create distribution using properly normalized base distribution
                #     base_dist = dist.Uniform(0, 1)
                #     transforms = [
                #         dist.transforms.AffineTransform(
                #             jnp.log10(param.low), jnp.log10(param.high)
                #         ),
                #         dist.transforms.PowerTransform(10.0),
                #     ]
                #     log_unif_param_dist = dist.TransformedDistribution(
                #         base_dist, transforms
                #     )
                #     param_mods[samp_name] = numpyro.sample(
                #         samp_name, log_unif_param_dist
                #     )
                # elif param.distribution == Distribution.normal:
                #     param_mods[samp_name] = numpyro.sample(
                #         samp_name,
                #         dist.Normal(
                #             param.loc,
                #             param.scale,  # low=param.low, high=param.high
                #         ),
                #     )
                # elif param.distribution == Distribution.log_normal:
                #     # Create distribution using properly normalized base distribution
                #     base_dist = dist.Normal(
                #         jnp.log10(param.loc), jnp.log10(param.scale)
                #     )
                #     transforms = [
                #         dist.transforms.PowerTransform(10.0),
                #     ]
                #     log_norm_param_dist = dist.TransformedDistribution(
                #         base_dist, transforms
                #     )
                #     param_mods[samp_name] = numpyro.sample(
                #         samp_name, log_norm_param_dist
                #     )
                # else:
                #     raise ValueError(
                #         f"Invalid distribution: {param.distribution} for parameter {param.name}"
                #     )

    # Sample white noise
    white_noise = numpyro.sample(
        "white_noise",
        numpyro.distributions.Uniform(
            template.white_noise.low, template.white_noise.high
        ),
    )

    # Reparameterize inner/outer radius relationship
    for prof in template.disk_profiles:
        param_mods[f"{prof.name}_outer_radius"] = numpyro.deterministic(
            f"{prof.name}_outer_radius",
            param_mods[f"{prof.name}_inner_radius"]
            + param_mods[f"{prof.name}_delta_radius"],
        )

    total_flux, total_disk_flux, total_line_flux = evaluate_disk_model(
        template, wave, param_mods
    )

    # Construct total error
    flux_err = flux_err if flux_err is not None else jnp.zeros_like(wave)
    total_error = jnp.sqrt(flux_err**2 + total_flux**2 * jnp.exp(2 * white_noise))

    numpyro.deterministic("disk_flux", total_disk_flux)
    numpyro.deterministic("line_flux", total_line_flux)

    with numpyro.plate("data", wave.shape[0]):
        numpyro.sample("total_flux", dist.Normal(total_flux, total_error), obs=flux)
