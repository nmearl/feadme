import astropy.constants as const
import astropy.units as u
import numpyro

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


def disk_model(
    template: Template,
    wave: jnp.ndarray,
    flux: jnp.ndarray | None = None,
    flux_err: jnp.ndarray | None = None,
    masks: dict[str, jnp.ndarray] | None = None,
):
    tot_disk_flux = jnp.zeros_like(wave)
    tot_line_flux = jnp.zeros_like(wave)

    flux_err = flux_err if flux_err is not None else jnp.zeros_like(wave)

    param_mods = {}

    # Sample all independent parameters with non-centered parameterization
    for prof in template.disk_profiles + template.line_profiles:
        for param in prof.independent:
            samp_name = f"{prof.name}_{param.name}"

            base_value = numpyro.sample(samp_name + "_base", dist.Uniform(0, 1))
            param_mods[samp_name] = numpyro.deterministic(
                samp_name,
                param.transform(base_value),
            )

            # if param.distribution == Distribution.uniform:
            #     base_value = numpyro.sample(samp_name + "_base", dist.Normal(0, 1))
            #     param_mods[samp_name] = numpyro.deterministic(
            #         samp_name,
            #         param.low
            #         + (param.high - param.low) * jax.scipy.stats.norm.cdf(base_value),
            #     )
            #
            # elif param.distribution == Distribution.log_uniform:
            #     base_value = numpyro.sample(samp_name + "_base", dist.Normal(0, 1))
            #     log_low, log_high = jnp.log10(param.low), jnp.log10(param.high)
            #     param_mods[samp_name] = numpyro.deterministic(
            #         samp_name,
            #         10
            #         ** (
            #             log_low
            #             + (log_high - log_low) * jax.scipy.stats.norm.cdf(base_value)
            #         ),
            #     )
            #
            # elif param.distribution == Distribution.normal:
            #     base_value = numpyro.sample(samp_name + "_base", dist.Normal(0, 1))
            #     param_mods[samp_name] = numpyro.deterministic(
            #         samp_name, param.loc + param.scale * base_value
            #     )
            #
            # elif param.distribution == Distribution.log_normal:
            #     base_value = numpyro.sample(samp_name + "_base", dist.Normal(0, 1))
            #     param_mods[samp_name] = numpyro.deterministic(
            #         samp_name, jnp.exp(param.loc + param.scale * base_value)
            #     )

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

    # Reparameterize inner/outer radius relationship
    for prof in template.disk_profiles:
        param_mods[f"{prof.name}_outer_radius"] = numpyro.deterministic(
            f"{prof.name}_outer_radius",
            param_mods[f"{prof.name}_inner_radius"]
            + param_mods[f"{prof.name}_delta_radius"],
        )

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

        tot_disk_flux += (
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

        tot_line_flux += line_flux

    total_flux = tot_disk_flux + tot_line_flux

    # Sample white noise
    white_noise_base = numpyro.sample(
        "white_noise_base", numpyro.distributions.Uniform(0, 1)
    )

    white_noise = numpyro.deterministic(
        "white_noise",
        template.white_noise.transform(white_noise_base),
    )

    # Construct total error
    # total_error = jnp.sqrt(flux_err**2 + white_noise**2)
    total_error = jnp.sqrt(flux_err**2 + total_flux**2 * jnp.exp(2 * white_noise))

    numpyro.deterministic("disk_flux", tot_disk_flux)
    numpyro.deterministic("line_flux", tot_line_flux)

    with numpyro.plate("data", wave.shape[0]):
        # numpyro.deterministic("total_flux", total_flux)
        numpyro.sample("obs", dist.Normal(total_flux, total_error), obs=flux)
