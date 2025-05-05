import astropy.constants as const
import astropy.units as u
import numpyro

import jax
import jax.numpy as jnp
import numpy as np
import numpyro.distributions as dist

from .models.disk import jax_integrate, quad_jax_integrate
from .parser import Distribution, Template
from numpyro.infer.reparam import TransformReparam, LocScaleReparam, CircularReparam


ERR = float(np.finfo(np.float32).tiny)
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value


def evaluate_disk_model(
    template: Template, wave: jnp.ndarray, param_mods: dict, use_quad: bool = False
):
    total_disk_flux = jnp.zeros_like(wave)
    total_line_flux = jnp.zeros_like(wave)

    # Compute disk flux
    for prof in template.disk_profiles:
        nu = c_cgs / (wave * 1e-8)
        nu0 = c_cgs / (param_mods[f"{prof.name}_center"] * 1e-8)
        X = nu / nu0 - 1

        local_sigma = param_mods[f"{prof.name}_sigma"] * 1e5 * nu0 / c_cgs

        xi1 = param_mods[f"{prof.name}_inner_radius"]
        xi2 = param_mods[f"{prof.name}_outer_radius"]
        phi1 = 0
        phi2 = 2 * jnp.pi - 1e-6

        integrator = quad_jax_integrate if use_quad else jax_integrate

        res = integrator(
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
            nu0,
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
    use_quad: bool = False,
):
    param_mods = {}

    # Build re-parameterization configuration more efficiently
    reparam_config = {}

    for prof in template.all_profiles:
        for param in prof.independent:
            samp_name = f"{prof.name}_{param.name}"

            if param.circular:
                # reparam_config[f"{samp_name}_x"] = LocScaleReparam()
                # reparam_config[f"{samp_name}_y"] = LocScaleReparam()
                # reparam_config[f"{samp_name}_wrap"] = CircularReparam()
                pass
            else:
                reparam_config[samp_name] = TransformReparam()

    # Pre-compute all profiles to iterate over
    with numpyro.handlers.reparam(config=reparam_config):
        for prof in template.all_profiles:
            for param in prof.independent:
                samp_name = f"{prof.name}_{param.name}"

                if param.distribution == Distribution.uniform:
                    if param.circular:
                        # param_mods[f"{samp_name}_wrap"] = numpyro.sample(
                        #     f"{samp_name}_wrap",
                        #     dist.VonMises(loc=0, concentration=1e-6),
                        # )
                        param_mods[f"{samp_name}_x"] = numpyro.sample(
                            f"{samp_name}_x",
                            dist.Normal(0, 1))
                        param_mods[f"{samp_name}_y"] = numpyro.sample(
                            f"{samp_name}_y",
                            dist.Normal(0, 1))
                    else:
                        base_dist = dist.Uniform(0, 1)
                        transforms = [
                            dist.transforms.AffineTransform(
                                param.low, param.high - param.low
                            )
                        ]
                        unif_param_dist = dist.TransformedDistribution(
                            base_dist, transforms
                        )
                        param_mods[samp_name] = numpyro.sample(
                            samp_name, unif_param_dist
                        )
                elif param.distribution == Distribution.log_uniform:
                    base_dist = dist.Uniform(0, 1)
                    transforms = [
                        dist.transforms.AffineTransform(
                            jnp.log(param.low),
                            jnp.log(param.high) - jnp.log(param.low),
                        ),
                        dist.transforms.ExpTransform(),
                    ]
                    log_unif_param_dist = dist.TransformedDistribution(
                        base_dist, transforms
                    )
                    param_mods[samp_name] = numpyro.sample(
                        samp_name, log_unif_param_dist
                    )
                elif param.distribution == Distribution.normal:
                    if param.circular:
                        # param_mods[f"{samp_name}_wrap"] = numpyro.sample(
                        #     f"{samp_name}_wrap",
                        #     dist.VonMises(
                        #         loc=(param.loc + jnp.pi) % (2 * jnp.pi) - jnp.pi,
                        #         concentration=3 / (param.scale**2),
                        #     ),
                        # )
                        param_mods[f"{samp_name}_x"] = numpyro.sample(
                            f"{samp_name}_x",
                            dist.Normal(0, 1))
                        param_mods[f"{samp_name}_y"] = numpyro.sample(
                            f"{samp_name}_y",
                            dist.Normal(0, 1))
                    else:
                        base_dist = dist.TruncatedNormal(
                            0,
                            1,
                            low=(param.low - param.loc) / param.scale,
                            high=(param.high - param.loc) / param.scale,
                        )
                        transforms = [
                            dist.transforms.AffineTransform(param.loc, param.scale),
                        ]
                        norm_param_dist = dist.TransformedDistribution(
                            base_dist, transforms
                        )
                        param_mods[samp_name] = numpyro.sample(
                            samp_name, norm_param_dist
                        )
                elif param.distribution == Distribution.log_normal:
                    base_dist = dist.TruncatedNormal(
                        0,
                        1,
                        low=(jnp.log(param.low) - jnp.log(param.loc))
                        / jnp.log(param.scale),
                        high=(jnp.log(param.high) - jnp.log(param.loc))
                        / jnp.log(param.scale),
                    )
                    transforms = [
                        dist.transforms.AffineTransform(
                            jnp.log(param.loc), jnp.log(param.scale)
                        ),
                        dist.transforms.ExpTransform(),
                    ]
                    log_norm_param_dist = dist.TransformedDistribution(
                        base_dist, transforms
                    )
                    param_mods[samp_name] = numpyro.sample(
                        samp_name, log_norm_param_dist
                    )
                else:
                    raise ValueError(
                        f"Invalid distribution: {param.distribution} for parameter {param.name}"
                    )

    for prof in template.all_profiles:
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

    # Sample white noise
    white_noise = numpyro.sample(
        "white_noise",
        numpyro.distributions.Uniform(
            template.white_noise.low, template.white_noise.high
        ),
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
                # param_mods[param_name] = numpyro.deterministic(
                #     param_name, param_mods[f"{param_name}_wrap"] % (2 * jnp.pi)
                # )
                param_mods[param_name] = numpyro.deterministic(
                    param_name,
                    jnp.arctan2(param_mods[f"{param_name}_y"], param_mods[f"{param_name}_x"])
                    % (2 * jnp.pi),
                )

    total_flux, total_disk_flux, total_line_flux = evaluate_disk_model(
        template, wave, param_mods, use_quad
    )

    # Construct total error
    flux_err = flux_err if flux_err is not None else jnp.zeros_like(wave)
    total_error = jnp.sqrt(flux_err**2 + total_flux**2 * jnp.exp(2 * white_noise))

    numpyro.deterministic("disk_flux", total_disk_flux)
    numpyro.deterministic("line_flux", total_line_flux)

    with numpyro.plate("data", wave.shape[0]):
        numpyro.sample("total_flux", dist.Normal(total_flux, total_error), obs=flux)
