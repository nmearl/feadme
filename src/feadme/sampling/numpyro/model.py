import astropy.constants as const
import astropy.units as u
import flax.struct
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.infer.reparam import CircularReparam

from ..base_model import BaseModel
from ...core.evaluators import (
    evaluate_model,
)
from ...core.parser import Distribution, Parameter

ERR = float(np.finfo(np.float32).tiny)
EPS = 1e-6
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value


def ratio_factor(name, F_num, F_den, ratio, sigma_ln=0.02, eps=1e-30):
    """
    Softly constrain F_num / F_den ~= ratio via a Normal penalty in log-space.
    sigma_ln is the 1-sigma width in ln(ratio).
      - sigma_ln=0.02 ~ 2% (tight)
      - sigma_ln=0.10 ~ 10% (loose)
    """
    log_r = jnp.log((F_num + eps) / (F_den + eps))
    numpyro.factor(name, dist.Normal(jnp.log(ratio), sigma_ln).log_prob(log_r))


@flax.struct.dataclass
class NumpyroModel(BaseModel):
    def __call__(self, wave, flux_err, flux=None):
        # Dictionary to store all sampled parameters
        param_mods = {}

        for prof in self.config.template.disk_profiles:
            # Explicitly handle inner and outer radii
            rin_name = prof.inner_radius.qualified_name
            rin_samp = self.sample_param(
                rin_name,
                prof.inner_radius,
                prof.inner_radius.low,
                prof.inner_radius.high,
            )
            param_mods[rin_name] = rin_samp

            rr_name = prof.radius_ratio.qualified_name
            rr_samp = self.sample_param(
                rr_name, prof.radius_ratio, prof.radius_ratio.low, 2e4 / rin_samp
            )
            param_mods[rr_name] = rr_samp

            # Sample independent parameters
            for param in prof.independent:
                if "radius" in param.name:
                    continue

                samp_name = param.qualified_name
                param_samp = self.sample_param(samp_name, param, param.low, param.high)
                param_mods[samp_name] = param_samp

            # Add fixed parameters
            for param in prof.fixed:
                samp_name = param.qualified_name
                param_samp = numpyro.deterministic(samp_name, param.value)
                param_mods[samp_name] = param_samp

        for prof in self.config.template.line_profiles:
            # Sample independent parameters
            for param in prof.independent:
                samp_name = param.qualified_name
                param_samp = self.sample_param(samp_name, param, param.low, param.high)
                param_mods[samp_name] = param_samp

            # Add fixed parameters
            for param in prof.fixed:
                samp_name = param.qualified_name
                param_samp = numpyro.deterministic(samp_name, param.value)
                param_mods[samp_name] = param_samp

        # Add shared parameters
        for prof in self.config.template.disk_profiles:
            for param in prof.shared:
                samp_name = param.qualified_name
                param_samp = numpyro.deterministic(
                    samp_name, param_mods[f"{param.shared}_{param.name}"]
                )
                param_mods[samp_name] = param_samp

        for prof in self.config.template.line_profiles:
            for param in prof.shared:
                samp_name = param.qualified_name
                param_samp = numpyro.deterministic(
                    samp_name, param_mods[f"{param.shared}_{param.name}"]
                )
                param_mods[samp_name] = param_samp

        # Compose outer radius from inner radius and radius ratio
        for prof in self.config.template.disk_profiles:
            inner_radius = param_mods[prof.inner_radius.qualified_name]
            radius_ratio = param_mods.pop(prof.radius_ratio.qualified_name)
            param_samp = numpyro.deterministic(
                f"{prof.name}_outer_radius", inner_radius * radius_ratio
            )
            param_mods[f"{prof.name}_outer_radius"] = param_samp

        niil_narrow_area = param_mods.get("niil_narrow_area", None)
        niir_narrow_area = param_mods.get("niir_narrow_area", None)
        ratio_factor(
            "nii_ratio_6583_6548",
            niir_narrow_area,
            niil_narrow_area,
            ratio=2.95,
            sigma_ln=0.02,
        )

        # Sample white noise with better bounds
        if self.config.template.white_noise.fixed:
            white_noise = numpyro.deterministic(
                "white_noise", self.config.template.white_noise.value
            )
        else:
            white_noise = self.sample_param(
                "white_noise",
                self.config.template.white_noise,
                self.config.template.white_noise.low,
                self.config.template.white_noise.high,
            )

        # Sample redshift
        if self.config.template.redshift.fixed:
            redshift = numpyro.deterministic(
                "redshift", self.config.template.redshift.value
            )
        else:
            redshift = self.sample_param(
                "redshift",
                self.config.template.redshift,
                self.config.template.redshift.low,
                self.config.template.redshift.high,
            )

        # rest_wave = wave / (1 + redshift)

        total_flux, total_disk_flux, total_line_flux = evaluate_model(
            template=self.config.template,
            wave=wave,
            param_mods=param_mods,
            redshift=redshift,
            integrator=self.integrator,
        )

        total_error = jnp.sqrt(flux_err**2 + jnp.exp(2 * white_noise))

        numpyro.deterministic("disk_flux", total_disk_flux)
        numpyro.deterministic("line_flux", total_line_flux)

        with numpyro.plate("data", wave.shape[0]):
            numpyro.sample(
                "total_flux",
                dist.Normal(total_flux, total_error),
                obs=flux,
            )

    @staticmethod
    def sample_param(
        samp_name: str,
        param: Parameter,
        lower_bound: float,
        upper_bound: float,
    ) -> ArrayLike:
        if param.circular:
            x = numpyro.sample(f"{samp_name}_x_base", dist.Normal(0.0, 1.0))
            y = numpyro.sample(f"{samp_name}_y_base", dist.Normal(0.0, 1.0))
            return numpyro.deterministic(
                samp_name, jnp.mod(jnp.arctan2(y, x), 2 * jnp.pi)
            )

        if param.name == "inclination":
            mu_min = jnp.cos(upper_bound)  # cos(i_max)
            mu_max = jnp.cos(lower_bound)  # cos(i_min)
            # mu = _logit_uniform(f"{samp_name}_base", mu_min, mu_max)
            mu = numpyro.sample(f"{samp_name}_base", dist.Uniform(mu_min, mu_max))
            incl = jnp.arccos(mu)

            return numpyro.deterministic(samp_name, incl)

        if param.distribution == Distribution.UNIFORM:
            # param_samp = _logit_uniform(samp_name, lower_bound, upper_bound)
            param_samp = numpyro.sample(
                samp_name, dist.Uniform(lower_bound, upper_bound)
            )

        elif param.distribution == Distribution.LOG_UNIFORM:
            # param_samp = _logit_loguniform(samp_name, lower_bound, upper_bound)
            param_samp = numpyro.sample(
                samp_name,
                dist.LogUniform(lower_bound, upper_bound),
            )

        elif param.distribution == Distribution.NORMAL:
            param_samp = numpyro.sample(
                samp_name,
                dist.TruncatedNormal(
                    param.loc, param.scale, low=lower_bound, high=upper_bound
                ),
            )

        elif param.distribution == Distribution.LOG_NORMAL:
            sigma_log = jnp.sqrt(jnp.log(1.0 + (param.scale / param.loc) ** 2))
            mu_log = jnp.log(param.loc) - 0.5 * sigma_log**2

            base = numpyro.sample(
                f"{samp_name}_base",
                dist.TruncatedNormal(
                    loc=mu_log,
                    scale=sigma_log,
                    low=jnp.log(lower_bound),
                    high=jnp.log(upper_bound),
                ),
            )
            param_samp = numpyro.deterministic(samp_name, jnp.exp(base))

        else:
            raise ValueError(f"Unsupported distribution: {param.distribution}")

        return param_samp
