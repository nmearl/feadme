import astropy.constants as const
import astropy.units as u
import flax.struct
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions.transforms import ExpTransform
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

        # Sample independent parameters
        for samp_name, param in self.config.template.independent_parameters.items():
            param_samp = self.sample_param(samp_name, param, param.low, param.high)
            param_mods[samp_name] = param_samp

        # Add fixed parameters
        for samp_name, param in self.config.template.fixed_parameters.items():
            param_samp = numpyro.deterministic(samp_name, param.value)
            param_mods[samp_name] = param_samp

        # Add shared parameters
        for (
            samp_name,
            targ_name,
        ) in self.config.template.map_shared_parameters.items():
            param_samp = numpyro.deterministic(samp_name, param_mods[targ_name])
            param_mods[samp_name] = param_samp

        # Compose outer radius from inner radius and radius ratio
        for prof in self.config.template.disk_profiles:
            inner_radius = param_mods[f"{prof.name}_inner_radius"]
            radius_ratio = param_mods[f"{prof.name}_radius_ratio"]
            param_samp = numpyro.deterministic(
                f"{prof.name}_outer_radius", inner_radius * radius_ratio
            )
            param_mods[f"{prof.name}_outer_radius"] = param_samp

        # Soft constraint: discourage rout > 2e4 without chopping the prior
        for prof in self.config.template.disk_profiles:
            cap = 2e4
            k = 50.0  # penalty strength
            excess = jnp.maximum(
                param_mods[f"{prof.name}_outer_radius"] / cap - 1.0, 0.0
            )
            numpyro.factor(f"{prof.name}_outer_radius_factor", -k * excess**2)

        # Soft constraint: [NII] 6583/6548 ratio should be ~2.95
        niil_narrow_area = param_mods["niil_narrow_area"]
        niir_narrow_area = param_mods["niir_narrow_area"]
        ratio_factor(
            "nii_ratio_6583_6548",
            niir_narrow_area,
            niil_narrow_area,
            ratio=2.95,
            sigma_ln=0.02,
        )

        total_flux, total_disk_flux, total_line_flux = evaluate_model(
            template=self.config.template,
            wave=wave,
            param_mods=param_mods,
            redshift=param_mods["redshift"],
            integrator=self.integrator,
        )

        total_error = jnp.sqrt(flux_err**2 + jnp.exp(2 * param_mods["white_noise"]))

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

        if "inclination" in samp_name:
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
            sigma_log = jnp.log1p(param.scale / param.loc)
            mu_log = jnp.log(param.loc)

            param_samp = numpyro.sample(
                samp_name,
                dist.TransformedDistribution(
                    dist.TruncatedNormal(
                        loc=mu_log,
                        scale=sigma_log,
                        low=jnp.log(lower_bound),
                        high=jnp.log(upper_bound),
                    ),
                    ExpTransform(),
                ),
            )

        else:
            raise ValueError(f"Unsupported distribution: {param.distribution}")

        return param_samp
