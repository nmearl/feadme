import astropy.constants as const
import astropy.units as u
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions.transforms import AffineTransform

from ..base_model import BaseModel
from ...core.distributions import BoundedLogNormal
from ...core.evaluators import (
    evaluate_model,
)
from ...core.parser import Distribution, Parameter

ERR = float(np.finfo(np.float32).tiny)
EPS = 1e-6
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value


def _find_ecc_apo_pairs(iter_independent):
    """Return {profile_name: (ecc_ref, apo_ref)} for profiles with both parameters."""
    ecc_by_profile = {}
    apo_by_profile = {}
    for param_ref in iter_independent:
        if param_ref.field_name == "eccentricity":
            ecc_by_profile[param_ref.profile_name] = param_ref
        elif param_ref.field_name == "apocenter":
            apo_by_profile[param_ref.profile_name] = param_ref
    return {
        pname: (ecc_by_profile[pname], apo_by_profile[pname])
        for pname in ecc_by_profile
        if pname in apo_by_profile
    }


def _sample_ecc_apo_hk(ecc_ref, apo_ref):
    """
    Non-Jacobian (h, k) reparameterization for eccentricity + apocenter.

    h, k ~ Normal(0, σ) with no factor correction.  The prior on (e, φ₀) is
    induced implicitly: e follows a Rayleigh-derived distribution bounded by
    [e_low, e_high] via tanh; φ₀ is uniform on [0, 2π].  σ is chosen so the
    Rayleigh mode lands at the template's eccentricity loc.
    """
    e_low = ecc_ref.param.low
    e_high = ecc_ref.param.high
    e_span = e_high - e_low
    e_loc = ecc_ref.param.loc if ecc_ref.param.loc is not None else e_low + 0.5 * e_span
    sigma_hk = float(np.arctanh(np.clip((e_loc - e_low) / e_span, 0.01, 0.99)))

    h = numpyro.sample(f"{apo_ref.name}_h", dist.Normal(0.0, sigma_hk))
    k = numpyro.sample(f"{apo_ref.name}_k", dist.Normal(0.0, sigma_hk))

    r = jnp.sqrt(h**2 + k**2)
    e = numpyro.deterministic(ecc_ref.name, e_low + e_span * jnp.tanh(r))
    phi0 = numpyro.deterministic(apo_ref.name, jnp.mod(jnp.arctan2(k, h), 2 * jnp.pi))
    return e, phi0


def ratio_factor(name, F_num, F_den, ratio, sigma_ln=0.05, eps=1e-30):
    """
    Softly constrain F_num / F_den ~= ratio via a Normal penalty in log-space.
    sigma_ln is the 1-sigma width in ln(ratio).
      - sigma_ln=0.05 ~ 5% (moderately tight)
      - sigma_ln=0.10 ~ 10% (loose)
    """
    log_r = jnp.log((F_num + eps) / (F_den + eps))
    numpyro.factor(name, dist.Normal(jnp.log(ratio), sigma_ln).log_prob(log_r))


def _distribution_from_param(
    param: Parameter, lower_bound: float, upper_bound: float
) -> dist.Distribution:
    if param.distribution == Distribution.UNIFORM:
        return dist.Uniform(lower_bound, upper_bound)

    if param.distribution == Distribution.LOG_UNIFORM:
        return dist.LogUniform(lower_bound, upper_bound)

    if param.distribution == Distribution.NORMAL:
        return dist.TruncatedNormal(
            param.loc, param.scale, low=lower_bound, high=upper_bound
        )

    if param.distribution == Distribution.LOG_NORMAL:
        sigma_log = jnp.log1p(param.scale / param.loc)
        mu_log = jnp.log(param.loc)
        return BoundedLogNormal(mu_log, sigma_log, lower_bound, upper_bound)

    if param.distribution == Distribution.BETA:
        if param.alpha is None or param.beta is None:
            raise ValueError("Beta parameters require alpha and beta values.")
        return dist.Beta(param.alpha, param.beta)

    raise ValueError(f"Unsupported distribution: {param.distribution}")


def _inclination_distribution_from_param(
    param: Parameter, lower_bound: float, upper_bound: float
) -> dist.Distribution:
    mu_min = jnp.cos(upper_bound)
    mu_max = jnp.cos(lower_bound)

    if param.distribution == Distribution.UNIFORM:
        return dist.Uniform(mu_min, mu_max)

    if param.distribution == Distribution.NORMAL:
        # Interpret the template's loc/scale in angle space, but sample in
        # mu = cos(i). Map the angular width into mu-space with a local
        # linearization about the prior centre.
        mu_loc = jnp.cos(param.loc)
        mu_scale = jnp.maximum(jnp.abs(jnp.sin(param.loc)) * param.scale, 1e-3)
        return dist.TruncatedNormal(
            mu_loc,
            mu_scale,
            low=mu_min,
            high=mu_max,
        )

    if param.distribution == Distribution.BETA:
        if param.alpha is None or param.beta is None:
            raise ValueError("Beta inclination priors require alpha and beta values.")
        return dist.TransformedDistribution(
            dist.Beta(param.alpha, param.beta),
            AffineTransform(mu_min, mu_max - mu_min),
        )

    raise ValueError(
        "Inclination only supports uniform, normal, or beta priors in the template, "
        f"got {param.distribution}."
    )


@flax.struct.dataclass
class NumpyroModel(BaseModel):
    def __call__(self, wave, flux_err, flux=None):
        # Dictionary to store all sampled parameters
        param_mods = {}

        # Sample eccentricity/apocenter pairs jointly via the (h, k) reparameterization
        # ecc_apo_pairs = _find_ecc_apo_pairs(self.config.template.iter_independent)
        # jointly_handled = set()
        # for profile_name, (ecc_ref, apo_ref) in ecc_apo_pairs.items():
        #     e, phi0 = _sample_ecc_apo_hk(ecc_ref, apo_ref)
        #     param_mods[ecc_ref.name] = e
        #     param_mods[apo_ref.name] = phi0
        #     jointly_handled.add(ecc_ref.name)
        #     jointly_handled.add(apo_ref.name)

        for param_ref in self.config.template.iter_independent:
            # if param_ref.name in jointly_handled:
            #     continue
            param_samp = self.sample_param(
                param_ref.name,
                param_ref.param,
                param_ref.param.low,
                param_ref.param.high,
            )
            param_mods[param_ref.name] = param_samp

        # Add fixed parameters
        for param_ref in self.config.template.iter_fixed:
            param_samp = numpyro.deterministic(param_ref.name, param_ref.param.value)
            param_mods[param_ref.name] = param_samp

        # Add shared parameters
        for param_ref in self.config.template.iter_shared:
            param_samp = numpyro.deterministic(
                param_ref.name, param_mods[param_ref.target_name]
            )
            param_mods[param_ref.name] = param_samp

        for profile in self.config.template.disk_profiles:
            inner_name = f"{profile.name}_inner_radius"
            ratio_name = f"{profile.name}_radius_ratio"
            outer_name = f"{profile.name}_outer_radius"
            inner_radius = param_mods.get(inner_name)
            radius_ratio = param_mods.get(ratio_name)
            if inner_radius is not None and radius_ratio is not None:
                param_mods[outer_name] = numpyro.deterministic(
                    outer_name, inner_radius * radius_ratio
                )

        # Soft constraint: [NII] 6583/6548 ratio should be close to the
        # atomic-physics expectation, while allowing modest decomposition error.
        niil_narrow_area = param_mods.get("niil_narrow_area")
        niir_narrow_area = param_mods.get("niir_narrow_area")

        if niil_narrow_area is not None and niir_narrow_area is not None:
            ratio_factor(
                "nii_ratio_6583_6548",
                niir_narrow_area,
                niil_narrow_area,
                ratio=3.05,
                sigma_ln=0.05,
            )

        total_flux, total_disk_flux, total_line_flux = evaluate_model(
            template=self.config.template,
            wave=wave,
            param_mods=param_mods,
            redshift=param_mods["redshift"],
            integrator=self.integrator,
        )

        # White-noise inflation should never reduce the measured data uncertainty.
        total_error = jnp.sqrt(
            jnp.square(flux_err)
            + jnp.square(total_flux) * jnp.exp(2.0 * param_mods["log_frac_noise"])
        )

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
            mu = numpyro.sample(
                f"{samp_name}_base",
                _inclination_distribution_from_param(param, lower_bound, upper_bound),
            )
            return numpyro.deterministic(samp_name, jnp.arccos(mu))

        return numpyro.sample(
            samp_name, _distribution_from_param(param, lower_bound, upper_bound)
        )
