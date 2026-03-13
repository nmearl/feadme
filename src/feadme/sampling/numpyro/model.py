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


def _find_ecc_apo_pairs(
    iter_independent,
) -> dict[str, tuple]:
    """
    Pre-pass over independent parameters to identify (eccentricity, apocenter)
    pairs that should be sampled jointly via the (h, k) reparameterization.

    Returns a dict keyed by profile_name mapping to (ecc_ref, apo_ref) tuples.
    Only profiles where *both* eccentricity and apocenter are free and independent
    are included. Profiles where either is fixed or shared are excluded, and those
    parameters fall through to standard sampling.
    """
    ecc_by_profile = {}
    apo_by_profile = {}

    for param_ref in iter_independent:
        if param_ref.field_name == "eccentricity":
            ecc_by_profile[param_ref.profile_name] = param_ref
        elif param_ref.field_name == "apocenter":
            apo_by_profile[param_ref.profile_name] = param_ref

    return {
        profile_name: (ecc_by_profile[profile_name], apo_by_profile[profile_name])
        for profile_name in ecc_by_profile
        if profile_name in apo_by_profile
    }


def _sample_ecc_apo_joint(ecc_ref, apo_ref) -> tuple[ArrayLike, ArrayLike]:
    """
    Sample eccentricity and apocenter jointly via a Cartesian (h, k)
    reparameterization.

    This transformation ignores prior definitions currently, and places
    Normal(0, 1) priors on unconstrained raws (z_h, z_k), then squashes them
    into the open unit disk via tanh:

        s    = tanh(r) / r,   where r = sqrt(z_h^2 + z_k^2)
        h    = z_h * s
        k    = z_k * s
        e    = sqrt(h^2 + k^2)        -- guaranteed < 1
        phi0 = arctan2(h, k) mod 2pi  -- uniform by symmetry

    The induced prior on e is approximately Rayleigh-like with most mass at
    moderate eccentricities, and falls to zero at e = 0 and e = 1. This is
    physically reasonable for AGN disk emitters and is sampling-friendlier
    than the original (bounded radial, circular angular) geometry, which
    creates coupling issues in HMC near e ~ 0.
    """
    base = apo_ref.name

    z_h = numpyro.sample(f"{base}_h_raw", dist.Normal(0.0, 1.0))
    z_k = numpyro.sample(f"{base}_k_raw", dist.Normal(0.0, 1.0))

    # Squash from R^2 into the open unit disk, preserving direction.
    r = jnp.sqrt(z_h**2 + z_k**2) + 1e-12
    s = jnp.tanh(r) / r

    h = z_h * s
    k = z_k * s

    e = numpyro.deterministic(ecc_ref.name, jnp.sqrt(h**2 + k**2))
    phi0 = numpyro.deterministic(apo_ref.name, jnp.mod(jnp.arctan2(h, k), 2 * jnp.pi))

    return e, phi0


@flax.struct.dataclass
class NumpyroModel(BaseModel):
    def __call__(self, wave, flux_err, flux=None):
        # Dictionary to store all sampled parameters
        param_mods = {}

        # Pre-pass: identify profiles where both eccentricity and apocenter are
        # free and independent. These are sampled jointly via (h, k); all other
        # parameters use the standard per-parameter path.
        joint_pairs = _find_ecc_apo_pairs(self.config.template.iter_independent)
        jointly_handled = {
            ref.name
            for ecc_ref, apo_ref in joint_pairs.values()
            for ref in (ecc_ref, apo_ref)
        }

        # Sample jointly-handled (eccentricity, apocenter) pairs first.
        for ecc_ref, apo_ref in joint_pairs.values():
            e, phi0 = _sample_ecc_apo_joint(ecc_ref, apo_ref)
            param_mods[ecc_ref.name] = e
            param_mods[apo_ref.name] = phi0

        # Sample all remaining independent parameters via the standard path.
        for param_ref in self.config.template.iter_independent:
            if param_ref.name in jointly_handled:
                continue

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
            mu = numpyro.sample(f"{samp_name}_base", dist.Uniform(mu_min, mu_max))
            incl = jnp.arccos(mu)

            return numpyro.deterministic(samp_name, incl)

        if param.distribution == Distribution.UNIFORM:
            param_samp = numpyro.sample(
                samp_name, dist.Uniform(lower_bound, upper_bound)
            )

        elif param.distribution == Distribution.LOG_UNIFORM:
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
