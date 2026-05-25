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
DISK_RADIUS_RATIO_LOC = 10.0
DISK_RADIUS_RATIO_SCALE = 6.0
DISK_RADIUS_RATIO_LOW = 2.0
DISK_RADIUS_RATIO_HIGH = 22.0


def ratio_factor(name, F_num, F_den, ratio, sigma_ln=0.05, eps=1e-30):
    """
    Softly constrain F_num / F_den ~= ratio via a Normal penalty in log-space.
    sigma_ln is the 1-sigma width in ln(ratio).
      - sigma_ln=0.05 ~ 5% (moderately tight)
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


def _eccentricity_apocenter_from_raw(
    z_h: ArrayLike,
    z_k: ArrayLike,
    eccentricity_low: float,
    eccentricity_high: float,
) -> tuple[ArrayLike, ArrayLike, ArrayLike]:
    raw_r = jnp.sqrt(z_h**2 + z_k**2)
    safe_r = raw_r + 1e-12
    unit_e = jnp.tanh(raw_r)
    eccentricity_span = eccentricity_high - eccentricity_low
    eccentricity = eccentricity_low + eccentricity_span * unit_e
    apocenter = jnp.mod(jnp.arctan2(z_h, z_k), 2 * jnp.pi)
    log_abs_det = (
        jnp.log(eccentricity_span)
        + jnp.log1p(-(unit_e**2))
        - jnp.log(safe_r)
    )
    return eccentricity, apocenter, log_abs_det


def _sample_ecc_apo_joint(ecc_ref, apo_ref) -> tuple[ArrayLike, ArrayLike]:
    """
    Sample eccentricity and apocenter jointly via a Cartesian (h, k)
    reparameterization.

    Sample base raws in Cartesian coordinates, then correct back to the
    template priors on (eccentricity, apocenter) with an explicit factor.

    The geometric map is:

        u    = tanh(r), where r = sqrt(z_h^2 + z_k^2)
        e    = e_low + (e_high - e_low) * u
        phi0 = arctan2(z_h, z_k) mod 2pi

    We keep the Cartesian geometry for sampling, but restore the intended
    target density by applying the configured priors on e and phi0 together
    with the log-abs-det Jacobian of the transform. The transform maps directly
    into the configured eccentricity interval, so template support is enforced
    as hard support rather than only through the prior correction factor.
    """
    base = apo_ref.name

    base_dist = dist.Normal(0.0, 1.0)
    z_h = numpyro.sample(f"{base}_h_raw", base_dist)
    z_k = numpyro.sample(f"{base}_k_raw", base_dist)

    e, phi0, log_abs_det = _eccentricity_apocenter_from_raw(
        z_h,
        z_k,
        ecc_ref.param.low,
        ecc_ref.param.high,
    )
    e = numpyro.deterministic(ecc_ref.name, e)
    phi0 = numpyro.deterministic(apo_ref.name, phi0)

    ecc_prior = _distribution_from_param(
        ecc_ref.param, ecc_ref.param.low, ecc_ref.param.high
    )
    apo_prior = _distribution_from_param(
        apo_ref.param, apo_ref.param.low, apo_ref.param.high
    )

    log_target = ecc_prior.log_prob(e) + apo_prior.log_prob(phi0)
    log_base = base_dist.log_prob(z_h) + base_dist.log_prob(z_k)

    numpyro.factor(f"{base}_prior_correction", log_target + log_abs_det - log_base)

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
