import astropy.constants as const
import astropy.units as u
import flax.struct
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
from jax.typing import ArrayLike
from numpyro.distributions.transforms import ExpTransform

from ..base_model import BaseModel
from ...core.evaluators import (
    evaluate_model,
)
from ...core.parser import Distribution, Parameter

ERR = float(np.finfo(np.float32).tiny)
EPS = 1e-6
c_cgs = const.c.cgs.value
c_kms = const.c.to(u.km / u.s).value
OUTER_RADIUS_CAP = 1e5


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


def _find_radius_pairs(
    iter_independent,
) -> dict[str, tuple]:
    """
    Identify (inner_radius, outer_radius) pairs that can be handled jointly
    with an ordering-preserving transform.
    """
    inner_by_profile = {}
    outer_by_profile = {}

    for param_ref in iter_independent:
        if param_ref.field_name == "inner_radius":
            inner_by_profile[param_ref.profile_name] = param_ref
        elif param_ref.field_name == "outer_radius":
            outer_by_profile[param_ref.profile_name] = param_ref

    return {
        profile_name: (inner_by_profile[profile_name], outer_by_profile[profile_name])
        for profile_name in inner_by_profile
        if profile_name in outer_by_profile
    }


def _find_area_pairs(
    iter_independent,
) -> dict[str, tuple]:
    """
    Identify competing disk/broad area pairs that should be sampled jointly
    as (total_area, disk_fraction).

    Pairing is based on a shared profile stem after removing the first
    ``_disk`` or ``_broad`` token from the profile name, e.g.
    ``halpha_disk`` <-> ``halpha_broad`` and ``halpha_disk_2`` <->
    ``halpha_broad_2``.

    Ambiguous stems with multiple disks or multiple broad components are
    skipped so the standard per-parameter path remains in control.
    """
    disk_by_stem = {}
    broad_by_stem = {}

    for param_ref in iter_independent:
        if param_ref.field_name != "area" or param_ref.profile_name is None:
            continue

        profile_name = param_ref.profile_name
        if "_disk" in profile_name:
            stem = profile_name.replace("_disk", "", 1)
            disk_by_stem.setdefault(stem, []).append(param_ref)
        elif "_broad" in profile_name:
            stem = profile_name.replace("_broad", "", 1)
            broad_by_stem.setdefault(stem, []).append(param_ref)

    return {
        stem: (disk_refs[0], broad_refs[0])
        for stem, disk_refs in disk_by_stem.items()
        if stem in broad_by_stem
        for broad_refs in [broad_by_stem[stem]]
        if len(disk_refs) == 1 and len(broad_refs) == 1
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
        return dist.TransformedDistribution(
            dist.TruncatedNormal(
                loc=mu_log,
                scale=sigma_log,
                low=jnp.log(lower_bound),
                high=jnp.log(upper_bound),
            ),
            ExpTransform(),
        )

    raise ValueError(f"Unsupported distribution: {param.distribution}")


def _sample_radius_joint(inner_ref, outer_ref) -> tuple[ArrayLike, ArrayLike]:
    """
    Sample the disk radial extent via an ordered (inner_radius, outer_radius)
    pair, with the outer radius constrained to remain above the sampled inner
    radius and below the global hard cap.

    The outer-radius transform enforces ordering and the global cap directly,
    replacing the previous soft wall in outer-radius space.
    """
    base = inner_ref.profile_name

    inner_prior = _distribution_from_param(
        inner_ref.param, inner_ref.param.low, inner_ref.param.high
    )
    inner_radius = numpyro.sample(inner_ref.name, inner_prior)

    outer_low = jnp.maximum(outer_ref.param.low, inner_radius * (1.0 + 1e-6))
    outer_high = jnp.minimum(outer_ref.param.high, OUTER_RADIUS_CAP)
    outer_high = jnp.maximum(outer_high, outer_low * (1.0 + 1e-6))

    outer_raw_dist = dist.Normal(0.0, 1.0)
    outer_raw = numpyro.sample(f"{base}_outer_radius_raw", outer_raw_dist)
    outer_frac = jax.nn.sigmoid(outer_raw)

    log_outer_low = jnp.log(outer_low)
    log_outer_high = jnp.log(outer_high)
    log_outer = log_outer_low + outer_frac * (log_outer_high - log_outer_low)
    outer_radius = numpyro.deterministic(outer_ref.name, jnp.exp(log_outer))

    outer_prior = _distribution_from_param(
        outer_ref.param, outer_ref.param.low, outer_ref.param.high
    )
    log_target = outer_prior.log_prob(outer_radius)
    log_base = outer_raw_dist.log_prob(outer_raw)

    # outer = exp(log_outer) and log_outer is affine in sigmoid(raw)
    log_abs_det = (
        jnp.log(outer_radius)
        + jnp.log(log_outer_high - log_outer_low)
        + jnp.log(outer_frac)
        + jnp.log1p(-outer_frac)
    )
    numpyro.factor(
        f"{base}_outer_radius_prior_correction", log_target + log_abs_det - log_base
    )

    return inner_radius, outer_radius


def _sample_ecc_apo_joint(ecc_ref, apo_ref) -> tuple[ArrayLike, ArrayLike]:
    """
    Sample eccentricity and apocenter jointly via a Cartesian (h, k)
    reparameterization.

    Sample base raws in Cartesian coordinates, then correct back to the
    template priors on (eccentricity, apocenter) with an explicit factor.

    The geometric map is:

        s    = tanh(r) / r,   where r = sqrt(z_h^2 + z_k^2)
        h    = z_h * s
        k    = z_k * s
        e    = sqrt(h^2 + k^2)        -- guaranteed < 1
        phi0 = arctan2(h, k) mod 2pi  -- uniform by symmetry

    We keep the Cartesian geometry for sampling, but restore the intended
    target density by applying the configured priors on e and phi0 together
    with the log-abs-det Jacobian of the transform.
    """
    base = apo_ref.name

    base_dist = dist.Normal(0.0, 1.0)
    z_h = numpyro.sample(f"{base}_h_raw", base_dist)
    z_k = numpyro.sample(f"{base}_k_raw", base_dist)

    # Squash from R^2 into the open unit disk, preserving direction.
    r = jnp.sqrt(z_h**2 + z_k**2) + 1e-12
    s = jnp.tanh(r) / r

    h = z_h * s
    k = z_k * s

    e = numpyro.deterministic(ecc_ref.name, jnp.sqrt(h**2 + k**2))
    phi0 = numpyro.deterministic(apo_ref.name, jnp.mod(jnp.arctan2(h, k), 2 * jnp.pi))

    ecc_prior = _distribution_from_param(
        ecc_ref.param, ecc_ref.param.low, ecc_ref.param.high
    )
    apo_prior = _distribution_from_param(
        apo_ref.param, apo_ref.param.low, apo_ref.param.high
    )

    log_target = ecc_prior.log_prob(e) + apo_prior.log_prob(phi0)
    log_base = base_dist.log_prob(z_h) + base_dist.log_prob(z_k)

    # In polar form z_h = r sin(phi0), z_k = r cos(phi0), and e = tanh(r).
    # Therefore |d(e, phi0) / d(z_h, z_k)| = (1 - e^2) / r.
    log_abs_det = jnp.log1p(-(e**2)) - jnp.log(r)
    numpyro.factor(f"{base}_prior_correction", log_target + log_abs_det - log_base)

    return e, phi0


def _sample_area_joint(disk_ref, broad_ref) -> tuple[ArrayLike, ArrayLike]:
    """
    Sample a competing disk/broad area pair via a positive total area and a
    disk flux fraction, then correct back to the configured component priors.
    """
    base = disk_ref.profile_name.replace("_disk", "", 1)
    eps = 1e-8

    total_low = max(disk_ref.param.low + broad_ref.param.low, eps)
    total_high = max(disk_ref.param.high + broad_ref.param.high, total_low * (1.0 + 1e-6))

    total_raw_dist = dist.Normal(0.0, 1.0)
    total_raw = numpyro.sample(f"{base}_total_area_raw", total_raw_dist)
    total_frac = jax.nn.sigmoid(total_raw)

    log_total_low = jnp.log(total_low)
    log_total_high = jnp.log(total_high)
    log_total_area = log_total_low + total_frac * (log_total_high - log_total_low)
    total_area = jnp.exp(log_total_area)

    frac_low = jnp.maximum(
        disk_ref.param.low / total_area,
        1.0 - broad_ref.param.high / total_area,
    )
    frac_high = jnp.minimum(
        disk_ref.param.high / total_area,
        1.0 - broad_ref.param.low / total_area,
    )
    frac_low = jnp.clip(frac_low, eps, 1.0 - eps)
    frac_high = jnp.clip(frac_high, frac_low + eps, 1.0)

    frac_raw_dist = dist.Normal(0.0, 1.0)
    frac_raw = numpyro.sample(f"{base}_disk_fraction_raw", frac_raw_dist)
    frac_unit = jax.nn.sigmoid(frac_raw)
    disk_fraction = frac_low + frac_unit * (frac_high - frac_low)

    disk_area = numpyro.deterministic(disk_ref.name, total_area * disk_fraction)
    broad_area = numpyro.deterministic(
        broad_ref.name, total_area * (1.0 - disk_fraction)
    )

    disk_prior = _distribution_from_param(
        disk_ref.param, disk_ref.param.low, disk_ref.param.high
    )
    broad_prior = _distribution_from_param(
        broad_ref.param, broad_ref.param.low, broad_ref.param.high
    )

    log_target = disk_prior.log_prob(disk_area) + broad_prior.log_prob(broad_area)
    log_base = total_raw_dist.log_prob(total_raw) + frac_raw_dist.log_prob(frac_raw)

    log_abs_det = (
        jnp.log(total_area)
        + jnp.log(log_total_high - log_total_low)
        + jnp.log(total_frac)
        + jnp.log1p(-total_frac)
        + jnp.log(frac_high - frac_low)
        + jnp.log(frac_unit)
        + jnp.log1p(-frac_unit)
        + jnp.log(total_area)
    )
    numpyro.factor(
        f"{base}_area_prior_correction", log_target + log_abs_det - log_base
    )

    return disk_area, broad_area




@flax.struct.dataclass
class NumpyroModel(BaseModel):
    def __call__(self, wave, flux_err, flux=None):
        # Dictionary to store all sampled parameters
        param_mods = {}

        # Pre-pass: identify profiles where both eccentricity and apocenter are
        # free and independent. These are sampled jointly via (h, k); all other
        # parameters use the standard per-parameter path.
        joint_pairs = _find_ecc_apo_pairs(self.config.template.iter_independent)
        radius_pairs = _find_radius_pairs(self.config.template.iter_independent)
        area_pairs = _find_area_pairs(self.config.template.iter_independent)
        jointly_handled = {
            ref.name
            for ecc_ref, apo_ref in joint_pairs.values()
            for ref in (ecc_ref, apo_ref)
        }
        jointly_handled |= {
            ref.name
            for inner_ref, ratio_ref in radius_pairs.values()
            for ref in (inner_ref, ratio_ref)
        }
        jointly_handled |= {
            ref.name
            for disk_ref, broad_ref in area_pairs.values()
            for ref in (disk_ref, broad_ref)
        }

        # Sample jointly-handled (eccentricity, apocenter) pairs first.
        for ecc_ref, apo_ref in joint_pairs.values():
            e, phi0 = _sample_ecc_apo_joint(ecc_ref, apo_ref)
            param_mods[ecc_ref.name] = e
            param_mods[apo_ref.name] = phi0

        for inner_ref, ratio_ref in radius_pairs.values():
            inner_radius, outer_radius = _sample_radius_joint(inner_ref, ratio_ref)
            param_mods[inner_ref.name] = inner_radius
            param_mods[ratio_ref.name] = outer_radius

        for disk_ref, broad_ref in area_pairs.values():
            disk_area, broad_area = _sample_area_joint(disk_ref, broad_ref)
            param_mods[disk_ref.name] = disk_area
            param_mods[broad_ref.name] = broad_area

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

        # Soft constraint: [NII] 6583/6548 ratio should be ~2.95
        niil_narrow_area = param_mods.get("niil_narrow_area")
        niir_narrow_area = param_mods.get("niir_narrow_area")

        if niil_narrow_area is not None and niir_narrow_area is not None:
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
