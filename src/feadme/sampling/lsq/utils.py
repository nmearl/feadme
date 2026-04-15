from astropy.modeling import Fittable1DModel, CompoundModel
from astropy.modeling.fitting import (
    TRFLSQFitter,
    model_to_fit_params,
)
import numpy as np

OUTER_RADIUS_CAP = 1e5


def format_posterior_samples(
    fitted_model: CompoundModel,
    wave: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
) -> dict[str, float]:
    # Construct fitted parameters dictionary
    model_names = [m.name for m in fitted_model]
    param_name_map = {}

    _, indices, _ = model_to_fit_params(fitted_model)
    fit_param_names = np.array(fitted_model.param_names)
    fit_params = np.array(fitted_model.parameters)
    fit_param_map = dict(zip(fit_param_names, fit_params))

    for fpn in fit_param_names:
        pn, pi = "_".join(fpn.split("_")[:-1]), int(fpn.split("_")[-1])
        param_name_map[fpn] = f"{model_names[pi]}_{pn}"

    init_params = {param_name_map[k]: fit_param_map[k] for k in fit_param_names}
    mod_bounds = {param_name_map[k]: fitted_model.bounds[k] for k in fit_param_names}
    distributions = fitted_model.meta["distributions"]

    def _get_linear_bounds(pn):
        """Return (low, high) in physical (linear) space for any parameter."""
        lo, hi = mod_bounds[pn]
        if lo is not None and hi is not None and "log" in distributions.get(pn, ""):
            lo, hi = 10**lo, 10**hi
        return lo, hi

    # Clip values too close to the bounds; NUTS struggles with parameters
    # at the bounds since it samples in unconstrained space where the
    # bounds are at infinity
    EPS_FRAC = 0.02  # stay within 2% of range from each bound

    for pn in list(init_params):
        if pn not in mod_bounds or (
            mod_bounds[pn][0] is None and mod_bounds[pn][1] is None
        ):
            continue

        lo, hi = mod_bounds[pn]
        margin = EPS_FRAC * (hi - lo)
        init_params[pn] = np.clip(init_params[pn], lo + margin, hi - margin)

    # Transform log-based parameters back to linear space
    for pn in fitted_model.meta["distributions"]:
        if "log" in fitted_model.meta["distributions"][pn]:
            init_params[pn] = 10 ** init_params.pop(pn)

            # Log normal parameters use a separate sampling site, so they
            # must be included explicitly
            if "log_normal" in fitted_model.meta["distributions"][pn]:
                init_params[f"{pn}_base"] = np.log(init_params[pn])

    # Get real redshift samples
    redshift_z = init_params.pop("redshift_z")
    init_params["redshift"] = 1 / (1 + redshift_z) - 1

    # Add latent raw/base sites for joint samplers so that LSQ values
    # propagate through to all numpyro sample sites.
    for pn in list(init_params):
        if "inclination" in pn:
            inclination = init_params[pn]
            init_params[f"{pn}_base"] = np.cos(inclination)

        elif "apocenter" in pn:
            apocenter = init_params[pn]
            # Circular fallback (apocenter sampled independently via sample_param)
            init_params[f"{pn}_x_base"] = np.cos(apocenter)
            init_params[f"{pn}_y_base"] = np.sin(apocenter)
            # Joint ecc/apo sampler: invert e = tanh(r), phi0 = arctan2(z_h, z_k)
            # => r = arctanh(e), z_h = r*sin(phi0), z_k = r*cos(phi0)
            e = init_params.get(pn.replace("apocenter", "eccentricity"))
            if e is not None:
                r = np.arctanh(np.clip(float(e), 0.0, 0.9999))
                init_params[f"{pn}_h_raw"] = r * np.sin(apocenter)
                init_params[f"{pn}_k_raw"] = r * np.cos(apocenter)

        elif "outer_radius" in pn:
            inner_name = pn.replace("outer_radius", "inner_radius")
            inner_radius = init_params.get(inner_name)
            outer_radius = init_params[pn]

            if inner_radius is None or outer_radius is None:
                continue

            # Use linear bounds (mod_bounds stores log10 for log-distributed params)
            outer_lo, outer_hi = _get_linear_bounds(pn)
            outer_low = max(outer_lo, inner_radius * (1.0 + 1e-6))
            outer_high = min(outer_hi, OUTER_RADIUS_CAP)
            outer_high = max(outer_high, outer_low * (1.0 + 1e-6))

            log_outer_low = np.log(outer_low)
            log_outer_high = np.log(outer_high)
            log_outer = np.log(np.clip(outer_radius, outer_low, outer_high))

            frac = (log_outer - log_outer_low) / (log_outer_high - log_outer_low)
            frac = np.clip(frac, 1e-6, 1.0 - 1e-6)
            raw = np.log(frac / (1.0 - frac))

            profile_name = pn.replace("_outer_radius", "")
            init_params[f"{profile_name}_outer_radius_raw"] = raw

    # Joint area sampler: detect paired disk/broad area profiles and invert
    # the (total_area, disk_fraction) reparameterization.
    disk_area_pns = {
        k
        for k in init_params
        if k.endswith("_area") and "_disk" in k[: k.rfind("_area")]
    }
    broad_area_pns = {
        k
        for k in init_params
        if k.endswith("_area") and "_broad" in k[: k.rfind("_area")]
    }

    # Build stem → key maps (stem = profile name with _disk/_broad removed)
    disk_by_stem = {k.replace("_disk", "", 1).replace("_area", ""): k for k in disk_area_pns}
    broad_by_stem = {k.replace("_broad", "", 1).replace("_area", ""): k for k in broad_area_pns}

    for stem in disk_by_stem:
        if stem not in broad_by_stem:
            continue

        disk_key = disk_by_stem[stem]
        broad_key = broad_by_stem[stem]
        disk_area = float(init_params[disk_key])
        broad_area = float(init_params[broad_key])

        disk_lo, disk_hi = _get_linear_bounds(disk_key)
        broad_lo, broad_hi = _get_linear_bounds(broad_key)

        total_area = disk_area + broad_area
        total_low = max(disk_lo + broad_lo, 1e-8)
        total_high = disk_hi + broad_hi
        total_high = max(total_high, total_low * (1.0 + 1e-6))

        log_total_low = np.log(total_low)
        log_total_high = np.log(total_high)
        log_total = np.log(np.clip(total_area, total_low, total_high))

        total_frac = (log_total - log_total_low) / (log_total_high - log_total_low)
        total_frac = np.clip(total_frac, 1e-6, 1.0 - 1e-6)
        init_params[f"{stem}_total_area_raw"] = float(np.log(total_frac / (1.0 - total_frac)))

        frac_low = max(disk_lo / total_area, 1.0 - broad_hi / total_area)
        frac_high = min(disk_hi / total_area, 1.0 - broad_lo / total_area)
        frac_low = np.clip(frac_low, 1e-8, 1.0 - 1e-8)
        frac_high = np.clip(frac_high, frac_low + 1e-8, 1.0)

        disk_fraction = np.clip(disk_area / total_area, frac_low, frac_high)
        frac_unit = (disk_fraction - frac_low) / (frac_high - frac_low)
        frac_unit = np.clip(frac_unit, 1e-6, 1.0 - 1e-6)
        init_params[f"{stem}_disk_fraction_raw"] = float(np.log(frac_unit / (1.0 - frac_unit)))

    init_params = {k: v.item() if hasattr(v, "item") else v for k, v in init_params.items()}

    return init_params
