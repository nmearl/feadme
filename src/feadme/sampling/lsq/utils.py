from astropy.modeling import Fittable1DModel, CompoundModel
from astropy.modeling.fitting import (
    TRFLSQFitter,
    model_to_fit_params,
)
import numpy as np

POSITIVE_SHIFT_FLOOR = 1e-3
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

    # Clip values too close to the bounds; NUTS struggles with parameters
    #  at the bounds since it samples in unconstrained space where the
    #  bounds are at infinity
    EPS_FRAC = 0.02  # stay within 2% of range from each bound

    for pn in [x for x in init_params]:
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
            #  must be included explicitly
            if "log_normal" in fitted_model.meta["distributions"][pn]:
                init_params[f"{pn}_base"] = np.log(init_params[pn])

    # Get real redshift samples
    redshift_z = init_params.pop("redshift_z")
    init_params["redshift"] = 1 / (1 + redshift_z) - 1

    # Retrieve deterministic values
    for pn in [k for k in init_params]:
        if "inclination" in pn:
            inclination = init_params[pn]
            init_params[f"{pn}_base"] = np.cos(inclination)
        elif "apocenter" in pn:
            apocenter = init_params[pn]
            init_params[f"{pn}_x_base"] = np.cos(apocenter)
            init_params[f"{pn}_y_base"] = np.sin(apocenter)
            e = init_params.get(pn.replace("apocenter", "eccentricity"), None)
            phi0 = init_params.get(pn)
            h = e * np.sin(phi0)
            k = e * np.cos(phi0)
            # Approximate inverse of tanh squashing for warm-start
            init_params[f"{pn}_h_raw"] = np.arctanh(np.clip(h, -0.99, 0.99))
            init_params[f"{pn}_k_raw"] = np.arctanh(np.clip(k, -0.99, 0.99))
        elif "outer_radius" in pn:
            inner_name = pn.replace("outer_radius", "inner_radius")
            inner_radius = init_params.get(inner_name)
            outer_radius = init_params[pn]

            if inner_radius is None or outer_radius is None:
                continue

            outer_low = max(mod_bounds[pn][0], inner_radius * (1.0 + 1e-6))
            outer_high = min(mod_bounds[pn][1], OUTER_RADIUS_CAP)
            outer_high = max(outer_high, outer_low * (1.0 + 1e-6))

            log_outer_low = np.log(outer_low)
            log_outer_high = np.log(outer_high)
            log_outer = np.log(np.clip(outer_radius, outer_low, outer_high))

            frac = (log_outer - log_outer_low) / (log_outer_high - log_outer_low)
            frac = np.clip(frac, 1e-6, 1.0 - 1e-6)
            raw = np.log(frac / (1.0 - frac))

            profile_name = pn.rsplit("_", 1)[0]
            init_params[f"{profile_name}_outer_radius_raw"] = raw
        elif "area" in pn:
            lo, hi = mod_bounds.get(pn, (None, None))
            if lo is None or hi is None:
                continue

            shift = max(POSITIVE_SHIFT_FLOOR, 1e-3 * (hi - lo))
            shifted_low = lo + shift
            shifted_high = hi + shift
            shifted_val = np.clip(init_params[pn] + shift, shifted_low, shifted_high)

            log_low = np.log(shifted_low)
            log_high = np.log(shifted_high)
            log_val = np.log(shifted_val)

            frac = (log_val - log_low) / (log_high - log_low)
            frac = np.clip(frac, 1e-6, 1.0 - 1e-6)
            init_params[f"{pn}_raw"] = np.log(frac / (1.0 - frac))

    init_params = {k: v.item() for k, v in init_params.items()}

    return init_params
