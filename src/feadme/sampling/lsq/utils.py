from astropy.modeling import Fittable1DModel, CompoundModel
from astropy.modeling.fitting import (
    TRFLSQFitter,
    model_to_fit_params,
)
import numpy as np


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

    init_params = {k: v.item() for k, v in init_params.items()}

    return init_params
