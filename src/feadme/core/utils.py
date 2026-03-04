import copy

import loguru
import numpy as np
from .parser import Template

logger = loguru.logger.opt(colors=True)

_SIGMA_LOG_DEFAULT = 0.5  # ~65% 1-sigma spread in log space for log_normal

# Multiplicative window applied around the LSQ estimate for log-scale params.
# e.g. 20 means bounds become [lsq/20, lsq*20], clipped to skeleton limits.
_LOG_BOUND_FACTOR = 20.0

# Multiplier applied to the LSQ estimate to set the upper bound for
# non-negative uniform params, e.g. area.
_UNIFORM_HIGH_FACTOR = 5.0


def _fill_param_from_lsq(
    param: dict,
    lsq_value: float,
    sigma_log: float = _SIGMA_LOG_DEFAULT,
    log_bound_factor: float = _LOG_BOUND_FACTOR,
    uniform_high_factor: float = _UNIFORM_HIGH_FACTOR,
) -> dict:
    """
    Return a copy of a parameter dict with loc/scale/value AND low/high filled
    from an LSQ-derived point estimate.

    Only called when ``param["loc"] is None`` (sentinel). The skeleton's
    original ``low``/``high`` are used as hard caps so the updated bounds
    never exceed what is physically meaningful.

    Parameters
    ----------
    param :
        Raw parameter dict (from ``Template.to_dict()``).
    lsq_value :
        Point estimate from the LSQ fit, in linear (non-log) space.
    sigma_log :
        Log-space width for log_normal scale derivation. Default 0.5.
    log_bound_factor :
        Multiplicative window for log-scale bounds: [lsq/factor, lsq*factor].
    uniform_high_factor :
        Upper bound for non-negative uniform params: lsq * factor.
    """
    p = copy.deepcopy(param)
    dist = p.get("distribution", "uniform")
    skeleton_low = p["low"]  # hard floor from skeleton
    skeleton_high = p["high"]  # hard ceiling from skeleton

    # Clip lsq_value strictly inside skeleton bounds.
    if lsq_value <= skeleton_low:
        logger.warning(
            "LSQ value %.4g <= skeleton low=%.4g; clipping.",
            lsq_value,
            skeleton_low,
        )
        lsq_value = skeleton_low * 1.02 if skeleton_low > 0 else skeleton_low + 1e-4
    if lsq_value >= skeleton_high:
        logger.warning(
            "LSQ value %.4g >= skeleton high=%.4g; clipping.",
            lsq_value,
            skeleton_high,
        )
        lsq_value = skeleton_high * 0.98

    p["value"] = lsq_value
    p["loc"] = lsq_value

    if dist in ("log_normal", "log_uniform"):
        # Tighten bounds to a multiplicative window, capped by skeleton limits.
        new_low = max(skeleton_low, lsq_value / log_bound_factor)
        new_high = min(skeleton_high, lsq_value * log_bound_factor)
        p["low"] = new_low
        p["high"] = new_high

        if dist == "log_normal":
            p["scale"] = lsq_value * np.sqrt(np.exp(sigma_log**2) - 1.0)

    elif dist == "uniform":
        # For non-negative params (low >= 0), set upper bound as a multiple of
        # the LSQ value; lower bound stays at skeleton_low (usually 0).
        if skeleton_low >= 0:
            new_high = min(skeleton_high, lsq_value * uniform_high_factor)
            new_high = max(new_high, lsq_value * 1.5)  # ensure lsq_value isn't near top
            p["high"] = new_high
        p["scale"] = (p["high"] - p["low"]) * 0.1

    elif dist == "normal":
        # Bounds are just truncation guards; set a generous window around loc.
        if p.get("scale") is not None:
            guard = p["scale"] * 5.0
        else:
            guard = abs(lsq_value) * 0.5 + 1e-4
        p["low"] = max(skeleton_low, lsq_value - guard)
        p["high"] = min(skeleton_high, lsq_value + guard)

    return p


def update_from_lsq(
    template: Template,
    init_params: dict,
    sigma_log: float = _SIGMA_LOG_DEFAULT,
) -> Template:
    """
    Return a new ``Template`` with sentinel parameters filled from LSQ results.

    A parameter is a sentinel when its ``loc`` field is ``None``. Parameters
    with a concrete ``loc`` are left entirely untouched.

    For sentinel parameters, both the distributional shape (loc/scale/value)
    AND the prior bounds (low/high) are updated. The skeleton's original bounds
    serve as hard caps — updated bounds are always a subset of them.

    Parameters
    ----------
    init_params :
        ``{qualified_name: float}`` dict returned by ``LSQInitializer``.
        Keys follow ``"{profile_name}_{field_name}"``.
        Values must be in *linear* (non-log) space.
    sigma_log :
        Log-space width for ``log_normal`` scale derivation. Default 0.5.

    Returns
    -------
    Template
        A new ``Template`` instance; ``self`` is unchanged.
    """
    raw = template.to_dict()

    def _is_param_dict(v) -> bool:
        return isinstance(v, dict) and "distribution" in v and "fixed" in v

    # ---- disk profiles -------------------------------------------------------
    for prof_dict in raw.get("disk_profiles", []):
        prof_name = prof_dict.get("name", "")
        for field_name, param_dict in list(prof_dict.items()):
            if not _is_param_dict(param_dict):
                continue
            if param_dict.get("fixed", False):
                continue
            # if param_dict.get("loc") is not None:
            #     continue  # user-specified; do not touch

            qname = f"{prof_name}_{field_name}"
            if qname not in init_params:
                logger.debug(f"No LSQ value for '{qname}'; leaving as sentinel.")
                continue

            prof_dict[field_name] = _fill_param_from_lsq(
                param_dict, float(init_params[qname]), sigma_log=sigma_log
            )
            logger.debug(
                f"Updated '{qname}': loc={prof_dict[field_name]["loc"]:.4g}, low={prof_dict[field_name]["low"]:.4g}, high={prof_dict[field_name]["high"]:.4g}"
            )

    # ---- line profiles -------------------------------------------------------
    for prof_dict in raw.get("line_profiles", []):
        prof_name = prof_dict.get("name", "")
        for field_name, param_dict in list(prof_dict.items()):
            if not _is_param_dict(param_dict):
                continue
            if param_dict.get("fixed", False):
                continue
            if param_dict.get("shared") is not None:
                continue  # shared params inherit from source profile
            # if param_dict.get("loc") is not None:
            #     continue  # user-specified; do not touch

            qname = f"{prof_name}_{field_name}"
            if qname not in init_params:
                logger.debug(f"No LSQ value for '{qname}'; leaving as sentinel.")
                continue

            prof_dict[field_name] = _fill_param_from_lsq(
                param_dict, float(init_params[qname]), sigma_log=sigma_log
            )
            logger.debug(
                f"Updated '{qname}': loc={prof_dict[field_name]["loc"]:.4g}, low={prof_dict[field_name]["low"]:.4g}, high={prof_dict[field_name]["high"]:.4g}"
            )

    # ---- redshift ------------------------------------------------------------
    redshift_dict = raw.get("redshift", {})
    if (
        _is_param_dict(redshift_dict)
        and not redshift_dict.get("fixed", False)
        and redshift_dict.get("loc") is None
        and "redshift" in init_params
    ):
        z_lsq = float(init_params["redshift"])
        skeleton_z_low = redshift_dict.get("low", 0.0)
        skeleton_z_high = redshift_dict.get("high", 1.0)
        redshift_dict.update(
            loc=z_lsq,
            value=z_lsq,
            low=max(skeleton_z_low, z_lsq - 0.01),
            high=min(skeleton_z_high, z_lsq + 0.01),
            scale=0.002,
        )
        raw["redshift"] = redshift_dict

    from pprint import pprint

    pprint(raw)

    return Template.from_dict(raw)
