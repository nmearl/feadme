import time
import csv
from pathlib import Path

import flax.struct
import jax.numpy as jnp
import jax.random as random
import loguru
import optax
import corner
import jax
from astropy.modeling.fitting import (
    TRFLSQFitter,
    DogBoxLSQFitter,
)
import matplotlib.pyplot as plt
from numpyro.handlers import seed, trace
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer import init_to_median, init_to_value
from numpyro.infer.autoguide import (
    AutoBNAFNormal,
    AutoMultivariateNormal,
)
from numpyro.infer.util import initialize_model, log_density
import numpy as np
import astropy.constants as const
import astropy.units as u

from .lsq.utils import format_posterior_samples
from ..core.parser import Config
from .base_model import BaseModel
from .lsq.model import _compose_model
from .numpyro.model import (
    DISK_RADIUS_RATIO_HIGH,
    DISK_RADIUS_RATIO_LOW,
)
from .utils import diagnose_init_params

logger = loguru.logger.opt(colors=True)

c_kms = const.c.to(u.km / u.s).value
INIT_BOUNDARY_MARGIN_FRAC = 1e-2


def _resolve_bounds(
    config: Config, full_name: str
) -> tuple[float | None, float | None]:
    param_ref = next((p for p in config.template.iter_all if p.name == full_name), None)
    if param_ref is not None:
        return param_ref.param.low, param_ref.param.high

    if full_name.endswith("_outer_radius"):
        profile_name = full_name.removesuffix("_outer_radius")
        disk = next(
            (
                profile
                for profile in config.template.disk_profiles
                if profile.name == profile_name
            ),
            None,
        )
        if (
            disk is not None
            and disk.inner_radius is not None
            and disk.radius_ratio is not None
        ):
            return (
                disk.inner_radius.low * disk.radius_ratio.low,
                disk.inner_radius.high * disk.radius_ratio.high,
            )
        return None, None
    return None, None


def _interp_within_bounds(
    low: float | None,
    high: float | None,
    frac: float,
    *,
    log_space: bool = False,
) -> float | None:
    if low is None or high is None:
        return None
    if not np.isfinite(low) or not np.isfinite(high) or not high > low:
        return None
    frac = float(np.clip(frac, 0.0, 1.0))
    if log_space and low > 0 and high > 0:
        return float(10 ** (np.log10(low) + frac * (np.log10(high) - np.log10(low))))
    return float(low + frac * (high - low))


def _apply_structured_start(
    candidate: dict[str, float],
    config: Config,
    *,
    profile_name: str,
    field_name: str,
    frac: float,
    log_space: bool = False,
):
    full_name = f"{profile_name}_{field_name}"
    low, high = _resolve_bounds(config, full_name)
    value = _interp_within_bounds(low, high, frac, log_space=log_space)
    if value is not None:
        candidate[full_name] = value


def _candidate_key(candidate: dict[str, float]) -> tuple[tuple[str, float], ...]:
    return tuple(sorted((k, round(float(v), 10)) for k, v in candidate.items()))


def _dedupe_start_candidates(
    candidates: list[dict[str, float]],
) -> list[dict[str, float]]:
    unique: list[dict[str, float]] = []
    seen: set[tuple[tuple[str, float], ...]] = set()
    for candidate in candidates:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _van_der_corput(index: int, base: int) -> float:
    value = 0.0
    denom = 1.0
    while index:
        index, remainder = divmod(index, base)
        denom *= base
        value += remainder / denom
    return value


def _expanded_lsq_candidate(config: Config, index: int) -> dict[str, float]:
    disk_profiles = list(config.template.disk_profiles)
    line_profiles = list(config.template.line_profiles)
    broad_profiles = [prof for prof in line_profiles if "broad" in (prof.name or "")]
    candidate: dict[str, float] = {}

    disk_fields = [
        ("inner_radius", 2, True),
        ("outer_radius", 3, True),
        ("inclination", 5, False),
        ("sigma", 7, True),
        ("q", 11, False),
        ("eccentricity", 13, False),
        ("apocenter", 17, False),
    ]
    for prof in disk_profiles:
        fractions = {
            field_name: 0.08 + 0.84 * _van_der_corput(index + offset, base)
            for field_name, base, _ in disk_fields
            for offset in [base]
        }
        # Bias expanded starts toward valid extended disks instead of wasting
        # LSQ attempts on nearly collapsed radius orderings.
        fractions["outer_radius"] = max(
            fractions["outer_radius"],
            min(0.94, fractions["inner_radius"] + 0.18),
        )
        for field_name, _, log_space in disk_fields:
            _apply_structured_start(
                candidate,
                config,
                profile_name=prof.name,
                field_name=field_name,
                frac=fractions[field_name],
                log_space=log_space,
            )

    broad_fields = [
        ("vel_width", 19, True),
        ("offset", 23, False),
        ("area", 29, True),
    ]
    for prof in broad_profiles:
        for field_name, base, log_space in broad_fields:
            frac = 0.08 + 0.84 * _van_der_corput(index + base, base)
            _apply_structured_start(
                candidate,
                config,
                profile_name=prof.name,
                field_name=field_name,
                frac=frac,
                log_space=log_space,
            )

    return candidate


def _structured_lsq_candidates(
    config: Config, target_count: int | None = None
) -> list[dict[str, float]]:
    candidates: list[dict[str, float]] = [{}]

    disk_profiles = list(config.template.disk_profiles)
    line_profiles = list(config.template.line_profiles)
    broad_profiles = [prof for prof in line_profiles if "broad" in (prof.name or "")]

    if disk_profiles:
        disk_presets = [
            {
                "inner_radius": (0.12, True),
                "outer_radius": (0.28, True),
                "inclination": (0.18, False),
                "sigma": (0.25, True),
                "q": (0.45, False),
                "eccentricity": (0.15, False),
                "apocenter": (0.18, False),
            },
            {
                "inner_radius": (0.18, True),
                "outer_radius": (0.34, True),
                "inclination": (0.78, False),
                "sigma": (0.35, True),
                "q": (0.30, False),
                "eccentricity": (0.82, False),
                "apocenter": (0.62, False),
            },
            {
                "inner_radius": (0.28, True),
                "outer_radius": (0.72, True),
                "inclination": (0.24, False),
                "sigma": (0.55, True),
                "q": (0.55, False),
                "eccentricity": (0.45, False),
                "apocenter": (0.38, False),
            },
            {
                "inner_radius": (0.35, True),
                "outer_radius": (0.82, True),
                "inclination": (0.82, False),
                "sigma": (0.68, True),
                "q": (0.25, False),
                "eccentricity": (0.88, False),
                "apocenter": (0.85, False),
            },
            {
                "inner_radius": (0.22, True),
                "outer_radius": (0.52, True),
                "inclination": (0.52, False),
                "sigma": (0.42, True),
                "q": (0.82, False),
                "eccentricity": (0.55, False),
                "apocenter": (0.12, False),
            },
            {
                "inner_radius": (0.10, True),
                "outer_radius": (0.26, True),
                "inclination": (0.48, False),
                "sigma": (0.12, True),
                "q": (0.52, False),
                "eccentricity": (0.72, False),
                "apocenter": (0.72, False),
            },
        ]

        for preset in disk_presets:
            candidate: dict[str, float] = {}
            for prof in disk_profiles:
                for field_name, (frac, log_space) in preset.items():
                    _apply_structured_start(
                        candidate,
                        config,
                        profile_name=prof.name,
                        field_name=field_name,
                        frac=frac,
                        log_space=log_space,
                    )
            if candidate:
                candidates.append(candidate)

    if broad_profiles:
        broad_presets = [
            {
                "vel_width": (0.18, True),
                "offset": (0.20, False),
                "area": (0.22, True),
            },
            {
                "vel_width": (0.42, True),
                "offset": (0.50, False),
                "area": (0.35, True),
            },
            {
                "vel_width": (0.78, True),
                "offset": (0.80, False),
                "area": (0.55, True),
            },
        ]
        for preset in broad_presets:
            candidate: dict[str, float] = {}
            for prof in broad_profiles:
                for field_name, (frac, log_space) in preset.items():
                    _apply_structured_start(
                        candidate,
                        config,
                        profile_name=prof.name,
                        field_name=field_name,
                        frac=frac,
                        log_space=log_space,
                    )
            if candidate:
                candidates.append(candidate)

    combined: list[dict[str, float]] = []
    disk_only = [c for c in candidates if c]
    if disk_profiles and broad_profiles:
        disk_candidates = [
            c
            for c in disk_only
            if any("_inclination" in k or "_eccentricity" in k for k in c)
        ][:4]
        broad_candidates = [
            c for c in disk_only if any("_vel_width" in k and "broad" in k for k in c)
        ][:3]
        for d in disk_candidates:
            for b in broad_candidates:
                combined.append({**d, **b})

    unique = _dedupe_start_candidates(candidates + combined)
    if target_count is None or len(unique) >= target_count:
        return unique

    expanded = list(unique)
    index = 1
    seen = {_candidate_key(candidate) for candidate in expanded}
    while len(expanded) < target_count:
        candidate = _expanded_lsq_candidate(config, index)
        index += 1
        if not candidate:
            break
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        expanded.append(candidate)
    return expanded


def _set_model_start_values(model, candidate: dict[str, float]) -> None:
    submodels = {
        getattr(submodel, "name", None): submodel
        for submodel in model
        if getattr(submodel, "name", None) is not None
    }
    for full_name, value in candidate.items():
        if full_name == "redshift":
            try:
                model["redshift"].z = 1.0 / (1.0 + float(value)) - 1.0
            except Exception:
                pass
            continue

        match = next(
            (
                (profile_name, field_name, submodels[profile_name])
                for profile_name in submodels
                for field_name in submodels[profile_name].meta.get(
                    "distributions", {}
                )
                if full_name == f"{profile_name}_{field_name}"
            ),
            None,
        )
        if match is None:
            continue
        _, field_name, submodel = match

        distributions = submodel.meta.get("distributions", {})
        param_value = float(value)
        if "log" in distributions.get(field_name, ""):
            param_value = float(np.log10(param_value))
        setattr(submodel, field_name, param_value)


def _weighted_objective(fit_mod, wave, flux, flux_err) -> float:
    safe_weights = np.where(
        np.isfinite(flux_err) & (flux_err > 0.0),
        1.0 / flux_err,
        0.0,
    )
    model_flux = np.asarray(fit_mod(wave), dtype=float)
    resid = (model_flux - np.asarray(flux, dtype=float)) * safe_weights
    if not np.all(np.isfinite(resid)):
        return float("inf")
    return float(np.sum(np.square(resid)))


def _strip_flux_outputs(sample_dict: dict) -> dict:
    return {k: v for k, v in sample_dict.items() if not k.endswith("_flux")}


def _add_radius_ratio_init_values(config: Config, params: dict) -> dict:
    params = dict(params)
    for profile in config.template.disk_profiles:
        inner_name = f"{profile.name}_inner_radius"
        outer_name = f"{profile.name}_outer_radius"
        ratio_name = f"{profile.name}_radius_ratio"
        if ratio_name in params or inner_name not in params or outer_name not in params:
            continue

        inner = max(float(params[inner_name]), 1e-30)
        outer = max(float(params[outer_name]), inner * DISK_RADIUS_RATIO_LOW)
        ratio = outer / inner
        params[ratio_name] = float(
            np.clip(
                ratio,
                DISK_RADIUS_RATIO_LOW + 1e-6,
                DISK_RADIUS_RATIO_HIGH - 1e-6,
            )
        )
    return params


def _move_init_params_inside_bounds(
    config: Config,
    params: dict,
    *,
    margin_frac: float = INIT_BOUNDARY_MARGIN_FRAC,
) -> dict:
    params = dict(params)
    for name, value in list(params.items()):
        low, high = _resolve_bounds(config, name)
        if low is None or high is None:
            continue
        if not np.isfinite(low) or not np.isfinite(high) or not high > low:
            continue
        margin = margin_frac * (high - low)
        params[name] = float(np.clip(float(value), low + margin, high - margin))

    for profile in config.template.disk_profiles:
        inner_name = f"{profile.name}_inner_radius"
        ratio_name = f"{profile.name}_radius_ratio"
        outer_name = f"{profile.name}_outer_radius"
        if inner_name in params and ratio_name in params:
            params[outer_name] = float(params[inner_name] * params[ratio_name])

    return params


def _feature_param_view(params: dict, config: Config | None = None) -> dict:
    view = dict(params)

    for name, value in list(params.items()):
        if name.endswith("_radius_ratio"):
            prefix = name.removesuffix("_radius_ratio")
            inner = params.get(f"{prefix}_inner_radius")
            if inner is not None and f"{prefix}_outer_radius" not in view:
                view[f"{prefix}_outer_radius"] = inner * value

    for name, value in list(params.items()):
        if name.endswith("_inclination_base"):
            prefix = name.removesuffix("_inclination_base")
            if f"{prefix}_inclination" not in view:
                view[f"{prefix}_inclination"] = jnp.arccos(value)

        if name.endswith("_apocenter_h_raw"):
            prefix = name.removesuffix("_apocenter_h_raw")
            z_h = params.get(name)
            z_k = params.get(f"{prefix}_apocenter_k_raw")
            if z_h is None or z_k is None:
                continue
            radius = jnp.sqrt(z_h**2 + z_k**2) + 1e-12
            scale = jnp.tanh(radius) / radius
            h = z_h * scale
            k = z_k * scale
            e_unit = jnp.sqrt(h**2 + k**2)
            low, high = (
                _resolve_bounds(config, f"{prefix}_eccentricity")
                if config is not None
                else (None, None)
            )
            if low is not None and high is not None:
                eccentricity = low + (high - low) * e_unit
            else:
                eccentricity = e_unit
            view.setdefault(f"{prefix}_eccentricity", eccentricity)
            view.setdefault(
                f"{prefix}_apocenter", jnp.mod(jnp.arctan2(h, k), 2 * jnp.pi)
            )

        if name.endswith("_apocenter_x_base"):
            prefix = name.removesuffix("_apocenter_x_base")
            x = params.get(name)
            y = params.get(f"{prefix}_apocenter_y_base")
            if x is None or y is None:
                continue
            view.setdefault(
                f"{prefix}_apocenter", jnp.mod(jnp.arctan2(y, x), 2 * jnp.pi)
            )

    return view


def _basin_feature_vector(
    params: dict,
    config: Config | None = None,
) -> np.ndarray:
    params = _feature_param_view(params, config)

    def val(name: str, default=np.nan):
        raw = params.get(name, default)
        try:
            return float(jax.device_get(raw))
        except Exception:
            return float(raw)

    features = []
    disk_prefixes = sorted(
        {
            name.removesuffix("_inner_radius")
            for name in params
            if name.endswith("_inner_radius")
        }
    )
    broad_prefixes = sorted(
        {
            name.removesuffix("_vel_width")
            for name in params
            if name.endswith("_vel_width") and "broad" in name
        }
    )

    for prefix in disk_prefixes:
        inner = max(val(f"{prefix}_inner_radius"), 1e-30)
        outer = max(val(f"{prefix}_outer_radius"), 1e-30)
        sigma = max(val(f"{prefix}_sigma"), 1e-30)
        inclination = val(f"{prefix}_inclination")
        eccentricity = val(f"{prefix}_eccentricity")
        apocenter = val(f"{prefix}_apocenter")
        features.extend(
            [
                np.log10(inner),
                np.log10(outer),
                np.cos(inclination),
                eccentricity * np.cos(apocenter),
                eccentricity * np.sin(apocenter),
                np.log10(sigma),
                val(f"{prefix}_q"),
            ]
        )

    for prefix in broad_prefixes:
        features.extend(
            [
                np.log10(max(val(f"{prefix}_vel_width"), 1e-30)),
                val(f"{prefix}_offset") / 1000.0,
                np.log10(max(val(f"{prefix}_area"), 1e-30)),
            ]
        )

    return np.asarray(features, dtype=float)


def _basin_distance(
    left: dict,
    right: dict,
    config: Config | None = None,
) -> float:
    left_features = _basin_feature_vector(left, config)
    right_features = _basin_feature_vector(right, config)
    n = min(left_features.size, right_features.size)
    if n == 0:
        return 0.0
    diff = left_features[:n] - right_features[:n]
    finite = np.isfinite(diff)
    if not np.any(finite):
        return 0.0
    return float(np.linalg.norm(diff[finite]))


def _dedupe_basin_candidates(
    candidates: list[dict],
    *,
    distance_threshold: float,
    params_key: str,
    config: Config | None = None,
) -> list[dict]:
    selected: list[dict] = []
    for candidate in candidates:
        params = candidate[params_key]
        if any(
            _basin_distance(params, kept[params_key], config) < distance_threshold
            for kept in selected
        ):
            continue
        selected.append(candidate)
    return selected


def _model_sample_site_names(model: BaseModel, config: Config) -> set[str]:
    model_trace = trace(seed(model, random.PRNGKey(0))).get_trace(
        wave=config.data.masked_wave,
        flux=config.data.masked_flux,
        flux_err=config.data.masked_flux_err,
    )
    return {
        name
        for name, site in model_trace.items()
        if site["type"] == "sample" and not site.get("is_observed", False)
    }


def _best_guide_sample_by_log_density(
    model: BaseModel,
    guide_samples: dict,
    config: Config,
):
    sample_site_names = _model_sample_site_names(model, config)
    candidate_names = [k for k in guide_samples if k in sample_site_names]

    if not candidate_names:
        raise ValueError("No guide sample parameters available for SVI initialization.")

    num_draws = min(200, int(guide_samples[candidate_names[0]].shape[0]))
    batch_params = {name: guide_samples[name][:num_draws] for name in candidate_names}

    model_args = ()
    model_kwargs = {
        "wave": config.data.masked_wave,
        "flux": config.data.masked_flux,
        "flux_err": config.data.masked_flux_err,
    }

    def score_one(params):
        logp, _ = log_density(model, model_args, model_kwargs, params)
        return logp

    all_logps = np.array(jax.device_get(jax.vmap(score_one)(batch_params)))

    finite_mask = np.isfinite(all_logps)
    if not np.any(finite_mask):
        raise ValueError("No finite SVI guide sample had a finite model log density.")

    best_idx = int(np.argmax(np.where(finite_mask, all_logps, -np.inf)))
    best_sample = {name: guide_samples[name][best_idx] for name in candidate_names}
    return best_sample, float(all_logps[best_idx]), best_idx


@flax.struct.dataclass
class BaseInitializer:
    debug_plot: bool = False

    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Call method must be implemented by subclasses.")


@flax.struct.dataclass
class DefaultInitializer(BaseInitializer):
    def __call__(self, config: Config = None, model: BaseModel = None):
        return {}, init_to_median(num_samples=1000)


@flax.struct.dataclass
class LSQInitializer(BaseInitializer):
    use_multistart: bool = True
    max_candidates: int = 8
    use_dogbox: bool = False

    def fit_candidates(self, config: Config = None, model: BaseModel = None):
        wave = config.data.masked_wave
        flux = config.data.masked_flux
        flux_err = config.data.masked_flux_err
        safe_weights = np.where(
            np.isfinite(flux_err) & (flux_err > 0), 1.0 / flux_err, 0.0
        )

        if not self.use_multistart:
            lsq_model = _compose_model(
                config.template,
                integrator=model.integrator,
                redshift=config.template.redshift.value,
            )
            fit_mod = TRFLSQFitter()(
                lsq_model,
                wave,
                flux,
                weights=safe_weights,
                maxiter=2_000,
                filter_non_finite=True,
            )
            obj = _weighted_objective(fit_mod, wave, flux, flux_err)
            init_params = _add_radius_ratio_init_values(
                config, format_posterior_samples(fit_mod, wave, flux, flux_err)
            )
            return [
                {
                    "fit_model": fit_mod,
                    "objective": obj,
                    "fitter": "trf",
                    "start_candidate": {},
                    "init_params": init_params,
                    "failed": False,
                }
            ]

        candidates = _structured_lsq_candidates(
            config, target_count=max(1, int(self.max_candidates))
        )[: self.max_candidates]
        fitters = [("trf", TRFLSQFitter())]
        if self.use_dogbox:
            fitters.append(("dogbox", DogBoxLSQFitter()))

        fit_candidates = []
        failures = 0
        for candidate in candidates:
            for fitter_name, fitter in fitters:
                lsq_model = _compose_model(
                    config.template,
                    integrator=model.integrator,
                    redshift=config.template.redshift.value,
                )
                _set_model_start_values(lsq_model, candidate)

                try:
                    trial_fit = fitter(
                        lsq_model,
                        wave,
                        flux,
                        weights=safe_weights,
                        maxiter=10_000,
                        filter_non_finite=True,
                    )
                    obj = _weighted_objective(trial_fit, wave, flux, flux_err)
                    if np.isfinite(obj):
                        init_params = _add_radius_ratio_init_values(
                            config,
                            format_posterior_samples(trial_fit, wave, flux, flux_err),
                        )
                        fit_candidates.append(
                            {
                                "fit_model": trial_fit,
                                "objective": obj,
                                "fitter": fitter_name,
                                "start_candidate": candidate,
                                "init_params": init_params,
                                "failed": False,
                            }
                        )
                except Exception:
                    failures += 1

        if not fit_candidates:
            raise RuntimeError(
                "All multi-start LSQ attempts failed "
                f"({len(candidates)} candidates, {len(fitters)} fitters)."
            )

        fit_candidates = sorted(fit_candidates, key=lambda item: item["objective"])
        logger.info(
            "Multi-start LSQ retained "
            f"{len(fit_candidates)} finite start(s) from "
            f"{len(candidates) * len(fitters)} attempts ({failures} failed). "
            f"Best objective={fit_candidates[0]['objective']:.4g}."
        )
        return fit_candidates

    def __call__(self, config: Config = None, model: BaseModel = None):
        fit_candidates = self.fit_candidates(config, model)
        fit_mod = fit_candidates[0]["fit_model"]
        init_params = fit_candidates[0].get("init_params")
        if init_params is None:
            init_params = _add_radius_ratio_init_values(
                config,
                format_posterior_samples(
                    fit_mod,
                    config.data.masked_wave,
                    config.data.masked_flux,
                    config.data.masked_flux_err,
                ),
            )

        init_strategy = init_to_value(values=init_params)

        if self.debug_plot:
            self.quick_plot(fit_mod, config, init_params)

        return init_params, init_strategy

    def quick_plot(self, fit_mod, config, init_params):
        disk_submodels = [
            fit_mod[sm_idx]
            for sm_idx in range(fit_mod.n_submodels)
            if fit_mod[sm_idx].name
            in [prof.name for prof in config.template.disk_profiles]
        ]

        if len(disk_submodels) > 0:
            disk_model = fit_mod["redshift"] | (np.sum(disk_submodels))
        else:
            disk_model = fit_mod["redshift"] | fit_mod["base"]

        narrow_line_submodels = [
            fit_mod[sm_idx]
            for sm_idx in range(fit_mod.n_submodels)
            if fit_mod[sm_idx].name
            in [
                prof.name
                for prof in config.template.line_profiles
                if "narrow" in prof.name
            ]
        ]

        if len(narrow_line_submodels) > 0:
            narrow_line_model = fit_mod["redshift"] | (np.sum(narrow_line_submodels))
        else:
            narrow_line_model = fit_mod["redshift"] | fit_mod["base"]

        broad_line_submodels = [
            fit_mod[sm_idx]
            for sm_idx in range(fit_mod.n_submodels)
            if fit_mod[sm_idx].name
            in [
                prof.name
                for prof in config.template.line_profiles
                if "broad" in prof.name
            ]
        ]

        if len(broad_line_submodels) > 0:
            broad_line_model = fit_mod["redshift"] | (np.sum(broad_line_submodels))
        else:
            broad_line_model = fit_mod["redshift"] | fit_mod["base"]

        n_sf = max(1, len(config.template.mask))

        fig, axes = plt.subplots(
            1,
            n_sf,
            layout="constrained",
            figsize=(10 * n_sf, 5),
        )

        tot_flux = fit_mod(config.data.masked_wave)
        disk_flux = disk_model(config.data.masked_wave)
        narrow_line_flux = narrow_line_model(config.data.masked_wave)
        broad_line_flux = broad_line_model(config.data.masked_wave)

        for i in range(n_sf):
            ax = axes.flat[i] if hasattr(axes, "flat") else axes
            mask_spec = None
            if i < len(config.template.mask):
                mask_spec = config.template.mask[i]
                mask = (config.data.masked_wave > mask_spec.lower_limit) & (
                    config.data.masked_wave < mask_spec.upper_limit
                )
            else:
                mask = np.ones_like(config.data.masked_wave, dtype=bool)

            wave = config.data.masked_wave[mask]
            flux = config.data.masked_flux[mask]
            flux_err = config.data.masked_flux_err[mask]

            ax.errorbar(
                wave / (1 + init_params["redshift"]),
                flux,
                yerr=flux_err,
                fmt="o",
                color="grey",
                zorder=-10,
                alpha=0.25,
            )

            ax.plot(
                wave / (1 + init_params["redshift"]),
                tot_flux[mask],
                label="LSQ Fit",
                color="C3",
            )
            ax.plot(
                wave / (1 + init_params["redshift"]),
                disk_flux[mask],
                label="Disk Fit",
                color="C4",
            )
            ax.plot(
                wave / (1 + init_params["redshift"]),
                narrow_line_flux[mask],
                label="Narrow Line Fit",
                color="C5",
                linestyle="--",
            )
            ax.plot(
                wave / (1 + init_params["redshift"]),
                broad_line_flux[mask],
                label="Broad Line Fit",
                color="C6",
                linestyle="--",
            )
            if mask_spec is not None:
                ax.set_title(
                    f"{mask_spec.lower_limit / (1 + init_params['redshift']):.0f}"
                    f"–{mask_spec.upper_limit / (1 + init_params['redshift']):.0f} Å"
                )

        # Print the initial parameter estimates on the figure
        # param_text = "\n".join(
        #     [
        #         f"{k:25} {v:.5g}"
        #         for k, v in init_params.items()
        #         if np.isfinite(v) and "_base" not in k
        #     ]
        # )
        # ax.text(
        #     0.05,
        #     0.95,
        #     param_text,
        #     transform=ax.transAxes,
        #     fontsize=6,
        #     verticalalignment="top",
        #     fontfamily="monospace",
        # )
        #
        # ax.set_title(f"LSQ Initial Fit for {config.template.name}")
        # ax.set_xlabel("Wavelength (Å)")
        # ax.set_ylabel("Flux (mJy)")

        ax.legend()
        fig.savefig(f"{config.output_path}/lsq_fit.png")
        plt.close(fig)


@flax.struct.dataclass
class SVIInitializer(BaseInitializer):
    lsq_candidates: int = 1
    svi_candidates: int = 1
    candidate_distance_threshold: float = 0.25
    max_candidate_loss_relative_std: float = 0.10
    num_steps: int = 2_000
    learning_rate: float = 5e-3
    decay_rate: float = 0.1
    decay_steps: int = 2_000  # ~num_steps
    progress_bar: bool = False
    num_samples: int = 1000

    def __call__(self, config: Config, model: BaseModel):
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)

        lsq_initializer = LSQInitializer(
            debug_plot=self.debug_plot,
            use_multistart=self.lsq_candidates > 1,
            max_candidates=max(1, int(self.lsq_candidates)),
        )
        lsq_candidates = lsq_initializer.fit_candidates(config, model)
        distinct_lsq_candidates = _dedupe_basin_candidates(
            lsq_candidates,
            distance_threshold=self.candidate_distance_threshold,
            params_key="init_params",
            config=config,
        )
        svi_candidate_count = max(
            1, min(int(self.svi_candidates), len(distinct_lsq_candidates))
        )
        candidates_to_run = distinct_lsq_candidates[:svi_candidate_count]

        logger.info(
            "Running SVI initialization from "
            f"{len(candidates_to_run)} distinct LSQ basin(s) "
            f"({len(lsq_candidates)} finite LSQ fit(s))."
        )

        results = []
        for candidate_idx, lsq_candidate in enumerate(candidates_to_run):
            rng_key, svi_key, sample_key = random.split(rng_key, 3)
            try:
                result = self._run_svi_candidate(
                    candidate_idx,
                    lsq_candidate,
                    svi_key,
                    sample_key,
                    config,
                    model,
                )
                results.append(result)
            except Exception as exc:
                logger.warning(
                    f"SVI candidate {candidate_idx} failed; falling back is still possible: {exc}"
                )

        if not results:
            logger.warning("All SVI candidates failed; falling back to best LSQ basin.")
            init_params = lsq_candidates[0]["init_params"]
            diagnose_init_params(model, init_params, config)
            return init_params, init_to_value(values=init_params)

        ranked_results = self._rank_candidates(results)
        distinct_results = _dedupe_basin_candidates(
            ranked_results,
            distance_threshold=self.candidate_distance_threshold,
            params_key="init_params",
            config=config,
        )
        selected = distinct_results[0]
        init_params = selected["init_params"]
        init_strategy = init_to_value(values=init_params)

        self._write_candidate_summary(config, ranked_results, selected)
        diagnose_init_params(model, init_params, config)

        logger.info(
            "Selected SVI candidate "
            f"{selected['candidate_id']} with log_density={selected['score']:.4g}, "
            f"selection_score={selected['selection_score']:.4g}, "
            f"final_loss={selected['final_loss']:.4g}, "
            f"loss_relative_std={selected['loss_relative_std']:.4g}, "
            f"lsq_objective={selected['lsq_objective']:.4g}."
        )

        if self.debug_plot:
            self.quick_plot(
                selected["svi_result"],
                selected["guide"],
                model,
                init_params,
                config,
                ignored={
                    k: True
                    for k, v in init_params.items()
                    if k in [p.name for p in config.template.iter_shared]
                },
            )

        return init_params, init_strategy

    def _rank_candidates(self, results: list[dict]) -> list[dict]:
        ranked = []
        for result in results:
            loss_relative_std = result["loss_relative_std"]
            score = result["score"]
            eligible = (
                np.isfinite(score)
                and np.isfinite(loss_relative_std)
                and loss_relative_std <= self.max_candidate_loss_relative_std
            )
            result = dict(result)
            result["selection_eligible"] = bool(eligible)
            result["selection_rejection_reason"] = (
                ""
                if eligible
                else (
                    "loss_relative_std"
                    if np.isfinite(score)
                    else "nonfinite_log_density"
                )
            )
            result["selection_score"] = float(score if eligible else -np.inf)
            ranked.append(result)

        if not any(result["selection_eligible"] for result in ranked):
            logger.warning(
                "No SVI candidates passed the loss stability threshold "
                f"({self.max_candidate_loss_relative_std:.4g}); selecting by "
                "log density anyway."
            )
            for result in ranked:
                result["selection_score"] = result["score"]
                result["selection_rejection_reason"] = "fallback_no_eligible_candidates"

        ranked = sorted(
            ranked,
            key=lambda item: (
                item["selection_score"],
                -item["loss_relative_std"],
                -item["lsq_objective"],
            ),
            reverse=True,
        )
        for rank, result in enumerate(ranked):
            result["selection_rank"] = rank
        return ranked

    def _run_svi_candidate(
        self,
        candidate_idx: int,
        lsq_candidate: dict,
        svi_key,
        sample_key,
        config: Config,
        model: BaseModel,
    ) -> dict:
        lsq_params = lsq_candidate["init_params"]
        init_strategy = init_to_value(values=lsq_params)

        guide = AutoMultivariateNormal(
            model,
            init_loc_fn=init_strategy,
        )

        schedule = optax.exponential_decay(
            self.learning_rate, self.decay_steps, self.decay_rate
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=schedule),
        )

        svi = SVI(model, guide, optimizer, Trace_ELBO())

        svi_result = svi.run(
            svi_key,
            num_steps=self.num_steps,
            wave=config.data.masked_wave,
            flux=config.data.masked_flux,
            flux_err=config.data.masked_flux_err,
            progress_bar=self.progress_bar,
        )

        recent_losses = svi_result.losses[-1000:]
        relative_std = jnp.nanstd(recent_losses) / jnp.abs(jnp.nanmean(recent_losses))

        if relative_std > 0.01:
            logger.warning(
                f"SVI candidate {candidate_idx} may not have converged! "
                f"Relative std: {relative_std:.4f}"
            )
        else:
            logger.info(
                f"SVI candidate {candidate_idx} converged. "
                f"Final loss: {svi_result.losses[-1]:.4f}"
            )

        # Use the best coherent guide sample under the actual model log density;
        # componentwise guide medians can land in low-density regions when
        # parameters are correlated.
        guide_samples = guide.sample_posterior(
            sample_key,
            svi_result.params,
            sample_shape=(self.num_samples,),
        )

        try:
            init_params, score, best_idx = _best_guide_sample_by_log_density(
                model,
                guide_samples,
                config,
            )
            logger.info(
                f"SVI candidate {candidate_idx}: using guide sample {best_idx} "
                f"with model log density {score:.4g}."
            )
        except Exception as exc:
            logger.warning(
                f"SVI candidate {candidate_idx}: falling back to LSQ initialization "
                f"because guide sample scoring failed: {exc}"
            )
            init_params = lsq_params
            score = float("-inf")

        init_params = _move_init_params_inside_bounds(config, init_params)

        return {
            "candidate_id": candidate_idx,
            "init_params": init_params,
            "score": float(score),
            "final_loss": float(jax.device_get(svi_result.losses[-1])),
            "loss_relative_std": float(jax.device_get(relative_std)),
            "lsq_objective": float(lsq_candidate["objective"]),
            "lsq_fitter": lsq_candidate["fitter"],
            "guide": guide,
            "svi_result": svi_result,
        }

    def _write_candidate_summary(
        self,
        config: Config,
        results: list[dict],
        selected: dict,
    ) -> None:
        output_path = Path(config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / "initialization_candidates.csv"
        rows = []
        for result in results:
            row = {
                "candidate_id": result["candidate_id"],
                "selected": result["candidate_id"] == selected["candidate_id"],
                "selection_rank": result.get("selection_rank", ""),
                "selection_eligible": result.get("selection_eligible", ""),
                "selection_score": result.get("selection_score", ""),
                "selection_rejection_reason": result.get(
                    "selection_rejection_reason", ""
                ),
                "log_density": result["score"],
                "final_loss": result["final_loss"],
                "loss_relative_std": result["loss_relative_std"],
                "lsq_objective": result["lsq_objective"],
                "lsq_fitter": result["lsq_fitter"],
                "distance_to_selected": _basin_distance(
                    result["init_params"], selected["init_params"], config
                ),
            }
            rows.append(row)

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def quick_plot(self, svi_result, guide, model, init_params, config, ignored={}):
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        guide_samples = guide.sample_posterior(
            rng_key,
            svi_result.params,
            sample_shape=(1000,),
        )

        plot_samples = {
            k: v
            for k, v in guide_samples.items()
            if not k.endswith("_flux")
            and not k.endswith("_base")
            and not k.endswith("_raw")
            and np.min(v) != np.max(v)
        }
        try:
            best_sample, _, _ = _best_guide_sample_by_log_density(
                model,
                guide_samples,
                config,
            )
        except Exception:
            best_sample = _strip_flux_outputs(
                {k: jnp.median(v, axis=0) for k, v in guide_samples.items()}
            )

        guide_mods = {
            k: best_sample[k] if k in best_sample else jnp.median(v, axis=0)
            for k, v in guide_samples.items()
        }
        redshift = float(
            jax.device_get(init_params.get("redshift", config.template.redshift.value))
        )

        plot_keys = [k for k in plot_samples if k not in ignored]
        axes_scale = []
        for k in plot_keys:
            param_ref = next((p for p in config.template.iter_all if p.name == k), None)
            axes_scale.append(
                "log"
                if param_ref is not None and "log" in param_ref.param.distribution.value
                else "linear"
            )

        fig = corner.corner(
            np.array([plot_samples[k] for k in plot_keys]).T,
            labels=plot_keys,
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
            axes_scale=axes_scale,
        )
        fig.savefig(f"{config.output_path}/guide_corner_plot.png")

        n_sf = max(1, len(config.template.mask))

        fig, axes = plt.subplots(
            1,
            n_sf,
            layout="constrained",
            figsize=(10 * n_sf, 5),
        )

        for i in range(n_sf):
            ax = axes.flat[i] if hasattr(axes, "flat") else axes
            mask_spec = None
            if i < len(config.template.mask):
                mask_spec = config.template.mask[i]
                mask = (config.data.masked_wave > mask_spec.lower_limit) & (
                    config.data.masked_wave < mask_spec.upper_limit
                )
            else:
                mask = np.ones_like(config.data.masked_wave, dtype=bool)

            wave = config.data.masked_wave[mask]
            flux = config.data.masked_flux[mask]
            flux_err = config.data.masked_flux_err[mask]

            disk_flux = guide_mods["disk_flux"][mask]
            line_flux = guide_mods["line_flux"][mask]
            tot_flux = disk_flux + line_flux

            ax.errorbar(
                wave / (1 + redshift),
                flux,
                yerr=flux_err,
                fmt="o",
                color="grey",
                zorder=-10,
                alpha=0.25,
            )

            ax.plot(
                wave / (1 + redshift),
                tot_flux,
                label="SVI Fit",
                color="C3",
            )
            ax.plot(
                wave / (1 + redshift),
                disk_flux,
                label="Disk Fit",
                color="C4",
            )
            ax.plot(
                wave / (1 + redshift),
                line_flux,
                label="Line Fit",
                color="C5",
                linestyle="--",
            )
            if mask_spec is not None:
                ax.set_title(
                    f"{mask_spec.lower_limit / (1 + redshift):.0f}"
                    f"–{mask_spec.upper_limit / (1 + redshift):.0f} Å"
                )
            ax.legend()

        fig.savefig(f"{config.output_path}/svi_fit.png")
        plt.close(fig)


@flax.struct.dataclass
class PathfinderInitializer(BaseInitializer):
    lsq_candidates: int = 8
    pathfinder_candidates: int = 8
    candidate_distance_threshold: float = 0.25
    num_samples: int = 32
    score_batch_size: int = 8
    maxiter: int = 30
    maxcor: int = 10
    maxls: int = 1000
    gtol: float = 1e-8
    ftol: float = 1e-5

    def __call__(self, config: Config, model: BaseModel):
        try:
            from blackjax.vi import pathfinder
        except ImportError as exc:
            raise RuntimeError(
                "Pathfinder initialization requires BlackJAX. Install BlackJAX "
                "before using --init-method=pathfinder."
            ) from exc

        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, model_key, pathfinder_key = random.split(rng_key, 3)

        lsq_initializer = LSQInitializer(
            debug_plot=self.debug_plot,
            use_multistart=self.lsq_candidates > 1,
            max_candidates=max(1, int(self.lsq_candidates)),
        )
        lsq_candidates = lsq_initializer.fit_candidates(config, model)
        distinct_lsq_candidates = _dedupe_basin_candidates(
            lsq_candidates,
            distance_threshold=self.candidate_distance_threshold,
            params_key="init_params",
            config=config,
        )
        candidates_to_run = distinct_lsq_candidates[
            : max(1, min(int(self.pathfinder_candidates), len(distinct_lsq_candidates)))
        ]

        logger.info(
            "Running multipathfinder initialization from "
            f"{len(candidates_to_run)} distinct LSQ basin(s) "
            f"({len(lsq_candidates)} finite LSQ fit(s))."
        )

        model_kwargs = {
            "wave": config.data.masked_wave,
            "flux": config.data.masked_flux,
            "flux_err": config.data.masked_flux_err,
        }
        initial_positions, usable_candidates, potential_fn, postprocess_fn = (
            self._make_initial_positions(
                model_key, candidates_to_run, model, model_kwargs
            )
        )

        def logdensity_fn(z):
            return -potential_fn(z)

        path_keys = random.split(pathfinder_key, len(initial_positions))
        results = []
        for candidate_idx, (key, initial_position, candidate) in enumerate(
            zip(path_keys, initial_positions, usable_candidates)
        ):
            try:
                result = self._run_pathfinder_candidate(
                    candidate_idx,
                    key,
                    initial_position,
                    candidate,
                    pathfinder,
                    logdensity_fn,
                    postprocess_fn,
                    config,
                )
                results.append(result)
            except Exception as exc:
                logger.warning(f"Pathfinder candidate {candidate_idx} failed: {exc}")

        if not results:
            raise RuntimeError("All Pathfinder candidates failed.")

        ranked_results = self._rank_candidates(results)
        distinct_results = _dedupe_basin_candidates(
            ranked_results,
            distance_threshold=self.candidate_distance_threshold,
            params_key="init_params",
            config=config,
        )
        selected = distinct_results[0]
        init_params = selected["init_params"]

        self._write_candidate_summary(config, ranked_results, selected)
        diagnose_init_params(model, init_params, config)

        logger.info(
            "Selected Pathfinder candidate "
            f"{selected['candidate_id']} with log_density={selected['score']:.4g}, "
            f"path_max_log_density={selected['path_max_log_density']:.4g}, "
            f"path_mean_log_density={selected['path_mean_log_density']:.4g}, "
            f"lsq_objective={selected['lsq_objective']:.4g}."
        )

        return init_params, init_to_value(values=init_params)

    def _make_initial_positions(
        self,
        rng_key,
        candidates: list[dict],
        model: BaseModel,
        model_kwargs: dict,
    ):
        positions = []
        usable_candidates = []
        potential_fn = None
        postprocess_fn = None
        keys = random.split(rng_key, len(candidates))

        for key, candidate in zip(keys, candidates):
            try:
                param_info, this_potential_fn, this_postprocess_fn, _ = (
                    initialize_model(
                        key,
                        model,
                        model_kwargs=model_kwargs,
                        init_strategy=init_to_value(values=candidate["init_params"]),
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Skipping LSQ candidate during Pathfinder initialization "
                    f"because unconstrained initialization failed: {exc}"
                )
                continue

            positions.append(param_info.z)
            usable_candidates.append(candidate)
            potential_fn = this_potential_fn
            postprocess_fn = this_postprocess_fn

        if not positions:
            raise RuntimeError("No LSQ candidate could initialize Pathfinder.")

        return positions, usable_candidates, potential_fn, postprocess_fn

    def _run_pathfinder_candidate(
        self,
        candidate_idx: int,
        rng_key,
        initial_position,
        candidate: dict,
        pathfinder,
        logdensity_fn,
        postprocess_fn,
        config: Config,
    ) -> dict:
        approx_key, sample_key = random.split(rng_key)
        state, _ = pathfinder.approximate(
            approx_key,
            logdensity_fn,
            initial_position,
            num_samples=max(1, int(self.num_samples)),
            maxiter=max(1, int(self.maxiter)),
            maxcor=max(1, int(self.maxcor)),
            maxls=max(1, int(self.maxls)),
            gtol=float(self.gtol),
            ftol=float(self.ftol),
        )
        samples, _ = pathfinder.sample(
            sample_key,
            state,
            num_samples=max(1, int(self.num_samples)),
        )
        logp = self._score_samples(logdensity_fn, samples)
        finite = np.isfinite(logp)
        if not np.any(finite):
            raise RuntimeError("no finite Pathfinder sample log densities")

        best_sample_idx = int(np.argmax(np.where(finite, logp, -np.inf)))
        init_params = jax.tree.map(lambda x: x[best_sample_idx], samples)
        init_params = postprocess_fn(init_params)
        init_params = _move_init_params_inside_bounds(config, init_params)

        logger.info(
            f"Pathfinder candidate {candidate_idx}: best log_density="
            f"{logp[best_sample_idx]:.4g}, mean log_density={np.nanmean(logp[finite]):.4g}."
        )

        return {
            "candidate_id": candidate_idx,
            "init_params": _strip_flux_outputs(init_params),
            "score": float(logp[best_sample_idx]),
            "path_max_log_density": float(np.nanmax(logp)),
            "path_mean_log_density": float(np.nanmean(logp[finite])),
            "path_median_log_density": float(np.nanmedian(logp[finite])),
            "path_finite_fraction": float(np.mean(finite)),
            "best_sample_idx": best_sample_idx,
            "lsq_objective": float(candidate["objective"]),
            "lsq_fitter": candidate["fitter"],
        }

    def _score_samples(self, logdensity_fn, samples) -> np.ndarray:
        sample_count = int(jax.tree.leaves(samples)[0].shape[0])
        batch_size = max(1, int(self.score_batch_size))
        values = []
        for start in range(0, sample_count, batch_size):
            stop = min(sample_count, start + batch_size)
            batch = jax.tree.map(lambda x: x[start:stop], samples)
            values.append(jax.device_get(jax.vmap(logdensity_fn)(batch)))
        return np.asarray(np.concatenate(values), dtype=float)

    def _rank_candidates(self, results: list[dict]) -> list[dict]:
        ranked = []
        for result in results:
            result = dict(result)
            finite = np.isfinite(result["score"])
            result["selection_eligible"] = bool(
                finite and result["path_finite_fraction"] > 0.0
            )
            result["selection_rejection_reason"] = (
                "" if result["selection_eligible"] else "nonfinite_log_density"
            )
            result["selection_score"] = (
                float(result["score"]) if result["selection_eligible"] else -np.inf
            )
            ranked.append(result)

        ranked = sorted(
            ranked,
            key=lambda item: (
                item["selection_score"],
                item["path_mean_log_density"],
                -item["lsq_objective"],
            ),
            reverse=True,
        )
        for rank, result in enumerate(ranked):
            result["selection_rank"] = rank
        return ranked

    def _write_candidate_summary(
        self,
        config: Config,
        results: list[dict],
        selected: dict,
    ) -> None:
        output_path = Path(config.output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        path = output_path / "initialization_candidates.csv"
        rows = []
        for result in results:
            row = {
                "candidate_id": result["candidate_id"],
                "selected": result["candidate_id"] == selected["candidate_id"],
                "selection_rank": result.get("selection_rank", ""),
                "selection_eligible": result.get("selection_eligible", ""),
                "selection_score": result.get("selection_score", ""),
                "selection_rejection_reason": result.get(
                    "selection_rejection_reason", ""
                ),
                "log_density": result["score"],
                "path_max_log_density": result["path_max_log_density"],
                "path_mean_log_density": result["path_mean_log_density"],
                "path_median_log_density": result["path_median_log_density"],
                "path_finite_fraction": result["path_finite_fraction"],
                "best_sample_idx": result["best_sample_idx"],
                "lsq_objective": result["lsq_objective"],
                "lsq_fitter": result["lsq_fitter"],
                "distance_to_selected": _basin_distance(
                    result["init_params"], selected["init_params"], config
                ),
            }
            rows.append(row)

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
