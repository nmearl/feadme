"""Benchmark FEADME disk integration choices.

This script compares the built-in disk integration families over a grid of
Xi/radius and phi resolutions. It evaluates only the disk profile, with matched
flux and normalization integrators, so the results isolate the expensive part of
the model used during sampling.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd
from quadax import ClenshawCurtisRule

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from feadme.core.disk import integrand, normalization_integrand
from feadme.core.evaluators import _compute_disk_flux_vectorized

HALPHA_REST = 6564.61
REST_MASK = (6200.0, 7000.0)
LN10 = float(np.log(10.0))


DEFAULT_CASES: dict[str, dict[str, float]] = {
    "moderate": {
        "redshift": 0.08,
        "halpha_disk_area": 45.0,
        "halpha_disk_inner_radius": 900.0,
        "halpha_disk_radius_ratio": 2600.0 / 900.0,
        "halpha_disk_sigma": 850.0,
        "halpha_disk_inclination": 0.55,
        "halpha_disk_q": 2.7,
        "halpha_disk_eccentricity": 0.25,
        "halpha_disk_apocenter": 2.3,
        "halpha_disk_offset": 0.0,
        "halpha_disk_baseline": 0.0,
    },
    "fragile": {
        "redshift": 0.070715,
        "halpha_disk_area": 58.14,
        "halpha_disk_inner_radius": 275.5,
        "halpha_disk_radius_ratio": 1142.0 / 275.5,
        "halpha_disk_sigma": 305.0,
        "halpha_disk_inclination": 0.21,
        "halpha_disk_q": 1.52,
        "halpha_disk_eccentricity": 0.42,
        "halpha_disk_apocenter": 0.65,
        "halpha_disk_offset": 0.0,
        "halpha_disk_baseline": 0.0,
    },
    "high_inclination": {
        "redshift": 0.06,
        "halpha_disk_area": 35.0,
        "halpha_disk_inner_radius": 800.0,
        "halpha_disk_radius_ratio": 2600.0 / 800.0,
        "halpha_disk_sigma": 500.0,
        "halpha_disk_inclination": 1.05,
        "halpha_disk_q": 2.4,
        "halpha_disk_eccentricity": 0.2,
        "halpha_disk_apocenter": 1.4,
        "halpha_disk_offset": 0.0,
        "halpha_disk_baseline": 0.0,
    },
    "narrow_sigma": {
        "redshift": 0.09,
        "halpha_disk_area": 40.0,
        "halpha_disk_inner_radius": 850.0,
        "halpha_disk_radius_ratio": 2400.0 / 850.0,
        "halpha_disk_sigma": 120.0,
        "halpha_disk_inclination": 0.5,
        "halpha_disk_q": 2.6,
        "halpha_disk_eccentricity": 0.35,
        "halpha_disk_apocenter": 2.8,
        "halpha_disk_offset": 0.0,
        "halpha_disk_baseline": 0.0,
    },
    "high_eccentricity": {
        "redshift": 0.08,
        "halpha_disk_area": 42.0,
        "halpha_disk_inner_radius": 700.0,
        "halpha_disk_radius_ratio": 2200.0 / 700.0,
        "halpha_disk_sigma": 450.0,
        "halpha_disk_inclination": 0.45,
        "halpha_disk_q": 2.1,
        "halpha_disk_eccentricity": 0.72,
        "halpha_disk_apocenter": 3.2,
        "halpha_disk_offset": 0.0,
        "halpha_disk_baseline": 0.0,
    },
    "large_extent": {
        "redshift": 0.08,
        "halpha_disk_area": 35.0,
        "halpha_disk_inner_radius": 600.0,
        "halpha_disk_radius_ratio": 18.0,
        "halpha_disk_sigma": 750.0,
        "halpha_disk_inclination": 0.45,
        "halpha_disk_q": 2.8,
        "halpha_disk_eccentricity": 0.35,
        "halpha_disk_apocenter": 2.2,
        "halpha_disk_offset": 0.0,
        "halpha_disk_baseline": 0.0,
    },
}


@dataclass(frozen=True)
class IntegratorPair:
    label: str
    method: str
    xi_param: int
    phi_param: int
    split_param: int | None
    integrator: Callable
    normalization_integrator: Callable


def make_quad_pair(xi_order: int, phi_order: int) -> IntegratorPair:
    fixed_quad_xi = ClenshawCurtisRule(order=xi_order).integrate
    fixed_quad_phi = ClenshawCurtisRule(order=phi_order).integrate

    @partial(jax.jit, static_argnums=(2, 3))
    def vector_integrator(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0):
        log_xi1 = jnp.log10(xi1)
        log_xi2 = jnp.log10(xi2)

        def integrate_over_phi(log_xi):
            xi = 10.0**log_xi
            values = fixed_quad_phi(
                lambda phi: integrand(phi, xi, X, inc, sigma, q, e, phi0),
                phi1,
                phi2,
                args=(),
            )[0]
            return values * (xi * LN10)

        return fixed_quad_xi(integrate_over_phi, log_xi1, log_xi2, args=())[0]

    @partial(jax.jit, static_argnums=(2, 3))
    def scalar_integrator(xi1, xi2, phi1, phi2, inc, sigma, q, e, phi0):
        log_xi1 = jnp.log10(xi1)
        log_xi2 = jnp.log10(xi2)

        def integrate_over_phi(log_xi):
            xi = 10.0**log_xi
            value = fixed_quad_phi(
                lambda phi: normalization_integrand(
                    phi, xi, inc, sigma, q, e, phi0
                ),
                phi1,
                phi2,
                args=(),
            )[0]
            return value * (xi * LN10)

        return fixed_quad_xi(integrate_over_phi, log_xi1, log_xi2, args=())[0]

    return IntegratorPair(
        label=f"quad_xi{xi_order}_phi{phi_order}",
        method="quad",
        xi_param=xi_order,
        phi_param=phi_order,
        split_param=None,
        integrator=vector_integrator,
        normalization_integrator=scalar_integrator,
    )


def make_mixed_pair(xi_order: int, phi_bins: int) -> IntegratorPair:
    fixed_quad_xi = ClenshawCurtisRule(order=xi_order).integrate

    @partial(jax.jit, static_argnums=(2, 3))
    def vector_integrator(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0):
        dphi = (phi2 - phi1) / phi_bins
        phi = phi1 + (jnp.arange(phi_bins) + 0.5) * dphi
        log_xi1 = jnp.log10(xi1)
        log_xi2 = jnp.log10(xi2)

        def integrate_over_log_xi(log_xi):
            xi = 10.0**log_xi
            values = integrand(
                phi[:, None],
                xi,
                jnp.asarray(X)[None, :],
                inc,
                sigma,
                q,
                e,
                phi0,
            )
            return jnp.sum(values * dphi, axis=0) * (xi * LN10)

        return fixed_quad_xi(integrate_over_log_xi, log_xi1, log_xi2, args=())[0]

    @partial(jax.jit, static_argnums=(2, 3))
    def scalar_integrator(xi1, xi2, phi1, phi2, inc, sigma, q, e, phi0):
        dphi = (phi2 - phi1) / phi_bins
        phi = phi1 + (jnp.arange(phi_bins) + 0.5) * dphi
        log_xi1 = jnp.log10(xi1)
        log_xi2 = jnp.log10(xi2)

        def integrate_over_log_xi(log_xi):
            xi = 10.0**log_xi
            values = normalization_integrand(phi, xi, inc, sigma, q, e, phi0)
            return jnp.sum(values * dphi, axis=0) * (xi * LN10)

        return fixed_quad_xi(integrate_over_log_xi, log_xi1, log_xi2, args=())[0]

    return IntegratorPair(
        label=f"mixed_xi{xi_order}_phi{phi_bins}",
        method="mixed",
        xi_param=xi_order,
        phi_param=phi_bins,
        split_param=None,
        integrator=vector_integrator,
        normalization_integrator=scalar_integrator,
    )


def make_trap_pair(xi_bins: int, phi_bins: int) -> IntegratorPair:
    phi_grid = jnp.linspace(0.0, 2.0 * jnp.pi, phi_bins + 1)
    phi_midpoints = 0.5 * (phi_grid[:-1] + phi_grid[1:])
    dphi = phi_grid[1:] - phi_grid[:-1]

    @partial(jax.jit, static_argnums=(2, 3))
    def vector_integrator(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0):
        xi_log = jnp.linspace(jnp.log10(xi1), jnp.log10(xi2), xi_bins)
        xi = 10.0**xi_log
        values = integrand(
            phi_midpoints[None, :, None],
            xi[:, None, None],
            jnp.asarray(X)[None, None, :],
            inc,
            sigma,
            q,
            e,
            phi0,
        )
        values = values * xi[:, None, None] * LN10
        phi_integrated = jnp.sum(values * dphi[None, :, None], axis=1)
        return jnp.trapezoid(phi_integrated, xi_log, axis=0)

    @partial(jax.jit, static_argnums=(2, 3))
    def scalar_integrator(xi1, xi2, phi1, phi2, inc, sigma, q, e, phi0):
        xi_log = jnp.linspace(jnp.log10(xi1), jnp.log10(xi2), xi_bins)
        xi = 10.0**xi_log
        values = normalization_integrand(
            phi_midpoints[None, :], xi[:, None], inc, sigma, q, e, phi0
        )
        values = values * xi[:, None] * LN10
        phi_integrated = jnp.sum(values * dphi[None, :], axis=1)
        return jnp.trapezoid(phi_integrated, xi_log)

    return IntegratorPair(
        label=f"trap_xi{xi_bins}_phi{phi_bins}",
        method="trap",
        xi_param=xi_bins,
        phi_param=phi_bins,
        split_param=None,
        integrator=vector_integrator,
        normalization_integrator=scalar_integrator,
    )


def make_split_quad_pair(xi_order: int, phi_order: int, n_split: int) -> IntegratorPair:
    fixed_quad_xi = ClenshawCurtisRule(order=xi_order).integrate
    fixed_quad_phi = ClenshawCurtisRule(order=phi_order).integrate

    @partial(jax.jit, static_argnums=(2, 3))
    def vector_integrator(xi1, xi2, phi1, phi2, X, inc, sigma, q, e, phi0):
        dphi = (phi2 - phi1) / n_split

        def integrate_over_phi(log_xi):
            xi = 10.0**log_xi
            total = jnp.zeros_like(X)

            def body(i, acc):
                lo = phi1 + i * dphi
                hi = lo + dphi
                values = fixed_quad_phi(
                    lambda phi: integrand(phi, xi, X, inc, sigma, q, e, phi0),
                    lo,
                    hi,
                    args=(),
                )[0]
                return acc + values

            total = jax.lax.fori_loop(0, n_split, body, total)
            return total * (xi * LN10)

        return fixed_quad_xi(
            integrate_over_phi, jnp.log10(xi1), jnp.log10(xi2), args=()
        )[0]

    @partial(jax.jit, static_argnums=(2, 3))
    def scalar_integrator(xi1, xi2, phi1, phi2, inc, sigma, q, e, phi0):
        dphi = (phi2 - phi1) / n_split

        def integrate_over_phi(log_xi):
            xi = 10.0**log_xi
            total = jnp.asarray(0.0)

            def body(i, acc):
                lo = phi1 + i * dphi
                hi = lo + dphi
                value = fixed_quad_phi(
                    lambda phi: normalization_integrand(
                        phi, xi, inc, sigma, q, e, phi0
                    ),
                    lo,
                    hi,
                    args=(),
                )[0]
                return acc + value

            total = jax.lax.fori_loop(0, n_split, body, total)
            return total * (xi * LN10)

        return fixed_quad_xi(
            integrate_over_phi, jnp.log10(xi1), jnp.log10(xi2), args=()
        )[0]

    return IntegratorPair(
        label=f"split_quad_xi{xi_order}_phi{phi_order}_split{n_split}",
        method="split_quad",
        xi_param=xi_order,
        phi_param=phi_order,
        split_param=n_split,
        integrator=vector_integrator,
        normalization_integrator=scalar_integrator,
    )


def parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_methods(value: str) -> list[str]:
    allowed = {"quad", "mixed", "trap", "split_quad"}
    methods = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(methods) - allowed)
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown method(s): {', '.join(unknown)}")
    return methods


def load_truth_json(path: Path) -> tuple[str, dict[str, float]]:
    with path.open() as handle:
        payload = json.load(handle)

    if "truth" in payload:
        payload = payload["truth"]

    values = {key: float(value) for key, value in payload.items()}
    return path.stem, values


def disk_outer_radius(truth: dict[str, float]) -> float:
    if "halpha_disk_outer_radius" in truth:
        return truth["halpha_disk_outer_radius"]
    return truth["halpha_disk_inner_radius"] * truth["halpha_disk_radius_ratio"]


def observed_wave_grid(truth: dict[str, float], n_wave: int) -> np.ndarray:
    z = truth["redshift"]
    return np.linspace(REST_MASK[0] * (1.0 + z), REST_MASK[1] * (1.0 + z), n_wave)


def evaluate_disk_profile(
    wave: np.ndarray | jax.Array,
    truth: dict[str, float],
    pair: IntegratorPair,
) -> jax.Array:
    z = truth["redshift"]
    center = jnp.asarray([HALPHA_REST * (1.0 + z)])
    inner_radius = jnp.asarray([truth["halpha_disk_inner_radius"]])
    outer_radius = jnp.asarray([disk_outer_radius(truth)])
    sigma = jnp.asarray([truth["halpha_disk_sigma"]])
    inclination = jnp.asarray([truth["halpha_disk_inclination"]])
    q = jnp.asarray([truth["halpha_disk_q"]])
    eccentricity = jnp.asarray([truth["halpha_disk_eccentricity"]])
    apocenter = jnp.asarray([truth["halpha_disk_apocenter"]])
    area = jnp.asarray([truth["halpha_disk_area"]])
    offset = jnp.asarray([truth.get("halpha_disk_offset", 0.0)])
    baseline = jnp.asarray([truth.get("halpha_disk_baseline", 0.0)])

    return _compute_disk_flux_vectorized(
        jnp.asarray(wave),
        center,
        inner_radius,
        outer_radius,
        sigma,
        inclination,
        q,
        eccentricity,
        apocenter,
        area,
        offset,
        baseline,
        integrator=pair.integrator,
        normalization_integrator=pair.normalization_integrator,
    )


def timed_profile(
    wave: np.ndarray,
    truth: dict[str, float],
    pair: IntegratorPair,
    n_repeats: int,
    n_warm_repeats: int,
) -> tuple[np.ndarray, dict[str, float]]:
    start = time.perf_counter()
    profile = evaluate_disk_profile(wave, truth, pair).block_until_ready()
    compile_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(n_warm_repeats):
        evaluate_disk_profile(wave, truth, pair).block_until_ready()

    times = []
    for _ in range(n_repeats):
        start = time.perf_counter()
        profile = evaluate_disk_profile(wave, truth, pair).block_until_ready()
        times.append((time.perf_counter() - start) * 1000.0)

    timing = {
        "compile_ms": compile_ms,
        "mean_ms": float(np.mean(times)),
        "median_ms": float(np.median(times)),
        "min_ms": float(np.min(times)),
        "p25_ms": float(np.percentile(times, 25.0)),
        "p75_ms": float(np.percentile(times, 75.0)),
        "std_ms": float(np.std(times)),
    }
    return np.asarray(profile), timing


def accuracy_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = candidate - reference
    ref_abs = np.abs(reference)
    support = ref_abs > max(1e-10, 1e-4 * float(ref_abs.max()))
    denom = np.maximum(ref_abs[support], 1e-12)
    rel = np.abs(delta[support]) / denom

    return {
        "max_abs_flux_diff": float(np.max(np.abs(delta))),
        "rms_abs_flux_diff": float(np.sqrt(np.mean(delta**2))),
        "max_rel_diff_support": float(np.max(rel)) if rel.size else np.nan,
        "rms_rel_diff_support": float(np.sqrt(np.mean(rel**2))) if rel.size else np.nan,
        "integrated_abs_diff": float(np.trapezoid(np.abs(delta))),
        "integrated_flux_ref": float(np.trapezoid(reference)),
        "integrated_flux_candidate": float(np.trapezoid(candidate)),
    }


def iter_integrator_pairs(args: argparse.Namespace) -> Iterable[IntegratorPair]:
    methods = parse_methods(args.methods)
    xi_orders = parse_ints(args.xi_orders)
    quad_phi_orders = parse_ints(args.quad_phi_orders)
    phi_bins = parse_ints(args.phi_bins)
    xi_bins = parse_ints(args.xi_bins)
    split_counts = parse_ints(args.split_counts)

    if "quad" in methods:
        for xi_order in xi_orders:
            for phi_order in quad_phi_orders:
                yield make_quad_pair(xi_order, phi_order)

    if "mixed" in methods:
        for xi_order in xi_orders:
            for phi_bin in phi_bins:
                yield make_mixed_pair(xi_order, phi_bin)

    if "trap" in methods:
        for xi_bin in xi_bins:
            for phi_bin in phi_bins:
                yield make_trap_pair(xi_bin, phi_bin)

    if "split_quad" in methods:
        for xi_order in xi_orders:
            for phi_order in quad_phi_orders:
                for split_count in split_counts:
                    yield make_split_quad_pair(xi_order, phi_order, split_count)


def make_reference_pair(args: argparse.Namespace) -> IntegratorPair:
    method = args.ref_method
    if method == "quad":
        return make_quad_pair(args.ref_xi_order, args.ref_phi_order)
    if method == "mixed":
        return make_mixed_pair(args.ref_xi_order, args.ref_phi_order)
    if method == "trap":
        return make_trap_pair(args.ref_xi_order, args.ref_phi_order)
    if method == "split_quad":
        return make_split_quad_pair(
            args.ref_xi_order, args.ref_phi_order, args.ref_split_count
        )
    raise ValueError(f"Unknown reference method: {method}")


def select_cases(args: argparse.Namespace) -> list[tuple[str, dict[str, float]]]:
    cases: list[tuple[str, dict[str, float]]] = []
    for name in args.case:
        if name == "all":
            cases.extend(DEFAULT_CASES.items())
        elif name in DEFAULT_CASES:
            cases.append((name, DEFAULT_CASES[name]))
        else:
            known = ", ".join(["all", *sorted(DEFAULT_CASES)])
            raise SystemExit(f"Unknown case '{name}'. Known cases: {known}")

    for path_text in args.truth_json:
        cases.append(load_truth_json(Path(path_text)))

    return cases


def print_recommendations(
    df: pd.DataFrame,
    time_metric: str,
    max_rel_threshold: float,
    top_k: int,
) -> None:
    print("\nRecommendations")
    grouped = df.groupby(["case", "n_wave"], sort=False)
    for (case_name, n_wave), group in grouped:
        valid = group[group["max_rel_diff_support"] <= max_rel_threshold]
        if valid.empty:
            ranked = group.sort_values(["max_rel_diff_support", time_metric]).head(top_k)
            note = f"no candidate below max_rel_diff_support <= {max_rel_threshold:g}"
        else:
            ranked = valid.sort_values([time_metric, "max_rel_diff_support"]).head(top_k)
            note = f"accuracy threshold {max_rel_threshold:g}"

        print(f"\n{case_name}, n_wave={n_wave} ({note})")
        columns = [
            "label",
            time_metric,
            "max_rel_diff_support",
            "rms_rel_diff_support",
            "max_abs_flux_diff",
        ]
        print(ranked[columns].to_string(index=False, float_format=lambda value: f"{value:.4g}"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        default=None,
        help="Built-in truth case to run. Use multiple times or pass 'all'.",
    )
    parser.add_argument(
        "--truth-json",
        action="append",
        default=[],
        help="JSON file containing a truth dictionary. May include top-level {'truth': ...}.",
    )
    parser.add_argument(
        "--methods",
        default="quad,mixed,trap",
        help="Comma-separated methods: quad,mixed,trap,split_quad.",
    )
    parser.add_argument(
        "--xi-orders",
        default="16,32,48",
        help="Comma-separated Clenshaw-Curtis Xi orders for quad/mixed/split_quad.",
    )
    parser.add_argument(
        "--quad-phi-orders",
        default="64,128,192",
        help="Comma-separated Clenshaw-Curtis phi orders for quad/split_quad.",
    )
    parser.add_argument(
        "--phi-bins",
        default="64,128,192,256",
        help="Comma-separated midpoint phi bins for mixed/trap.",
    )
    parser.add_argument(
        "--xi-bins",
        default="16,32,64",
        help="Comma-separated log-Xi trapezoid bins for trap.",
    )
    parser.add_argument(
        "--split-counts",
        default="4",
        help="Comma-separated Xi split counts for split_quad.",
    )
    parser.add_argument(
        "--n-wave-values",
        default="128,256,512",
        help="Comma-separated wavelength grid sizes to test.",
    )
    parser.add_argument(
        "--ref-method",
        default="quad",
        choices=["quad", "mixed", "trap", "split_quad"],
        help="Reference integration method used to compute accuracy metrics.",
    )
    parser.add_argument(
        "--ref-xi-order",
        type=int,
        default=96,
        help="Reference Xi order/bins. For trap this is log-Xi bins.",
    )
    parser.add_argument(
        "--ref-phi-order",
        type=int,
        default=384,
        help="Reference phi order/bins. For mixed/trap this is midpoint bins.",
    )
    parser.add_argument(
        "--ref-split-count",
        type=int,
        default=4,
        help="Reference phi split count for --ref-method=split_quad.",
    )
    parser.add_argument("--n-repeats", type=int, default=10)
    parser.add_argument("--n-warm-repeats", type=int, default=2)
    parser.add_argument(
        "--time-metric",
        default="min_ms",
        choices=["min_ms", "p25_ms", "median_ms", "mean_ms"],
        help="Timing metric used for recommendations.",
    )
    parser.add_argument(
        "--max-rel-threshold",
        type=float,
        default=2e-2,
        help="Maximum relative profile error on support used for recommendations.",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--output-csv",
        default=str(ROOT / "results" / "integrator_benchmark.csv"),
        help="Path for benchmark table.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.n_repeats < 1:
        raise SystemExit("--n-repeats must be >= 1")

    if args.case is None:
        args.case = ["moderate", "fragile", "large_extent"]
    cases = select_cases(args)
    n_wave_values = parse_ints(args.n_wave_values)
    configs = list(iter_integrator_pairs(args))
    reference_pair = make_reference_pair(args)

    print(f"JAX backend: {jax.default_backend()}")
    print(f"JAX devices: {jax.devices()}")
    print(f"Cases: {', '.join(name for name, _ in cases)}")
    print(f"Candidate integrators: {len(configs)}")
    print(f"Reference: {reference_pair.label}")

    rows: list[dict[str, float | int | str | None]] = []
    for case_name, truth in cases:
        for n_wave in n_wave_values:
            wave = observed_wave_grid(truth, n_wave)
            reference = np.asarray(evaluate_disk_profile(wave, truth, reference_pair).block_until_ready())

            for pair in configs:
                profile, timing = timed_profile(
                    wave,
                    truth,
                    pair,
                    n_repeats=args.n_repeats,
                    n_warm_repeats=args.n_warm_repeats,
                )
                metrics = accuracy_metrics(reference, profile)
                row = {
                    "case": case_name,
                    "n_wave": n_wave,
                    "label": pair.label,
                    "method": pair.method,
                    "reference_label": reference_pair.label,
                    "reference_method": reference_pair.method,
                    "xi_param": pair.xi_param,
                    "phi_param": pair.phi_param,
                    "split_param": pair.split_param,
                    "inner_radius": truth["halpha_disk_inner_radius"],
                    "radius_ratio": truth["halpha_disk_radius_ratio"],
                    "outer_radius": disk_outer_radius(truth),
                    **timing,
                    **metrics,
                }
                rows.append(row)
                print(
                    f"{case_name:>16} n={n_wave:<4d} {pair.label:<32} "
                    f"{args.time_metric}={row[args.time_metric]:8.3f} "
                    f"max_rel={row['max_rel_diff_support']:.4g}"
                )

    df = pd.DataFrame(rows)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nWrote benchmark table to {output_csv}")
    print_recommendations(
        df,
        time_metric=args.time_metric,
        max_rel_threshold=args.max_rel_threshold,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
