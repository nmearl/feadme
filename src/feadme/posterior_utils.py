from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr


def select_representative_draw(
    posterior: xr.Dataset,
    posterior_predictive: xr.Dataset | None = None,
    *,
    flux_name: str = "total_flux",
) -> tuple[int, int]:
    """
    Choose the posterior draw whose predicted flux is closest to the posterior
    predictive median in L2 distance.
    """
    if (
        posterior_predictive is not None
        and flux_name in posterior_predictive
        and "chain" in posterior_predictive[flux_name].dims
        and "draw" in posterior_predictive[flux_name].dims
    ):
        pp_total = posterior_predictive[flux_name]
        obs_dist = pp_total.stack(sample=("chain", "draw")).values
        pp_median = np.median(obs_dist, axis=1)
        distances = np.sum((obs_dist - pp_median[:, None]) ** 2, axis=0)
        best_flat = int(np.argmin(distances))
        chain_idx, draw_idx = np.unravel_index(
            best_flat, (pp_total.sizes["chain"], pp_total.sizes["draw"])
        )
        return int(chain_idx), int(draw_idx)

    if "chain" not in posterior.sizes or "draw" not in posterior.sizes:
        raise ValueError("Posterior dataset is missing chain/draw dimensions")
    return 0, 0


def extract_draw_values(
    posterior: xr.Dataset,
    chain_idx: int,
    draw_idx: int,
) -> dict[str, float]:
    return {
        name: float(var.values[chain_idx, draw_idx])
        for name, var in posterior.data_vars.items()
        if "chain" in var.dims and "draw" in var.dims and var.ndim == 2
    }


def load_representative_draw_values(
    results_nc: str | Path,
    *,
    flux_name: str = "total_flux",
) -> dict[str, float]:
    posterior = xr.open_dataset(results_nc, group="posterior")
    try:
        posterior_predictive = xr.open_dataset(results_nc, group="posterior_predictive")
    except Exception:
        posterior_predictive = None
    chain_idx, draw_idx = select_representative_draw(
        posterior,
        posterior_predictive,
        flux_name=flux_name,
    )
    return extract_draw_values(posterior, chain_idx, draw_idx)
