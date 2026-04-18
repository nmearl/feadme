from astropy.utils import lazyproperty

from .core.parser import Config
import arviz as az
import arviz_stats
import pandas as pd
import numpy as np
import scipy.stats
import xarray as xr
from pathlib import Path


class Reporter:
    def __init__(self, config: Config, idata: xr.DataTree):
        self._config = config
        self._idata = idata

    @lazyproperty
    def summary(self) -> pd.DataFrame:
        """
        Get the summary statistics of the posterior samples.
        """
        # Compute the original summary
        with pd.option_context("display.precision", 10):
            linear_summary = az.summary(
                self._idata,
                kind="all_median",
                ci_prob=0.68,
                var_names=[
                    x
                    for x in self._idata["posterior"].dataset.data_vars
                    if x in self._config.template.fitted_parameter_names()
                    or x.endswith("_outer_radius")
                ],
                round_to=10,
            )

            circ_names = [
                x
                for x in self._idata["posterior"].dataset.data_vars
                if x in [pr.name for pr in self._config.template.iter_circular]
            ]
            if circ_names:
                circ_rows = {}
                for var in circ_names:
                    samples = self._idata["posterior"].dataset[var].values.ravel()
                    circ_mean = scipy.stats.circmean(samples, high=2 * np.pi, low=0)
                    hdi_bounds = arviz_stats.hdi(
                        self._idata["posterior"].dataset[var], prob=0.68, circular=True
                    )
                    lb = float(hdi_bounds.sel(ci_bound="lower")) % (2 * np.pi)
                    ub = float(hdi_bounds.sel(ci_bound="upper")) % (2 * np.pi)
                    # err_lo/err_hi are wrapped distances from the circular mean
                    # to the HDI bounds. The HDI is computed independently from
                    # the mean, so these are not symmetric errors about the mean;
                    # they are forward wrapped distances from the reported center
                    # to each reported interval endpoint.
                    circ_rows[var] = {
                        "value": circ_mean,
                        "mean": circ_mean,
                        "hdi68_lb": lb,
                        "hdi68_ub": ub,
                        "err_lo": (circ_mean - lb) % (2 * np.pi),
                        "err_hi": (ub - circ_mean) % (2 * np.pi),
                    }
                circular_summary = pd.DataFrame.from_dict(circ_rows, orient="index")
                circular_summary.index.name = "parameter"
            else:
                circular_summary = None

        linear_summary.index.name = "parameter"
        ci_kind = "hdi" if "hdi68_lb" in linear_summary.columns else "eti"
        lb_col = f"{ci_kind}68_lb"
        ub_col = f"{ci_kind}68_ub"
        linear_summary["err_lo"] = linear_summary["median"] - linear_summary[lb_col]
        linear_summary["err_hi"] = linear_summary[ub_col] - linear_summary["median"]
        linear_summary["value"] = linear_summary["median"]

        if circular_summary is not None:
            summary = pd.concat([linear_summary, circular_summary])
        else:
            summary = linear_summary

        # Add in fixed parameters with zero error bars
        for param_ref in self._config.template.iter_fixed:
            if param_ref.name in self._idata["posterior"].dataset.data_vars:
                val = float(self._idata["posterior"].dataset[param_ref.name].values.flat[0])
                summary.loc[param_ref.name] = {
                    "value": val,
                    "err_lo": 0.0,
                    "err_hi": 0.0,
                }

        # Reorder columns for improved readability
        summary = summary[
            ["value", "err_lo", "err_hi"]
            + [
                col
                for col in summary.columns
                if col not in ["value", "err_lo", "err_hi"]
            ]
        ]

        summary["shared"] = summary.index.isin(
            [param_ref.name for param_ref in self._config.template.iter_shared]
        )

        summary["fixed"] = summary.index.isin(
            [param_ref.name for param_ref in self._config.template.iter_fixed]
        )

        # Sort table so fixed variables are at the bottom, then sort by index
        summary = summary.sort_values(
            by=["fixed", "shared", "parameter"], ascending=[True, True, True]
        )

        return summary

    def write_results(self):
        """
        Write the results of the sampling to the output path specified in the config.
        """
        if self._idata is None:
            raise ValueError("Inference data not available. Run the sampler first.")

        out_path = Path(f"{self._config.output_path}") / "results.nc"

        self._idata.to_netcdf(str(out_path))

        self.summary.to_csv(
            f"{self._config.output_path}/summary.csv",
            index=True,
        )
