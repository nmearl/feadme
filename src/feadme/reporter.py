from astropy.utils import lazyproperty

from .core.parser import Config
import arviz as az
import pandas as pd
import numpy as np
from pathlib import Path


class Reporter:
    def __init__(self, config: Config, idata: az.InferenceData):
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
                stat_focus="median",
                hdi_prob=0.68,
                var_names=[
                    x
                    for x in self._idata.posterior.data_vars
                    if x in self._config.template.fitted_parameter_names()
                    or "outer_radius" in x
                ],
                round_to=10,
            )

            if len(self._config.template.iter_circular) > 0:
                circular_summary = az.summary(
                    self._idata,
                    stat_focus="mean",
                    hdi_prob=0.68,
                    var_names=[
                        x
                        for x in self._idata.posterior.data_vars
                        if x in [pr.name for pr in self._config.template.iter_circular]
                    ],
                    circ_var_names=[
                        x
                        for x in self._idata.posterior.data_vars
                        if x in [pr.name for pr in self._config.template.iter_circular]
                    ],
                    round_to=10,
                )
            else:
                circular_summary = None

        for summary in [linear_summary, circular_summary]:
            if summary is None:
                continue

            # Rename index column to "parameter", then remove index
            summary.index.name = "parameter"

            col_stat = "hdi" if "hdi_16%" in summary.columns else "eti"
            val_stat = "median" if "median" in summary.columns else "mean"

            if summary is circular_summary:
                # Wrap mean and percentiles to [0, 2π]
                summary["mean"] = summary["mean"] % (2 * np.pi)
                summary[f"{col_stat}_16%"] = summary[f"{col_stat}_16%"] % (2 * np.pi)
                summary[f"{col_stat}_84%"] = summary[f"{col_stat}_84%"] % (2 * np.pi)

            summary["err_lo"] = summary[val_stat] - summary[f"{col_stat}_16%"]
            summary["err_hi"] = summary[f"{col_stat}_84%"] - summary[val_stat]
            summary["value"] = summary[val_stat]

        if circular_summary is not None:
            summary = pd.concat([linear_summary, circular_summary])
        else:
            summary = linear_summary

        # Add in fixed parameters with zero error bars
        for param_ref in self._config.template.iter_fixed:
            if param_ref.name in self._idata.posterior.data_vars:
                val = float(self._idata.posterior[param_ref.name].values.flat[0])
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

        az.to_netcdf(self._idata, str(out_path))

        self.summary.to_csv(
            f"{self._config.output_path}/summary.csv",
            index=True,
        )
