from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, field

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.time import Time
from loguru import logger
from numpyro.diagnostics import summary as summarize
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer.util import Predictive
from scipy.stats import gaussian_kde

from ..plotting import plot_results
from ..utils import circular_rhat
from ..parser import Template

finfo = np.finfo(float)


@dataclass
class SamplerConfig:
    """Configuration class for sampler parameters."""

    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 4
    progress_bar: bool = True
    use_quad: bool = False
    rng_key: Optional[jax.random.PRNGKey] = None
    convergence_threshold: tuple = (0.99, 1.01)
    max_convergence_attempts: int = 5
    double_samples_every: int = 2

    def __post_init__(self):
        if self.rng_key is None:
            self.rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))


@dataclass
class DataContainer:
    """Container for observational data."""

    wave: np.ndarray
    flux: np.ndarray
    flux_err: np.ndarray

    def apply_mask(self, mask: np.ndarray) -> "DataContainer":
        """Apply a mask to the data and return a new container."""
        return DataContainer(
            wave=self.wave[mask], flux=self.flux[mask], flux_err=self.flux_err[mask]
        )


class Sampler(ABC):
    """Abstract base class for MCMC samplers with improved extensibility."""

    def __init__(
        self,
        model: Callable,
        template: Template,
        data: DataContainer,
        output_dir: str,
        label: str,
        config: Optional[SamplerConfig] = None,
    ):
        self._model = model
        self._template = template
        self._config = config or SamplerConfig()

        # Apply data masking
        self._data = self._apply_data_mask(data, template)

        # Setup output
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._label = label

        # Template properties
        self._mjd = template.mjd
        self._redshift = template.redshift

        # Initialize RNG keys
        self._rng_keys = jax.random.split(self._config.rng_key, self._config.num_chains)

        # State
        self._idata = None
        self._load_existing_results()

    @staticmethod
    def _apply_data_mask(data: DataContainer, template: Template) -> DataContainer:
        """Apply template masks to the data."""
        masks = [
            np.bitwise_and(data.wave > m.lower_limit, data.wave < m.upper_limit)
            for m in template.mask
        ]

        if len(masks) > 1:
            combined_mask = np.bitwise_or.reduce(masks)
        else:
            combined_mask = masks[0]

        return data.apply_mask(combined_mask)

    def _load_existing_results(self) -> None:
        """Load existing MCMC results if available."""
        result_path = self._output_dir / f"{self._label}.nc"
        if result_path.exists():
            logger.info(f"Loading existing MCMC sampler from {result_path}")
            self._idata = az.from_netcdf(str(result_path))

    @abstractmethod
    def flat_posterior_samples(self) -> Dict[str, np.ndarray]:
        """Return flattened posterior samples."""
        raise NotImplementedError("Flat posterior samples not implemented.")

    @abstractmethod
    def _create_sampler(self, **kwargs) -> Any:
        """Create the specific sampler instance."""
        raise NotImplementedError("Sampler creation not implemented.")

    @abstractmethod
    def _run_sampling(self, sampler: Any, **kwargs) -> Any:
        """Run the sampling process."""
        raise NotImplementedError("Sampling execution not implemented.")

    @abstractmethod
    def _compose_inference_data(self, sampler_result: Any) -> az.InferenceData:
        """Create ArviZ InferenceData from sampler results."""
        raise NotImplementedError("Inference data composition not implemented.")

    def sample(self, **kwargs) -> None:
        """Main sampling method with convergence checking."""
        if self._idata is not None and self.check_convergence(show=False):
            logger.info(f"Results for {self._label} already converged.")
            self.plot_results()
            return

        start_time = Time.now()

        sampler = None
        converged = False
        attempt = 0

        current_config = self._config

        while not converged and attempt < current_config.max_convergence_attempts:
            try:
                if sampler is None:
                    sampler = self._create_sampler(**kwargs)

                sampler_result = self._run_sampling(sampler, **kwargs)
                self._idata = self._compose_inference_data(sampler_result)

                converged = self.check_convergence()

                if not converged:
                    attempt += 1
                    sampler = self._handle_convergence_failure(attempt, current_config)
                    if sampler is None:  # Config was updated, need new sampler
                        current_config = self._update_config_for_retry(
                            attempt, current_config
                        )

            except Exception as e:
                logger.error(f"Sampling failed for {self._label}: {e}")
                if attempt >= current_config.max_convergence_attempts - 1:
                    raise
                attempt += 1
                sampler = None

        if not converged:
            logger.critical(
                f"Convergence failed for {self._label} after {attempt} attempts."
            )

        self._finalize_results()

        delta_time = (Time.now() - start_time).to_datetime()
        logger.info(f"Finished sampling {self._label} in {delta_time}.")

    def _handle_convergence_failure(
        self, attempt: int, config: SamplerConfig
    ) -> Optional[Any]:
        """Handle convergence failure and decide whether to continue or restart."""
        logger.info(f"R_hat values not converged. Attempt {attempt}")

        self.write_results()  # Save intermediate results
        self.plot_results()

        if attempt % config.double_samples_every == 0:
            logger.warning(f"Doubling samples after {attempt} attempts")
            return None  # Signal to create new sampler with updated config

        return self._get_current_sampler()  # Continue with current sampler

    def _update_config_for_retry(
        self, attempt: int, config: SamplerConfig
    ) -> SamplerConfig:
        """Update configuration for retry attempts."""
        new_config = SamplerConfig(
            num_warmup=config.num_warmup * 2,
            num_samples=config.num_samples * 2,
            num_chains=config.num_chains,
            progress_bar=config.progress_bar,
            use_quad=config.use_quad,
            rng_key=config.rng_key,
            convergence_threshold=config.convergence_threshold,
            max_convergence_attempts=config.max_convergence_attempts,
            double_samples_every=config.double_samples_every,
        )
        self._config = new_config
        return new_config

    def _get_current_sampler(self) -> Optional[Any]:
        """Get the current sampler state for continuation."""
        return None  # Default implementation, override in subclasses

    def _finalize_results(self) -> None:
        """Finalize and save results."""
        self.write_results()
        self.plot_results()
        self.write_run()

    # Properties
    @property
    def label(self) -> str:
        return self._label

    @property
    def mjd(self) -> float:
        return self._mjd

    @property
    def redshift(self) -> float:
        return self._redshift

    @property
    def config(self) -> SamplerConfig:
        return self._config

    @property
    def data(self) -> DataContainer:
        return self._data

    @property
    def fixed_fields(self) -> List[str]:
        """Get list of fixed parameter fields."""
        ff = [
            f"{prof.name}_{param.name}"
            for prof in self._template.disk_profiles + self._template.line_profiles
            for param in prof.fixed
        ]

        # Add shared parameters that are fixed
        ff += [
            f"{prof.name}_{param.name}"
            for prof in self._template.disk_profiles + self._template.line_profiles
            for param in prof.shared
            if f"{param.shared}_{param.name}" in ff
        ]

        if hasattr(self._template, "white_noise") and self._template.white_noise.fixed:
            ff.append(self._template.white_noise.name)

        return ff

    # Analysis methods
    def _get_fitting_summary(self) -> Optional[pd.DataFrame]:
        """Get fitting summary statistics."""
        if self._idata is None:
            return None

        prior_summary = summarize(
            self._idata.to_dict()["posterior"], group_by_chain=True
        )

        data = {key: [] for key in ["param", "mean", "std", "n_eff", "r_hat"]}

        for param, stats in prior_summary.items():
            if self._should_skip_parameter(param):
                continue

            data["param"].append(param)
            data["mean"].append(stats["mean"])
            data["std"].append(stats["std"])
            data["n_eff"].append(np.mean(stats["n_eff"]))
            data["r_hat"].append(self._calculate_rhat(param, stats))

        return pd.DataFrame(data) if data["param"] else None

    def _should_skip_parameter(self, param: str) -> bool:
        """Check if parameter should be skipped in summary."""
        skip_patterns = ["disk_flux", "line_flux", "_base", "_wrap"]
        return (
            any(pattern in param for pattern in skip_patterns)
            or param in self.fixed_fields
        )

    def _calculate_rhat(self, param: str, stats: Dict) -> float:
        """Calculate R-hat statistic for parameter."""
        if param.endswith("apocenter"):
            return circular_rhat(self._idata.posterior[param].values)
        return np.mean(stats["r_hat"])

    def _get_results_summary(self) -> Table:
        """Get comprehensive results summary."""
        fit_summary = self._get_fitting_summary()
        if fit_summary is None:
            return Table()

        fit_dict = fit_summary.set_index("param").to_dict(orient="index")
        posterior_summary = self._summarize_posterior(self.flat_posterior_samples())

        result_data = {
            key: []
            for key in [
                "label",
                "mjd",
                "redshift",
                "param",
                "value",
                "err_lo",
                "err_hi",
                "n_eff",
                "r_hat",
            ]
        }

        for param, values in posterior_summary.items():
            if self._should_skip_parameter(param):
                continue

            result_data["label"].append(self.label)
            result_data["mjd"].append(self.mjd)
            result_data["redshift"].append(self.redshift)
            result_data["param"].append(param)
            result_data["value"].append(values["median"])
            result_data["err_lo"].append(values["median"] - values["16%"])
            result_data["err_hi"].append(values["84%"] - values["median"])

            # Get fit statistics with fallback
            fit_stats = fit_dict.get(param, {"n_eff": np.nan, "r_hat": np.nan})
            result_data["n_eff"].append(fit_stats["n_eff"])
            result_data["r_hat"].append(fit_stats["r_hat"])

        return Table(result_data)

    def _summarize_posterior(
        self, samples: Dict[str, np.ndarray]
    ) -> Dict[str, Dict[str, float]]:
        """Summarize posterior samples with proper handling of circular parameters."""
        summary = {}

        for param, values in samples.items():
            if param.endswith("apocenter"):
                summary[param] = self._summarize_circular_parameter(values)
            else:
                summary[param] = self._summarize_linear_parameter(values)

        return summary

    @staticmethod
    def _summarize_circular_parameter(values: np.ndarray) -> Dict[str, float]:
        """Summarize circular parameter (e.g., angles)."""
        mean_angle = np.arctan2(np.mean(np.sin(values)), np.mean(np.cos(values))) % (
            2 * np.pi
        )
        rotated_values = (values - mean_angle + np.pi) % (2 * np.pi)

        lower_bound_rot = np.percentile(rotated_values, 16)
        upper_bound_rot = np.percentile(rotated_values, 84)

        return {
            "median": mean_angle,
            "16%": (lower_bound_rot - np.pi + mean_angle) % (2 * np.pi),
            "84%": (upper_bound_rot - np.pi + mean_angle) % (2 * np.pi),
        }

    @staticmethod
    def _summarize_linear_parameter(values: np.ndarray) -> Dict[str, float]:
        """Summarize linear parameter."""
        return {
            "median": float(jnp.median(values)),
            "16%": float(jnp.percentile(values, 16)),
            "84%": float(jnp.percentile(values, 84)),
        }

    def check_convergence(self, show: bool = True) -> bool:
        """Check MCMC convergence using R-hat statistics."""
        summary_df = self._get_fitting_summary()

        if summary_df is None or summary_df.empty:
            return False

        if show:
            logger.info(
                f"Fitting results for {self._label}:\n{summary_df.to_markdown()}"
            )

        avg_rhat = summary_df["r_hat"].mean()
        logger.info(f"Average R_hat: {avg_rhat:.4f}")

        threshold_low, threshold_high = self._config.convergence_threshold
        return threshold_low <= avg_rhat < threshold_high

    # I/O methods
    def write_results(self) -> None:
        """Write results to CSV file."""
        results_table = self._get_results_summary()
        output_path = self._output_dir / "disk_param_results.csv"

        results_table.write(str(output_path), format="ascii.csv", overwrite=True)
        logger.debug(f"Results written to {output_path}")

    def write_run(self) -> None:
        """Write MCMC run to NetCDF file."""
        if self._idata is not None:
            output_path = self._output_dir / f"{self.label}.nc"
            az.to_netcdf(self._idata, str(output_path))
            logger.debug(f"MCMC run written to {output_path}")

    def plot_results(self) -> None:
        """Generate result plots."""
        if self._idata is not None:
            plot_results(
                self._template,
                str(self._output_dir),
                self._idata,
                self._get_results_summary(),
                self._data.wave,
                self._data.flux,
                self._data.flux_err,
                self.label,
                self.fixed_fields,
            )
