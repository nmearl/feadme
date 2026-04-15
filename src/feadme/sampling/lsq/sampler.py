import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import arviz as az
import flax.struct
import loguru
import numpy as np
from astropy.modeling.fitting import TRFLSQFitter
from astropy.modeling.models import Const1D
from jax.typing import ArrayLike

from .model import LSQModel
from ..base_sampler import BaseSampler
from ..initializers import BaseInitializer, DefaultInitializer
from ...core.parser import Config

logger = loguru.logger.opt(colors=True)


def _bootstrap_worker(
    i: int,
    model,
    wave: np.ndarray,
    best_fit_flux: np.ndarray,
    residuals: np.ndarray,
    flux_err: np.ndarray,
    maxiter: int,
    seed: int,
) -> tuple[int, np.ndarray]:
    """
    Worker function for a single bootstrap iteration.

    Parameters
    ----------
    i : int
        Bootstrap iteration index
    model : Model
        Astropy model to fit
    wave : np.ndarray
        Wavelength array
    best_fit_flux : np.ndarray
        Best-fit flux values
    residuals : np.ndarray
        Residuals from best fit
    flux_err : np.ndarray
        Flux errors
    maxiter : int
        Maximum iterations for fitter
    seed : int
        Random seed

    Returns
    -------
    tuple
        (iteration_index, fitted_parameters or None)
    """
    # Set random seed for reproducibility
    np.random.seed(seed)

    n_data = len(wave)

    try:
        # Residual bootstrap: resample residuals, add to fitted values
        boot_indices = np.random.choice(n_data, size=n_data, replace=True)
        boot_flux = best_fit_flux + residuals[boot_indices]

        # Refit the model to bootstrap sample
        fitter = TRFLSQFitter(calc_uncertainties=False)
        boot_mod = fitter(
            model.copy(),  # Use a fresh copy
            wave,
            boot_flux,
            maxiter=maxiter,
            weights=1 / flux_err,
            filter_non_finite=True,
        )

        return i, boot_mod.parameters

    except Exception as e:
        # Return None to indicate failure
        return i, None


@flax.struct.dataclass
class LSQSampler(BaseSampler):
    sampler_type = "lsq"
    maxiter: int = 2000
    n_bootstrap: int = 500  # Number of bootstrap samples
    estimate_uncertainties: bool = False
    n_jobs: int = -1  # Number of parallel jobs (-1 = all CPUs)
    initializer: BaseInitializer = DefaultInitializer()

    def __call__(self, config: Config, model: LSQModel) -> az.InferenceData:
        wave = config.data.masked_wave
        flux = config.data.masked_flux
        flux_err = config.data.masked_flux_err

        # First, fit the model to get the best-fit parameters
        fitter = TRFLSQFitter(calc_uncertainties=False)

        fit_mod = fitter(
            model.model,
            wave,
            flux,
            maxiter=self.maxiter,
            weights=1 / flux_err,
            filter_non_finite=True,
        )

        if self.estimate_uncertainties:
            logger.info(
                f"Running bootstrap with {self.n_bootstrap} samples using {self._get_n_workers()} workers..."
            )
            posterior_samples_array = self._bootstrap_sampling(
                model, wave, flux, flux_err, fit_mod
            )
        else:
            # Just use the point estimate
            posterior_samples_array = np.tile(fit_mod.parameters, (100, 1))

        # Convert to dict with proper parameter names
        posterior_samples = self._format_posterior_samples(
            config, model, fit_mod, posterior_samples_array
        )

        # Construct posterior predictive samples
        posterior_predictive_samples = self._construct_posterior_predictive(
            config, fit_mod, wave, posterior_samples_array
        )

        return self._compose_inference_data(
            config, model, posterior_samples, posterior_predictive_samples
        )

    def _get_n_workers(self) -> int:
        """Get number of worker threads to use."""
        import os

        if self.n_jobs == -1:
            return os.cpu_count() or 1
        return max(1, self.n_jobs)

    def _bootstrap_sampling(
        self,
        model: LSQModel,
        wave: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        fit_mod,
    ) -> np.ndarray:
        """
        Perform bootstrap resampling to estimate parameter uncertainties.
        Uses multithreading for parallel execution.
        """
        n_params = len(fit_mod.parameters)
        bootstrap_params = np.zeros((self.n_bootstrap, n_params))

        # Get residuals from best fit
        best_fit_flux = fit_mod(wave)
        residuals = flux - best_fit_flux

        # Use different seeds for each bootstrap iteration
        base_seed = int(time.time() * 1000) % 2**32

        n_workers = self._get_n_workers()
        completed = 0
        failed = 0

        # Using threads for parallel execution (JAX safe)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            # Submit all tasks
            futures = {
                executor.submit(
                    _bootstrap_worker,
                    i,
                    model.model,
                    wave,
                    best_fit_flux,
                    residuals,
                    flux_err,
                    self.maxiter,
                    base_seed + i,
                ): i
                for i in range(self.n_bootstrap)
            }

            # Process results as they complete
            for future in as_completed(futures):
                try:
                    i, params = future.result()

                    if params is not None:
                        bootstrap_params[i] = params
                    else:
                        # Use best-fit parameters for failed iterations
                        bootstrap_params[i] = fit_mod.parameters
                        failed += 1

                    completed += 1

                    # Log progress every 10% or every 50 samples
                    log_interval = max(1, min(self.n_bootstrap // 10, 50))
                    if completed % log_interval == 0:
                        logger.info(
                            f"Bootstrap progress: {completed}/{self.n_bootstrap} "
                            f"({100*completed/self.n_bootstrap:.1f}%) - "
                            f"Failed: {failed}"
                        )

                except Exception as e:
                    logger.error(f"Error processing bootstrap result: {e}")
                    i = futures[future]
                    bootstrap_params[i] = fit_mod.parameters
                    failed += 1

        if failed > 0:
            logger.warning(
                f"Bootstrap completed with {failed}/{self.n_bootstrap} failures "
                f"({100*failed/self.n_bootstrap:.1f}%)"
            )
        else:
            logger.info("Bootstrap completed successfully!")

        return bootstrap_params

    def _format_posterior_samples(
        self,
        config: Config,
        model: LSQModel,
        fit_mod,
        posterior_samples_array: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Convert parameter array to named dictionary."""
        model_names = [m.name for m in model.model]
        param_names = []

        for pn in fit_mod.param_names:
            pn, pi = "_".join(pn.split("_")[:-1]), int(pn.split("_")[-1])
            param_names.append(f"{model_names[pi]}_{pn}")

        posterior_samples = {
            k: posterior_samples_array[:, i] for i, k in enumerate(param_names)
        }

        # Reconstruct shared parameter distributions
        for prof in config.template.disk_profiles + config.template.line_profiles:
            for param in prof.shared:
                shared_param_name = f"{param.shared}_{param.name}"
                posterior_samples[param.qualified_name] = posterior_samples[
                    shared_param_name
                ]

        # Get real redshift samples
        fit_z = 1 / (1 + posterior_samples.pop("redshift_z")) - 1
        posterior_samples["redshift"] = fit_z

        # Transform log-based parameters back to linear space
        for pn in fit_mod.meta["log_dist"]:
            posterior_samples[pn] = 10 ** posterior_samples.pop(pn)

        # Add fractional model-jitter term for downstream summary compatibility.
        if self.estimate_uncertainties:
            posterior_samples["log_frac_noise"] = np.random.normal(
                loc=0, scale=1e-3, size=len(posterior_samples_array)
            )
        else:
            posterior_samples["log_frac_noise"] = np.zeros(len(posterior_samples_array))

        return posterior_samples

    def _construct_posterior_predictive(
        self,
        config: Config,
        fit_mod,
        wave: np.ndarray,
        posterior_samples_array: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Construct posterior predictive samples by looping over parameter sets."""
        disk_submodels = [
            fit_mod[sm_idx]
            for sm_idx in range(fit_mod.n_submodels)
            if fit_mod[sm_idx].name
            in [prof.name for prof in config.template.disk_profiles]
        ]
        disk_model = fit_mod["redshift"] | (
            np.sum(disk_submodels) if len(disk_submodels) > 0 else Const1D(amplitude=0)
        )

        line_submodels = [
            fit_mod[sm_idx]
            for sm_idx in range(fit_mod.n_submodels)
            if fit_mod[sm_idx].name
            in [prof.name for prof in config.template.line_profiles]
        ]
        line_model = fit_mod["redshift"] | (
            np.sum(line_submodels) if len(line_submodels) > 0 else Const1D(amplitude=0)
        )

        if self.estimate_uncertainties:
            n_samples = len(posterior_samples_array)
            total_flux = np.zeros((n_samples, len(wave)))

            logger.info("Evaluating posterior predictive samples...")
            for i in range(n_samples):
                try:
                    # Set parameters for this sample
                    fit_mod.parameters = posterior_samples_array[i]

                    # Evaluate models
                    total_flux[i] = fit_mod(wave)

                    if (i + 1) % 100 == 0:
                        logger.info(
                            f"Posterior predictive progress: {i + 1}/{n_samples}"
                        )
                except Exception as e:
                    logger.warning(f"Sample {i} evaluation failed: {e}")
                    # Use NaNs for failed evaluations
                    total_flux[i] = np.nan

            posterior_predictive_samples = {
                "disk_flux": np.array([np.tile(disk_model(wave), (n_samples, 1))]),
                "line_flux": np.array([np.tile(line_model(wave), (n_samples, 1))]),
                "total_flux": total_flux[np.newaxis, :, :],
            }
        else:
            n_samples = len(posterior_samples_array)
            posterior_predictive_samples = {
                "disk_flux": np.array([np.tile(disk_model(wave), (n_samples, 1))]),
                "line_flux": np.array([np.tile(line_model(wave), (n_samples, 1))]),
                "total_flux": np.array([np.tile(fit_mod(wave), (n_samples, 1))]),
            }

        return posterior_predictive_samples

    @staticmethod
    def _compose_inference_data(
        config: Config,
        model: LSQModel,
        posterior_samples: dict[str, ArrayLike],
        posterior_predictive_samples: dict[str, ArrayLike],
    ) -> az.InferenceData:
        # Simulate multiple chains by reshaping the samples
        num_chains = 2

        flat_samples = {
            k: v.reshape(num_chains, -1) for k, v in posterior_samples.items()
        }

        idata = az.from_dict(
            posterior=flat_samples,
            posterior_predictive=posterior_predictive_samples,
        )

        return idata
