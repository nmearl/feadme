import time

import arviz as az
import flax.struct
import jax.random as random
import loguru
import numpy as np
from astropy.modeling.fitting import (
    TRFLSQFitter,
    model_to_fit_params,
)
from astropy.modeling.models import Const1D
from jax.typing import ArrayLike

from .model import LSQModel
from ..base_sampler import BaseSampler
from ..initializers import BaseInitializer, DefaultInitializer
from ...core.parser import Config

logger = loguru.logger.opt(colors=True)


@flax.struct.dataclass
class LSQSampler(BaseSampler):
    sampler_type = "lsq"
    maxiter: int = 2000
    estimate_uncertainties: bool = False
    initializer: BaseInitializer = DefaultInitializer()

    def __call__(self, config: Config, model: LSQModel) -> az.InferenceData:
        wave = config.data.masked_wave
        flux = config.data.masked_flux
        flux_err = config.data.masked_flux_err

        _, indices, _ = model_to_fit_params(model.model)

        fitter = TRFLSQFitter(calc_uncertainties=True)

        fit_mod = fitter(
            model.model,
            wave,
            flux,
            maxiter=self.maxiter,
            weights=1 / flux_err,
            filter_non_finite=True,
        )
        cov = fitter.fit_info["param_cov"]

        # Create posterior samples by sampling from a multivariate normal
        posterior_samples_array = np.tile(fit_mod.parameters, (self.maxiter, 1))

        if self.estimate_uncertainties:
            rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
            posterior_samples_array[:, indices] = random.multivariate_normal(
                rng_key,
                mean=fit_mod.parameters[indices],
                cov=cov,
                shape=(self.maxiter,),
            )

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

        if self.estimate_uncertainties:
            # Add in simulated normal values for log uncertainty inflation
            posterior_samples["log_white_noise"] = np.random.normal(
                loc=0, scale=1e-3, size=self.maxiter
            )
        else:
            posterior_samples["log_white_noise"] = np.zeros(self.maxiter)

        # Construct posterior predictive samples for likelihood evaluation
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
            total_flux = []

            for i in range(self.maxiter):
                new_pars = posterior_samples_array[i]

                if np.all(np.isfinite(new_pars)):
                    fit_mod.parameters = new_pars
                    total_flux.append(fit_mod(wave))

            total_flux = np.array([total_flux])
        else:
            total_flux = np.array([np.tile(fit_mod(wave), (self.maxiter, 1))])

        posterior_predictive_samples = {
            "disk_flux": np.array([np.tile(disk_model(wave), (self.maxiter, 1))]),
            "line_flux": np.array([np.tile(line_model(wave), (self.maxiter, 1))]),
            "total_flux": total_flux,
        }

        return self._compose_inference_data(
            config, model, posterior_samples, posterior_predictive_samples
        )

    def _compose_inference_data(
        self,
        config: Config,
        model: LSQModel,
        posterior_samples: dict[str, ArrayLike],
        posterior_predictive_samples: dict[str, ArrayLike],
    ) -> az.InferenceData:
        # "Simulate" multiple chains by reshaping the samples. This is a bit
        # hacky, but it allows us to use ArviZ's functionality for multiple
        # chains without actually running multiple MCMC chains.
        num_chains = 2

        flat_samples = {
            k: v.reshape(num_chains, -1) for k, v in posterior_samples.items()
        }

        idata = az.from_dict(
            posterior=flat_samples,
            posterior_predictive=posterior_predictive_samples,
        )

        return idata
