import time

import flax.struct
import jax.numpy as jnp
import jax.random as random
import loguru
import optax
import corner
import jax
from astropy.modeling.fitting import (
    TRFLSQFitter,
    model_to_fit_params,
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
from numpyro.infer.util import log_density
import numpy as np
import astropy.constants as const
import astropy.units as u

from .lsq.utils import format_posterior_samples
from ..core.parser import Config
from .base_model import BaseModel
from .lsq.model import _compose_model
from .utils import diagnose_init_params

logger = loguru.logger.opt(colors=True)

c_kms = const.c.to(u.km / u.s).value


def _strip_flux_outputs(sample_dict: dict) -> dict:
    return {k: v for k, v in sample_dict.items() if not k.endswith("_flux")}


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

    # Score a subset — finding the argmax doesn't need the full sample set.
    num_draws = min(200, int(guide_samples[candidate_names[0]].shape[0]))
    best_logp = -np.inf
    best_idx = None

    model_args = ()
    model_kwargs = {
        "wave": config.data.masked_wave,
        "flux": config.data.masked_flux,
        "flux_err": config.data.masked_flux_err,
    }

    for idx in range(num_draws):
        params = {name: guide_samples[name][idx] for name in candidate_names}
        logp, _ = log_density(model, model_args, model_kwargs, params)
        logp_val = float(jax.device_get(logp))

        if np.isfinite(logp_val) and logp_val > best_logp:
            best_logp = logp_val
            best_idx = idx

    if best_idx is None:
        raise ValueError("No finite SVI guide sample had a finite model log density.")

    best_sample = {name: guide_samples[name][best_idx] for name in candidate_names}
    return best_sample, best_logp, best_idx


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
    def __call__(self, config: Config = None, model: BaseModel = None):
        wave = config.data.masked_wave
        flux = config.data.masked_flux
        flux_err = config.data.masked_flux_err
        safe_weights = np.where(
            np.isfinite(flux_err) & (flux_err > 0), 1.0 / flux_err, 0.0
        )

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
            maxiter=10_000,
            filter_non_finite=True,
        )

        init_params = format_posterior_samples(fit_mod, wave, flux, flux_err)

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

        n_sf = max(1, len(config.template.disk_profiles))

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

            mask_arr = []

            for j in range(len(config.template.mask)):
                mask_arr.append(
                    (config.data.masked_wave > config.template.mask[j].lower_limit)
                    & (config.data.masked_wave < config.template.mask[j].upper_limit)
                )

            mask = np.logical_or.reduce(mask_arr)

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
    num_steps: int = 2_000
    learning_rate: float = 5e-3
    decay_rate: float = 0.1
    decay_steps: int = 2_000  # ~num_steps
    progress_bar: bool = False
    num_samples: int = 1000

    def __call__(self, config: Config, model: BaseModel):
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, svi_key, sample_key = random.split(rng_key, 3)

        lsq_initializer = LSQInitializer(debug_plot=self.debug_plot)
        init_params, init_strategy = lsq_initializer(config, model)

        diagnose_init_params(model, init_params, config)

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
                f"SVI may not have converged! Relative std: {relative_std:.4f}"
            )
        else:
            logger.info(f"SVI converged. Final loss: {svi_result.losses[-1]:.4f}")

        # Draw samples from the fitted guide and take the median as the single
        # init point. Use the single best coherent guide sample under the
        # actual model log density rather than componentwise medians, which
        # can land in a low-density region when parameters are correlated.
        guide_samples = guide.sample_posterior(
            sample_key,
            svi_result.params,
            sample_shape=(self.num_samples,),
        )

        lsq_params = init_params

        try:
            init_params, best_logp, best_idx = _best_guide_sample_by_log_density(
                model,
                guide_samples,
                config,
            )
            logger.info(
                "Using best SVI guide sample for initialization "
                f"(sample {best_idx}, log_density={best_logp:.4f})"
            )
        except Exception as exc:
            logger.warning(
                "Falling back to LSQ initialization because SVI sample scoring "
                f"failed: {exc}"
            )
            init_params = lsq_params

        init_strategy = init_to_value(values=init_params)

        if self.debug_plot:
            self.quick_plot(
                svi_result,
                guide,
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

        fig = corner.corner(
            np.array([v for k, v in plot_samples.items() if k not in ignored]).T,
            labels=[k for k, v in plot_samples.items() if k not in ignored],
            # truths=[starters.get(k, None) for k in guide_samples],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
        )
        fig.savefig(f"{config.output_path}/guide_corner_plot.png")

        n_sf = max(1, len(config.template.disk_profiles))

        fig, axes = plt.subplots(
            1,
            n_sf,
            layout="constrained",
            figsize=(10 * n_sf, 5),
        )

        for i in range(n_sf):
            ax = axes.flat[i] if hasattr(axes, "flat") else axes

            mask_arr = []

            for j in range(len(config.template.mask)):
                mask_arr.append(
                    (config.data.masked_wave > config.template.mask[j].lower_limit)
                    & (config.data.masked_wave < config.template.mask[j].upper_limit)
                )

            mask = np.logical_or.reduce(mask_arr)

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
            ax.legend()

        fig.savefig(f"{config.output_path}/svi_fit.png")
        plt.close(fig)
