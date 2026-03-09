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
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer import init_to_median, init_to_value
from numpyro.infer.autoguide import (
    AutoBNAFNormal,
    AutoMultivariateNormal,
)
import numpy as np
import astropy.constants as const
import astropy.units as u
from numpyro.infer.initialization import init_to_value
from pygments.unistring import Lm

from .lsq.utils import format_posterior_samples
from ..core.parser import Config
from .base_model import BaseModel
from .lsq.model import _compose_model
from .utils import diagnose_init_params

logger = loguru.logger.opt(colors=True)

c_kms = const.c.to(u.km / u.s).value


@flax.struct.dataclass
class BaseInitializer:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("Call method must be implemented by subclasses.")


@flax.struct.dataclass
class DefaultInitializer(BaseInitializer):
    def __call__(self, config: Config = None, model: BaseModel = None):
        return init_to_median(num_samples=1000)


@flax.struct.dataclass
class LSQInitializer(BaseInitializer):
    def __call__(self, config: Config = None, model: BaseModel = None):
        wave = config.data.masked_wave
        flux = config.data.masked_flux
        flux_err = config.data.masked_flux_err

        lsq_model = _compose_model(config.template, integrator=model.integrator)

        fit_mod = TRFLSQFitter()(
            lsq_model,
            wave,
            flux,
            # weights=1 / flux_err,
            maxiter=10_000,
            filter_non_finite=True,
        )

        init_params = format_posterior_samples(fit_mod, wave, flux, flux_err)

        init_strategy = init_to_value(values=init_params)

        # Quick plot
        self.quick_plot(fit_mod, config, init_params)

        return init_params, init_strategy

    def quick_plot(self, fit_mod, config, init_params):
        fig, ax = plt.subplots(layout="constrained")

        ax.errorbar(
            config.data.masked_wave / (1 + init_params["redshift"]),
            config.data.masked_flux,
            yerr=config.data.masked_flux_err,
            fmt="o",
            color="grey",
            zorder=-10,
            alpha=0.25,
        )

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

        new_wave = np.linspace(
            config.data.masked_wave[0], config.data.masked_wave[-1], 1000
        )
        tot_flux = fit_mod(new_wave)
        disk_flux = disk_model(new_wave)
        narrow_line_flux = narrow_line_model(new_wave)
        broad_line_flux = broad_line_model(new_wave)

        ax.plot(
            new_wave / (1 + init_params["redshift"]),
            tot_flux,
            label="LSQ Fit",
            color="C3",
        )
        ax.plot(
            new_wave / (1 + init_params["redshift"]),
            disk_flux,
            label="Disk Fit",
            color="C4",
        )
        ax.plot(
            new_wave / (1 + init_params["redshift"]),
            narrow_line_flux,
            label="Narrow Line Fit",
            color="C5",
            linestyle="--",
        )
        ax.plot(
            new_wave / (1 + init_params["redshift"]),
            broad_line_flux,
            label="Broad Line Fit",
            color="C6",
            linestyle="--",
        )

        # Print the initial parameter estimates on the figure
        param_text = "\n".join(
            [
                f"{k:25} {v:.5g}"
                for k, v in init_params.items()
                if np.isfinite(v) and "_base" not in k
            ]
        )
        ax.text(
            0.05,
            0.95,
            param_text,
            transform=ax.transAxes,
            fontsize=6,
            verticalalignment="top",
            fontfamily="monospace",
        )

        ax.set_title(f"LSQ Initial Fit for {config.template.name}")
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Flux (mJy)")

        ax.legend()
        fig.savefig(f"{config.output_path}/lsq_fit.png", dpi=300)
        plt.close(fig)


@flax.struct.dataclass
class SVIInitializer(BaseInitializer):
    num_steps: int = 2_000
    learning_rate: float = 5e-3
    decay_rate: float = 0.1
    decay_steps: int = 2_000  # ~num_steps
    progress_bar: bool = True
    num_samples: int = 100

    def __call__(self, config: Config, model: BaseModel):
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, svi_key, sample_key = random.split(rng_key, 3)

        lsq_initializer = LSQInitializer()
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
            stable_update=True,
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
        # init point. More robust than the raw LSQ point because the guide has
        # been optimized against the full likelihood, not just the data peak.
        guide_samples = guide.sample_posterior(
            sample_key,
            svi_result.params,
            sample_shape=(1000,),
        )

        init_params = {
            k: jnp.median(v, axis=0)
            for k, v in guide_samples.items()
            if not k.endswith("_area")  # filter any guide-internal flux sites
        }

        init_strategy = init_to_value(values=init_params)

        # self.quick_plot(svi_result, guide, config)

        return init_params, init_strategy

    def quick_plot(self, svi_result, guide, config):
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
            and np.min(v) != np.max(v)
        }
        guide_mods = {k: jnp.median(v, axis=0) for k, v in guide_samples.items()}

        fig = corner.corner(
            np.array([v for k, v in plot_samples.items()]).T,
            labels=[k for k, v in plot_samples.items()],
            # truths=[starters.get(k, None) for k in guide_samples],
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
        )
        fig.savefig(f"{config.output_path}/guide_corner_plot.png")

        fig, axes = plt.subplots(
            nrows=len(plot_samples),
            ncols=2,
            figsize=(10, 3 * len(plot_samples)),
            layout="constrained",
        )

        import arviz as az

        az.plot_trace(
            az.from_dict(plot_samples),
            var_names=[k for k in plot_samples],
            axes=axes,
        )

        # tot_flux, disk_flux, line_flux = evaluate_model(
        #     config.template, new_wave, param_mods
        # )

        fig.savefig(f"{config.output_path}/guide_trace_plot.png")

        fig, ax = plt.subplots()

        ax.errorbar(
            config.data.masked_wave,
            config.data.masked_flux,
            yerr=config.data.masked_flux_err,
            fmt="o",
            color="grey",
            zorder=-10,
            alpha=0.25,
        )

        ax.plot(
            config.data.masked_wave,
            guide_mods["disk_flux"],
            label="Disk Fit",
            color="C3",
        )
        ax.plot(
            config.data.masked_wave,
            guide_mods["line_flux"],
            label="Line Fit",
            color="C4",
        )
        ax.plot(
            config.data.masked_wave,
            guide_mods["disk_flux"] + guide_mods["line_flux"],
            label="Total Fit",
            color="C5",
        )
        ax.legend()
        fig.savefig(f"{config.output_path}/svi_fit.png", dpi=300)
        plt.close(fig)
