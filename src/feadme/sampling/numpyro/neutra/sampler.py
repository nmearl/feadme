import time

import arviz as az
import corner
import flax.struct
import jax
import jax.numpy as jnp
import jax.random as random
import loguru
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.typing import ArrayLike
from numpyro.infer import MCMC, NUTS, SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoMultivariateNormal
from numpyro.infer.reparam import NeuTraReparam

from ....core.parser import Config
from ...base_sampler import BaseSampler
from ...initializers import LSQInitializer
from ..model import BaseModel

logger = loguru.logger.opt(colors=True)


@flax.struct.dataclass
class NeuTraSampler(BaseSampler):
    sampler_type = "neutra"

    num_svi_steps: int = 5_000
    svi_learning_rate: float = 5e-3
    svi_decay_rate: float = 0.1
    svi_decay_steps: int = 5_000

    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 1
    target_accept_prob: float = 0.85
    max_tree_depth: int = 10
    dense_mass: bool = False

    @property
    def chain_method(self) -> str:
        return "vectorized" if jax.local_device_count() == 1 else "parallel"

    def __call__(self, config: Config, model: BaseModel) -> az.InferenceData:
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, svi_key, mcmc_key = random.split(rng_key, 3)

        model_kwargs = dict(
            wave=config.data.masked_wave,
            flux=config.data.masked_flux,
            flux_err=config.data.masked_flux_err,
        )

        # LSQ warm-start
        lsq_initializer = LSQInitializer()
        _, init_strategy = lsq_initializer(config, model)

        # Train the SVI guide
        guide = AutoMultivariateNormal(model, init_loc_fn=init_strategy)

        schedule = optax.exponential_decay(
            self.svi_learning_rate, self.svi_decay_steps, self.svi_decay_rate
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=schedule),
        )

        svi = SVI(model, guide, optimizer, Trace_ELBO())
        svi_result = svi.run(
            svi_key,
            num_steps=self.num_svi_steps,
            progress_bar=self.progress_bar,
            stable_update=True,
            **model_kwargs,
        )

        recent_losses = svi_result.losses[-1000:]
        relative_std = jnp.nanstd(recent_losses) / jnp.abs(jnp.nanmean(recent_losses))

        if relative_std > 0.01:
            logger.warning(
                f"SVI may not have converged! Relative std: {relative_std:.4f}. "
                "NeuTra geometry may be suboptimal."
            )
        else:
            logger.info(f"SVI converged. Final loss: {svi_result.losses[-1]:.4f}")

        self._plot_svi(svi_result, guide, config)

        # Build the NeuTra-reparameterized model
        neutra_reparam = NeuTraReparam(guide, svi_result.params)
        reparam_model = neutra_reparam.reparam(model)

        # Run NUTS in the whitened latent space
        kernel = NUTS(
            reparam_model,
            target_accept_prob=self.target_accept_prob,
            max_tree_depth=self.max_tree_depth,
            dense_mass=self.dense_mass,
            find_heuristic_step_size=True,
        )

        mcmc = MCMC(
            kernel,
            num_warmup=self.num_warmup,
            num_samples=self.num_samples,
            num_chains=self.num_chains,
            chain_method=self.chain_method,
            progress_bar=self.progress_bar,
        )

        mcmc.run(mcmc_key, **model_kwargs, extra_fields=("num_steps", "diverging"))

        posterior_samples = mcmc.get_samples(group_by_chain=True)

        extra = mcmc.get_extra_fields()
        num_steps = extra["num_steps"]
        tree_depth = jnp.log2(num_steps).astype(int) + 1
        max_depth = kernel._max_tree_depth

        divergences = extra["diverging"].sum()
        div_rate = 100 * divergences / extra["diverging"].size
        depth_hits = 100 * (tree_depth >= max_depth).mean()

        logger.info(
            f"Treedepth: {depth_hits:.1f}% at max={max_depth} | "
            f"Divergences: {int(divergences)} ({div_rate:.2f}%)"
        )

        return self._compose_inference_data(config, model, mcmc, posterior_samples)

    def _compose_inference_data(
        self,
        config: Config,
        model: BaseModel,
        mcmc: MCMC,
        posterior_samples: dict[str, ArrayLike],
    ) -> az.InferenceData:
        # NeuTraReparam stores whitened draws under a single joint site named
        # "_{model_name}_latent" — these have no physical meaning and should
        # be excluded from the posterior.
        flat_samples = {
            k: v.reshape(-1, *v.shape[2:]) if v.ndim > 2 else v.reshape(-1)
            for k, v in posterior_samples.items()
            if not k.endswith("_latent")
        }

        predictive_post, predictive_prior, log_lik = self._inference_data(
            config, model, flat_samples
        )

        idata = az.from_numpyro(
            mcmc,
            posterior_predictive=predictive_post,
            prior=predictive_prior,
            log_likelihood=log_lik,
        )

        # Drop NeuTra's internal whitened sites from the ArviZ posterior group
        vars_to_drop = [v for v in idata.posterior.data_vars if v.endswith("_latent")]
        if vars_to_drop:
            idata.posterior = idata.posterior.drop_vars(vars_to_drop)

        return idata

    def _plot_svi(self, svi_result, guide, config) -> None:
        rng_key = random.PRNGKey(int(time.time() * 1000) % 2**32)
        guide_samples = guide.sample_posterior(
            rng_key, svi_result.params, sample_shape=(1000,)
        )

        plot_samples = {
            k: v
            for k, v in guide_samples.items()
            if not k.endswith("_flux")
            and not k.endswith("_base")
            and not k.endswith("_raw")
            and np.min(v) != np.max(v)
        }
        guide_mods = {k: jnp.median(v, axis=0) for k, v in guide_samples.items()}

        fig = corner.corner(
            np.array([v for v in plot_samples.values()]).T,
            labels=list(plot_samples.keys()),
            quantiles=[0.16, 0.5, 0.84],
            show_titles=True,
        )
        fig.savefig(f"{config.output_path}/neutra_guide_corner.png")
        plt.close(fig)

        fig, ax = plt.subplots(layout="constrained")
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
            config.data.masked_wave, guide_mods["disk_flux"], label="Disk", color="C3"
        )
        ax.plot(
            config.data.masked_wave, guide_mods["line_flux"], label="Lines", color="C4"
        )
        ax.plot(
            config.data.masked_wave,
            guide_mods["disk_flux"] + guide_mods["line_flux"],
            label="Total",
            color="C5",
        )
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Flux (mJy)")
        ax.set_title(f"NeuTra SVI Guide Fit — {config.template.name}")
        ax.legend()
        fig.savefig(f"{config.output_path}/neutra_svi_fit.png", dpi=300)
        plt.close(fig)
