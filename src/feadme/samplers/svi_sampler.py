import time

import arviz as az
import jax
import jax.numpy as jnp
import loguru
import matplotlib.pyplot as plt
import numpy as np
import optax
from jax.typing import ArrayLike
from numpyro import optim
from numpyro.infer import SVI, Trace_ELBO, init_to_median
from numpyro.infer.autoguide import AutoBNAFNormal

from .base_sampler import BaseSampler
from ..compose import evaluate_model
from ..models.lsq import lsq_model_fitter

logger = loguru.logger.opt(colors=True)


class SVISampler(BaseSampler):
    """
    Variational inference sampler using AutoBNAFNormal guide.
    """

    def __init__(self, model, config, prior_model=None, **svi_kwargs):
        """
        Parameters
        ----------
        model : Callable
            NumPyro model function
        config : Config
            Configuration object
        prior_model : Callable, optional
            Prior model for generating prior samples
        **svi_kwargs : dict
            SVI settings:
            - num_steps: int (default 50000)
            - learning_rate: float (default 0.001)
            - decay_rate: float (default 0.1)
            - decay_steps: int (default num_steps // 2)
            - num_flows: int (default 2)
            - hidden_factors: list (default [8, 8])
            - num_posterior_samples: int (default 10000)
        """
        super().__init__(model, config, prior_model)
        self.svi_kwargs = svi_kwargs
        self._svi = None
        self._svi_result = None
        self._guide = None

    def sample(self):
        """Run SVI to approximate posterior"""
        # SVI parameters
        num_steps = self.svi_kwargs.get("num_steps", 25000)
        learning_rate = self.svi_kwargs.get("learning_rate", 0.001)
        decay_rate = self.svi_kwargs.get("decay_rate", 0.5)
        decay_steps = self.svi_kwargs.get("decay_steps", int(num_steps * 0.7))
        num_flows = self.svi_kwargs.get("num_flows", 4)
        hidden_factors = self.svi_kwargs.get("hidden_factors", [16, 16])

        logger.info(
            f"Starting SVI with {num_steps} steps, "
            f"{num_flows} flows, hidden_factors={hidden_factors}"
        )

        rng_key = jax.random.PRNGKey(int(time.time() * 1000) % 2**32)
        rng_key, svi_key = jax.random.split(rng_key)

        # Create guide
        self._guide = AutoBNAFNormal(
            self.model,
            hidden_factors=hidden_factors,
            num_flows=num_flows,
            init_loc_fn=init_to_median(),
        )

        # Setup optimizer with learning rate schedule
        schedule = optax.exponential_decay(learning_rate, decay_steps, decay_rate)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=schedule),
        )

        # Create SVI object
        self._svi = SVI(
            self.model,
            self._guide,
            optimizer,
            Trace_ELBO(),
        )

        # Run SVI
        logger.info("Running SVI optimization...")
        self._svi_result = self._svi.run(
            svi_key,
            num_steps,
            template=self.template,
            wave=self.wave,
            flux=self.flux,
            flux_err=self.flux_err,
            progress_bar=self.sampler.progress_bar,
            stable_update=True,
        )

        # Check convergence
        self._check_convergence()

        # Plot convergence
        self._plot_convergence()

        # Sample from guide
        num_posterior_samples = self.svi_kwargs.get("num_posterior_samples", 10000)
        logger.info(f"Sampling {num_posterior_samples} from variational posterior...")

        posterior_samples = self._guide.sample_posterior(
            jax.random.PRNGKey(42),
            self._svi_result.params,
            sample_shape=(num_posterior_samples,),
        )

        # Convert to arviz format
        self._idata = self._convert_to_arviz(posterior_samples)

        logger.info("SVI completed successfully")

    def _check_convergence(self):
        """Check if SVI has converged"""
        recent_losses = self._svi_result.losses[-1000:]
        relative_std = jnp.nanstd(recent_losses) / jnp.abs(jnp.nanmean(recent_losses))

        if relative_std > 0.01:
            logger.warning(
                f"SVI may not have converged! Relative std: {relative_std:.4f}. "
                f"Consider increasing num_steps or adjusting learning rate."
            )
        elif jnp.any(jnp.isnan(recent_losses)):
            logger.error("SVI encountered NaNs in losses. Training failed.")
            # raise ValueError("SVI training produced NaN losses")
        else:
            logger.info(
                f"SVI converged. Final loss: {self._svi_result.losses[-1]:.4f}, "
                f"Relative std: {relative_std:.6f}"
            )

    def _plot_convergence(self):
        """Plot SVI loss convergence"""
        fig, ax = plt.subplots(figsize=(10, 4))

        losses = self._svi_result.losses[1000:]
        ax.plot(losses, alpha=0.6, label="Loss")

        # Plot moving average
        window = min(1000, len(losses) // 10)
        if window > 1:
            moving_avg = jnp.convolve(losses, jnp.ones(window) / window, mode="valid")
            ax.plot(
                jnp.arange(window - 1, len(losses)),
                moving_avg,
                color="red",
                linewidth=2,
                label=f"Moving avg (window={window})",
            )

        ax.set_xlabel("Iteration")
        ax.set_ylabel("ELBO Loss")
        ax.set_title("SVI Convergence")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(f"{self._config.output_path}/svi_convergence.png", dpi=150)
        plt.close(fig)

    def _plot_guide_fit(self, posterior_samples):
        """Plot model fit from guide samples"""
        # Compute median fluxes
        line_flux = jnp.median(posterior_samples["line_flux"], axis=0)
        line_flux = jnp.where(jnp.isfinite(line_flux), line_flux, 0.0)
        disk_flux = jnp.median(posterior_samples["disk_flux"], axis=0)
        disk_flux = jnp.where(jnp.isfinite(disk_flux), disk_flux, 0.0)

        # Get median parameters and evaluate model
        param_mods = {
            k: jnp.median(v)
            for k, v in posterior_samples.items()
            if "_flux" not in k and "_base" not in k
        }

        redshift = param_mods.get("redshift", 0.0)
        rest_wave = self.wave / (1 + redshift)
        q_tot_flux, q_disk_flux, q_line_flux = evaluate_model(
            self.template, rest_wave, param_mods
        )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))

        white_noise = param_mods.get("white_noise", -10.0)
        tot_err = jnp.sqrt(self.flux_err**2 + self.flux**2 * jnp.exp(2 * white_noise))

        ax.errorbar(
            self.wave,
            self.flux,
            yerr=tot_err,
            fmt="o",
            color="grey",
            alpha=0.5,
            label="Data",
        )
        ax.plot(self.wave, line_flux, label="Line Flux (median)", color="C1")
        ax.plot(self.wave, disk_flux, label="Disk Flux (median)", color="C2")
        ax.plot(
            self.wave,
            line_flux + disk_flux,
            label="Total Flux (median)",
            color="C3",
            linewidth=2,
        )

        ax.plot(self.wave, q_line_flux, linestyle="--", color="C1", alpha=0.5)
        ax.plot(self.wave, q_disk_flux, linestyle="--", color="C2", alpha=0.5)
        ax.plot(
            self.wave,
            q_tot_flux,
            linestyle="--",
            color="C3",
            alpha=0.5,
            label="Reconstructed",
        )

        ax.set_xlabel("Wavelength [Å]")
        ax.set_ylabel("Flux [mJy]")
        ax.set_title(f"SVI Guide Fit: {self.template.name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.savefig(f"{self._config.output_path}/svi_model_fit.png", dpi=150)
        plt.close(fig)

    def get_posterior_samples(self, mcmc=None, neutra=None):
        """Get posterior samples from the guide"""
        if self._guide is None or self._svi_result is None:
            raise ValueError("No results available. Run sample() first.")

        num_samples = self.svi_kwargs.get("num_posterior_samples", 10000)
        return self._guide.sample_posterior(
            jax.random.PRNGKey(42),
            self._svi_result.params,
            sample_shape=(num_samples,),
        )

    def _convert_to_arviz(self, posterior_samples):
        """Convert SVI posterior samples to arviz InferenceData"""
        n_samples = len(list(posterior_samples.values())[0])

        # Split into 4 chains
        n_chains = 4
        samples_per_chain = n_samples // n_chains

        # Reshape to (chain, draw) or (chain, draw, dim)
        posterior_dict = {}
        for k, v in posterior_samples.items():
            if not k.endswith("_flux"):
                arr = np.array(v)[: samples_per_chain * n_chains]

                # Handle multi-dimensional parameters
                if arr.ndim == 1:
                    posterior_dict[k] = arr.reshape(n_chains, samples_per_chain)
                elif arr.ndim == 2:
                    # For circular _base parameters with shape (n_samples, 2)
                    posterior_dict[k] = arr.reshape(
                        n_chains, samples_per_chain, arr.shape[-1]
                    )
                else:
                    logger.warning(
                        f"Skipping parameter {k} with unexpected shape {arr.shape}"
                    )

        # Compute derived parameters
        all_param_dicts = []
        for chain in range(n_chains):
            for draw in range(samples_per_chain):
                param_dict = {}
                for k in posterior_dict.keys():
                    # Skip _base parameters (used in reparam, not in evaluate_model)
                    if k.endswith("_base"):
                        continue

                    if posterior_dict[k].ndim == 2:
                        param_dict[k] = float(posterior_dict[k][chain, draw])
                    else:  # Multi-dimensional parameters (shouldn't reach here after filtering)
                        param_dict[k] = posterior_dict[k][chain, draw]

                # Add fixed parameters
                for prof in self.template.disk_profiles + self.template.line_profiles:
                    for param in prof.fixed:
                        param_dict[param.qualified_name] = param.value

                # Add shared parameters
                for prof in self.template.disk_profiles + self.template.line_profiles:
                    for param in prof.shared:
                        param_dict[param.qualified_name] = param_dict[
                            f"{param.shared}_{param.name}"
                        ]

                # Compute outer_radius
                for prof in self.template.disk_profiles:
                    inner_key = f"{prof.name}_inner_radius"
                    scale_key = f"{prof.name}_radius_scale"
                    if inner_key in param_dict and scale_key in param_dict:
                        param_dict[f"{prof.name}_outer_radius"] = 10 ** (
                            np.log10(param_dict[inner_key])
                            + (np.log10(5e4) - np.log10(param_dict[inner_key]))
                            * param_dict[scale_key]
                        )

                # Fixed template-level params
                if self.template.redshift.fixed:
                    param_dict["redshift"] = self.template.redshift.value
                if self.template.white_noise.fixed:
                    param_dict["white_noise"] = self.template.white_noise.value

                all_param_dicts.append(param_dict)

        # Add computed params to posterior
        for key in all_param_dicts[0].keys():
            if key not in posterior_dict:
                values = np.array([p[key] for p in all_param_dicts])
                posterior_dict[key] = values.reshape(n_chains, samples_per_chain)

        # Compute fluxes
        disk_fluxes = []
        line_fluxes = []
        total_fluxes = []

        for param_dict in all_param_dicts:
            redshift = param_dict.get("redshift", 0.0)
            rest_wave = self.wave / (1 + redshift)
            total_flux, disk_flux, line_flux = evaluate_model(
                self.template, rest_wave, param_dict
            )
            disk_fluxes.append(np.array(disk_flux))
            line_fluxes.append(np.array(line_flux))
            total_fluxes.append(np.array(total_flux))

        posterior_dict["disk_flux"] = np.array(disk_fluxes).reshape(
            n_chains, samples_per_chain, -1
        )
        posterior_dict["line_flux"] = np.array(line_fluxes).reshape(
            n_chains, samples_per_chain, -1
        )

        posterior_predictive_dict = {
            "total_flux": np.array(total_fluxes).reshape(
                n_chains, samples_per_chain, -1
            ),
            "disk_flux": posterior_dict["disk_flux"],
            "line_flux": posterior_dict["line_flux"],
        }

        # Log likelihood
        log_likelihood_dict = {
            "total_flux": np.zeros((n_chains, samples_per_chain, len(self.wave)))
        }

        for chain in range(n_chains):
            for draw in range(samples_per_chain):
                param_dict = all_param_dicts[chain * samples_per_chain + draw]
                redshift = param_dict.get("redshift", 0.0)
                rest_wave = self.wave / (1 + redshift)
                total_flux, _, _ = evaluate_model(self.template, rest_wave, param_dict)
                total_flux = np.array(total_flux)

                white_noise = param_dict.get("white_noise", -10.0)
                total_error_sq = np.array(
                    self.flux_err
                ) ** 2 + total_flux**2 * np.exp(2 * white_noise)
                residuals = np.array(self.flux) - total_flux
                log_like_per_point = -0.5 * (
                    residuals**2 / total_error_sq + np.log(2 * np.pi * total_error_sq)
                )
                log_likelihood_dict["total_flux"][chain, draw, :] = log_like_per_point

        # Prior samples
        from numpyro.infer.util import Predictive

        rng_key = jax.random.PRNGKey(0)
        prior_predictive = Predictive(self._prior_model, num_samples=2000)(
            rng_key,
            template=self.template,
            wave=self.wave,
            flux=None,
            flux_err=self.flux_err,
        )

        # Filter valid prior samples
        prior_dict = {}
        valid_indices = []

        for i in range(2000):
            is_valid = True
            for k, v in prior_predictive.items():
                val = v[i]
                if isinstance(val, (np.ndarray, jnp.ndarray)):
                    if not np.all(np.isfinite(val)):
                        is_valid = False
                        break
                else:
                    if not np.isfinite(val):
                        is_valid = False
                        break

            if is_valid:
                valid_indices.append(i)

        # Filter valid prior samples (keep this part)
        for k, v in prior_predictive.items():
            prior_dict[k] = np.array([v[i] for i in valid_indices])

        logger.info(f"Generated {len(valid_indices)}/2000 valid prior samples")

        # Reshape prior samples to (chains, draws) format for arviz
        n_prior_valid = len(valid_indices)
        prior_chains = 4
        prior_draws = n_prior_valid // prior_chains

        # Trim and reshape all prior samples
        for k, v in prior_dict.items():
            arr = np.array(v)[: prior_draws * prior_chains]
            if arr.ndim == 1:
                prior_dict[k] = arr.reshape(prior_chains, prior_draws)
            elif arr.ndim == 2:
                # For flux arrays (n_samples, wavelength)
                prior_dict[k] = arr.reshape(prior_chains, prior_draws, arr.shape[-1])

        # Plot guide fit
        self._plot_guide_fit(posterior_samples)

        # Create InferenceData
        idata = az.from_dict(
            posterior=posterior_dict,
            posterior_predictive=posterior_predictive_dict,
            prior=prior_dict,
            log_likelihood=log_likelihood_dict,
        )

        return idata

    def run(self):
        """Run the sampler"""
        self.sample()
