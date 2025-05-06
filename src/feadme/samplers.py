from abc import ABC, abstractmethod
from datetime import date
from pathlib import Path

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.time import Time
from loguru import logger
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer.util import Predictive
from scipy.stats import gaussian_kde

from .plotting import plot_results
from .utils import circular_rhat

finfo = np.finfo(float)


class Sampler(ABC):
    def __init__(
        self,
        model: callable,
        template: object,
        wave: np.ndarray,
        flux: np.ndarray,
        flux_err: np.ndarray,
        output_dir: str,
        label: str,
        num_warmup: int,
        num_samples: int,
        num_chains: int,
        *,
        rng_key: jax.random.PRNGKey = None,
        progress_bar: bool = True,
    ):
        self._model = model
        self._template = template

        # Construct masks
        mask = [
            np.bitwise_and(wave > m.lower_limit, wave < m.upper_limit)
            for m in template.mask
        ]

        if len(mask) > 1:
            mask = np.bitwise_or(*mask)
        else:
            mask = mask[0]

        self._wave = wave[mask]
        self._flux = flux[mask]
        self._flux_err = flux_err[mask]
        self._output_dir = output_dir
        self._label = label
        self._mjd = template.mjd
        self._redshift = template.redshift
        self._num_warmup = num_warmup
        self._num_samples = num_samples
        self._num_chains = num_chains
        self._rng_key = (
            rng_key
            if rng_key is not None
            else jax.random.key(int(date.today().strftime("%Y%m%d")))
        )
        self._rng_keys = jax.random.split(self._rng_key, num_chains)
        self._progress_bar = progress_bar

        self._idata = None

        path_exists = Path(f"{output_dir}/{label}.nc").exists()

        if path_exists:
            logger.info(f"Loading existing MCMC sampler from {output_dir}/{label}.nc")
            self._idata = az.from_netcdf(f"{output_dir}/{label}.nc")

    @property
    @abstractmethod
    def posterior_samples(self):
        raise NotImplementedError("Posterior samples not implemented.")

    @property
    @abstractmethod
    def posterior_predictive_samples(self):
        raise NotImplementedError("Posterior predictive samples not implemented.")

    @abstractmethod
    def sample(self):
        raise NotImplementedError("Sampling not implemented.")

    @property
    def label(self):
        return self._label

    @property
    def mjd(self):
        return self._mjd

    @property
    def redshift(self):
        return self._redshift

    def _fitting_summary(self):
        if self._idata is None:
            return None

        prior_summary = summary(self._idata.to_dict()["posterior"], group_by_chain=True)
        pivot_prior_summary = {}

        for k, v in prior_summary.items():
            if (
                "disk_flux" in k
                or "line_flux" in k
                or "_base" in k
                or "_wrap" in k
                or k in self.fixed_fields
                or k.endswith("apocenter")
            ):
                continue

            pivot_prior_summary.setdefault("param", []).append(k)
            pivot_prior_summary.setdefault("mean", []).append(v["mean"])
            pivot_prior_summary.setdefault("std", []).append(v["std"])
            pivot_prior_summary.setdefault("n_eff", []).append(v["n_eff"])

            # if k.endswith("apocenter"):
            #     pivot_prior_summary.setdefault("r_hat", []).append(
            #         circular_rhat(self._idata.posterior[k].values)
            #     )
            #     pass
            # else:
            pivot_prior_summary.setdefault("r_hat", []).append(v["r_hat"])

        pivot_prior_summary = pd.DataFrame(pivot_prior_summary)

        return pivot_prior_summary

    def _results_summary(self):
        fit_summary = self._fitting_summary().set_index("param").to_dict(orient="index")

        param_dict = {}

        def summarize_posterior(samples):
            summary = {}
            for param, values in samples.items():
                if param.endswith("apocenter"):
                    apo_mean = np.arctan2(
                        np.mean(np.sin(values)), np.mean(np.cos(values))
                    ) % (2 * np.pi)
                    apo_rot = (values - apo_mean + np.pi) % (2 * np.pi)
                    lower_bound_rot = np.percentile(apo_rot, 16)
                    upper_bound_rot = np.percentile(apo_rot, 84)
                    lower_bound = (lower_bound_rot - np.pi + apo_mean) % (2 * np.pi)
                    upper_bound = (upper_bound_rot - np.pi + apo_mean) % (2 * np.pi)
                    # kde = gaussian_kde(apo_rot)
                    # x_grid = np.linspace(0, 2 * np.pi, 1000)
                    # rot_median = x_grid[np.argmax(kde(x_grid))]
                    # median = (rot_median - np.pi + apo_mean) % (2 * np.pi)
                    median = apo_mean
                else:
                    median = jnp.median(values)
                    lower_bound = jnp.percentile(values, 16)
                    upper_bound = jnp.percentile(values, 86)

                summary[param] = {
                    "median": median,
                    "16%": lower_bound,
                    "84%": upper_bound,
                }
            return summary

        posterior_summary = summarize_posterior(self.posterior_samples)

        for k, v in posterior_summary.items():
            if "disk_flux" in k or "line_flux" in k or "_base" in k:
                continue

            param_dict.setdefault("label", []).append(self.label)
            param_dict.setdefault("mjd", []).append(self.mjd)
            param_dict.setdefault("redshift", []).append(self.redshift)
            param_dict.setdefault("param", []).append(k)
            param_dict.setdefault("value", []).append(v["median"])
            param_dict.setdefault("err_lo", []).append(v["median"] - v["16%"])
            param_dict.setdefault("err_hi", []).append(v["84%"] - v["median"])
            param_dict.setdefault("n_eff", []).append(
                fit_summary.get(f"{k}", {"n_eff": np.nan})["n_eff"]
            )
            param_dict.setdefault("r_hat", []).append(
                fit_summary.get(f"{k}", {"r_hat": np.nan})["r_hat"]
            )

        param_tab = Table(param_dict)

        return param_tab

    def check_convergence(self, show=True):
        pivot_prior_summary = self._fitting_summary()

        if pivot_prior_summary is None:
            return False

        if show:
            logger.info(f"Fitting complete: displaying results... \n{pivot_prior_summary.to_markdown()}")

        est_r_hat = pivot_prior_summary["r_hat"].mean()

        logger.info(f"Average R_hat values: {est_r_hat}")

        return 0.99 <= est_r_hat < 1.01

    @property
    def _idata_transformed(self):
        return az.from_dict(
            posterior={
                k: np.expand_dims(a=np.asarray(v), axis=0)
                for k, v in self.posterior_samples.items()
            },
            posterior_predictive={
                k: np.expand_dims(a=np.asarray(v), axis=0)
                for k, v in self.posterior_predictive_samples.items()
            },
        )

    @property
    def fixed_fields(self):
        return [
            f"{prof.name}_{param.name}"
            for prof in self._template.disk_profiles + self._template.line_profiles
            for param in prof.fixed
        ]

    def write_results(self):
        """Write the results of the MCMC run to a CSV file."""
        param_tab = self._results_summary()

        param_tab.write(
            f"{self._output_dir}/disk_param_results.csv",
            format="ascii.csv",
            overwrite=True,
        )

    def write_run(self):
        """Write the MCMC run to a netcdf file."""
        az.to_netcdf(
            self._idata,
            f"{self._output_dir}/{self.label}.nc",
        )

    def plot_results(self):
        """Plot the results of the MCMC run."""
        plot_results(
            self._template,
            self._output_dir,
            self._idata,
            self._idata_transformed,
            self._results_summary(),
            self._wave,
            self._flux,
            self._flux_err,
            self.label,
        )


class NUTSSampler(Sampler):
    @property
    def posterior_samples(self):
        return {
            var: np.array(self._idata.posterior[var]).reshape(
                -1, *self._idata.posterior[var].shape[2:]
            )
            for var in self._idata.posterior.data_vars
        }

    @property
    def posterior_predictive_samples(self):
        return Predictive(
            model=self._model,
            posterior_samples=self.posterior_samples,
        )(
            jax.random.PRNGKey(0),
            template=self._template,
            wave=self._wave,
            flux=None,
            flux_err=self._flux_err,
        )

    def sample(self, init_strategy=init_to_median, dense_mass=True, **kwargs):
        mcmc = None
        converged = False
        conv_num = 0

        if self._idata is not None:
            converged = self.check_convergence(show=False)
            self.plot_results()

        chain_method = "vectorized" if jax.local_device_count() == 1 else "parallel"

        logger.info(
            f"Fitting {self._label} using the `{chain_method}` method with "
            f"`{jax.local_device_count()}` local devices and "
            f"`{self._num_chains}` chains."
        )

        start_time = Time.now()

        num_warmup = self._num_warmup
        num_samples = self._num_samples

        while not converged:
            if mcmc is None:
                logger.debug(f"Constructing MCMC for {self._label}.")

                rng_keys = self._rng_keys

                nuts_kernel = NUTS(
                    self._model,
                    init_strategy=init_strategy,
                    # find_heuristic_step_size=True,
                    dense_mass=dense_mass,
                    # max_tree_depth=20,
                    # adapt_step_size=True,
                    target_accept_prob=0.9,
                    **kwargs,
                )

                mcmc = MCMC(
                    nuts_kernel,
                    num_warmup=num_warmup,
                    num_samples=num_samples,
                    num_chains=self._num_chains,
                    chain_method=chain_method,
                    progress_bar=self._progress_bar,
                )
            else:
                logger.info(
                    f"R_hat values are not converged. Re-running MCMC ({conv_num})"
                )
                mcmc.post_warmup_state = mcmc.last_state
                rng_keys = mcmc.post_warmup_state.rng_key

            mcmc.run(
                rng_keys,
                template=self._template,
                wave=jnp.asarray(self._wave),
                flux=jnp.asarray(self._flux),
                flux_err=jnp.asarray(self._flux_err),
            )

            # mcmc.print_summary()
            self._idata = az.from_numpyro(mcmc)

            converged = self.check_convergence()

            if not converged:
                conv_num += 1

                self.plot_results()
                self.write_results()

                if conv_num % 2 == 0:
                    logger.warning(
                        f"Convergence failed for {self._label} after {conv_num} attempts. "
                        f"Retrying with double the samples."
                    )
                    mcmc = None
                    num_warmup *= 2
                    num_samples *= 2
                elif conv_num >= 3:
                    logger.critical(
                        f"Convergence failed for {self._label} after 5 attempts. Skipping."
                    )
                    break

        self.plot_results()
        self.write_results()

        delta_time = (Time.now() - start_time).to_datetime()

        logger.info(f"Finished sampling {self._label} in {delta_time}.")
