import pickle
from datetime import date
from pathlib import Path
from abc import ABC, abstractmethod

import arviz as az
import blackjax
from blackjax.util import run_inference_algorithm
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.time import Time
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, init_to_median
from numpyro.infer.util import initialize_model, Predictive
from loguru import logger

from .plotting import plot_results
from .utils import dict_to_namedtuple

finfo = np.finfo(float)


def check_convergence(mcmc):
    prior_summary = summary(mcmc.get_samples(), group_by_chain=False)
    pivot_prior_summary = {}

    for k, v in prior_summary.items():
        if "disk_flux" in k or "line_flux" in k or "_base" not in k:
            continue

        pivot_prior_summary.setdefault("param", []).append(k)
        pivot_prior_summary.setdefault("mean", []).append(v["mean"])
        pivot_prior_summary.setdefault("std", []).append(v["std"])
        pivot_prior_summary.setdefault("n_eff", []).append(v["n_eff"])
        pivot_prior_summary.setdefault("r_hat", []).append(v["r_hat"])

    pivot_prior_summary = pd.DataFrame(pivot_prior_summary)
    logger.info(f"Average R_hat values: {pivot_prior_summary['r_hat'].mean()}")

    return 0.99 <= pivot_prior_summary["r_hat"].mean() < 1.01


class Sampler(ABC):
    def __init__(
        self,
        model,
        template,
        wave,
        flux,
        flux_err,
        output_dir,
        label,
        num_warmup,
        num_samples,
        num_chains,
        *,
        rng_key=None,
        progress_bar=True,
    ):
        self._model = model
        self._template = (
            template  # dict_to_namedtuple("NTTemplate", template.model_dump())
        )

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
        self._progress_bar = progress_bar

        self._mcmc = None

        path_exists = Path(f"{output_dir}/{label}.pkl").exists()

        if path_exists:
            logger.info(f"Loading existing MCMC sampler from {output_dir}/{label}.pkl")
            with open(f"{output_dir}/{label}.pkl", "rb") as f:
                self._mcmc = pickle.load(f)

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

    @property
    def converged(self):
        if self._mcmc is None:
            return False

        return check_convergence(self._mcmc)

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
        fit_summary = summary(self.posterior_samples, group_by_chain=False)

        param_dict = {}

        def summarize_posterior(samples):
            summary = {}
            for param, values in samples.items():
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

        param_tab.write(
            f"{self._output_dir}/disk_param_results.csv",
            format="ascii.csv",
            overwrite=True,
        )

    def write_run(self):
        with open(f"{self._output_dir}/{self.label}.pkl", "wb") as f:
            pickle.dump(self._mcmc, f)

    def plot_results(self):
        plot_results(
            self._template,
            self._output_dir,
            self.posterior_predictive_samples,
            self._idata_transformed,
            self._wave,
            self._flux,
            self._flux_err,
            self.label,
        )


class NUTSSampler(Sampler):
    @property
    def posterior_samples(self):
        return {
            k: v
            for k, v in self._mcmc.get_samples().items()
            if "_flux" not in k
            # and "_base" not in k
            # and "_decentered" not in k
            and k not in self.fixed_fields
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
        converged = False
        conv_num = 0

        if self._mcmc is not None:
            converged = check_convergence(self._mcmc)

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
            if self._mcmc is None:
                logger.debug(f"Constructing MCMC for {self._label}.")

                rng_key = self._rng_key

                nuts_kernel = NUTS(
                    self._model,
                    init_strategy=init_strategy,
                    # find_heuristic_step_size=True,
                    dense_mass=dense_mass,
                    # max_tree_depth=20,
                    # adapt_step_size=True,
                    # target_accept_prob=0.9,
                    **kwargs,
                )

                self._mcmc = MCMC(
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
                self._mcmc.post_warmup_state = self._mcmc.last_state
                rng_key = self._mcmc.post_warmup_state.rng_key

            self._mcmc.run(
                rng_key,
                template=self._template,
                wave=jnp.asarray(self._wave),
                flux=jnp.asarray(self._flux),
                flux_err=jnp.asarray(self._flux_err),
            )

            self._mcmc.print_summary()

            converged = check_convergence(self._mcmc)

            if not converged:
                conv_num += 1

                self.plot_results()

                if conv_num % 2 == 0:
                    logger.warning(
                        f"Convergence failed for {self._label} after {conv_num} attempts. "
                        f"Retrying with double the samples."
                    )
                    self._mcmc = None
                    num_warmup *= 2
                    num_samples *= 2
                elif conv_num >= 3:
                    logger.critical(
                        f"Convergence failed for {self._label} after 5 attempts. Skipping."
                    )
                    break

        delta_time = (Time.now() - start_time).to_datetime()

        logger.info(f"Finished sampling {self._label} in {delta_time}.")

        self.write_run()


class NUTSWithAdaptationSampler(Sampler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._posterior_samples = None
        self._posterior_samples_transformed = None
        self._posterior_predictive_samples_transformed = None

    @staticmethod
    def inference_loop(rng_key, kernel, initial_state, num_samples):
        @jax.jit
        def one_step(state, rng_key):
            state, info = kernel(rng_key, state)
            return state, (state, info)

        keys = jax.random.split(rng_key, num_samples)
        _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

        return states, (
            infos.acceptance_rate,
            infos.is_divergent,
            infos.num_integration_steps,
        )

    @property
    def posterior_samples(self):
        return self._posterior_samples_transformed

    @property
    def posterior_predictive_samples(self):
        return self._posterior_predictive_samples_transformed

    def sample(self):
        model_kwargs = {
            "template": self._template,
            "wave": jnp.asarray(self._wave),
            "flux": jnp.asarray(self._flux),
            "flux_err": jnp.asarray(self._flux_err),
        }

        rng_key, init_key = jax.random.split(self._rng_key)
        init_params, potential_fn_gen, postprocess_fn, *_ = initialize_model(
            init_key,
            self._model,
            model_args=tuple(model_kwargs.values()),
            dynamic_args=True,
        )

        logdensity_fn = lambda position: -potential_fn_gen(*model_kwargs.values())(
            position
        )
        initial_position = init_params.z

        adapt = blackjax.window_adaptation(
            blackjax.nuts, logdensity_fn, target_acceptance_rate=0.8
        )

        rng_key, warmup_key = jax.random.split(rng_key)
        (last_state, parameters), _ = adapt.run(
            warmup_key, initial_position, self._num_warmup
        )
        kernel = blackjax.nuts(logdensity_fn, **parameters)

        rng_key, sample_key = jax.random.split(rng_key)
        states, infos = run_inference_algorithm(
            sample_key, kernel, self._num_samples, last_state, progress_bar=True
        )

        _ = states.position["halpha_disk_center_base"].block_until_ready()

        logger.info(f"Average acceptance rate: {infos[0]:.2f}")
        logger.info(f"There were {100 * infos[1]:.2f}% divergent transitions")

        self._posterior_samples_transformed = jax.vmap(
            postprocess_fn(*model_kwargs.values())
        )(states.position)

        self._posterior_predictive_samples_transformed = Predictive(
            model=self._model,
            posterior_samples=self._posterior_samples_transformed,
        )(rng_key, **model_kwargs)
