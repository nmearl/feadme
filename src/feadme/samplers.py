import pickle
from datetime import date
from functools import cached_property
from pathlib import Path

import arviz as az
import astropy.uncertainty as unc
import blackjax
import corner
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import numpyro
import optax
from astropy.table import Table
from numpyro.infer import MCMC, NUTS, init_to_uniform, init_to_median
from numpyro.infer.util import initialize_model, Predictive
from blackjax.util import run_inference_algorithm
from numpyro.diagnostics import summary
import pandas as pd
from .plotting import plot_results

finfo = np.finfo(float)


def check_convergence(mcmc):
    prior_summary = summary(mcmc.get_samples(), group_by_chain=False)
    pivot_prior_summary = {}

    for k, v in prior_summary.items():
        if "disk_flux" in k or "line_flux" in k or "base" not in k:
            continue

        pivot_prior_summary.setdefault("param", []).append(k)
        pivot_prior_summary.setdefault("mean", []).append(v["mean"])
        pivot_prior_summary.setdefault("std", []).append(v["std"])
        pivot_prior_summary.setdefault("n_eff", []).append(v["n_eff"])
        pivot_prior_summary.setdefault("r_hat", []).append(v["r_hat"])

    pivot_prior_summary = pd.DataFrame(pivot_prior_summary)
    print(f"Average R_hat values: {pivot_prior_summary['r_hat'].mean()}")

    return 0.99 <= pivot_prior_summary["r_hat"].mean() < 1.01


class Sampler:
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
        self._num_warmup = num_warmup
        self._num_samples = num_samples
        self._num_chains = num_chains
        self._rng_key = (
            rng_key
            if rng_key is not None
            else jax.random.key(int(date.today().strftime("%Y%m%d")))
        )
        self._mcmc = None

        path_exists = Path(f"{output_dir}/{label}.pkl").exists()

        if path_exists:
            with open(f"{output_dir}/{label}.pkl", "rb") as f:
                self._mcmc = pickle.load(f)

        # _, self._potential_fn_gen, self._postprocess_fn, *_ = initialize_model(
        #     self._rng_key,
        #     model,
        #     model_args=(template, wave, flux, flux_err),
        #     dynamic_args=True,
        # )

    def sample(self):
        pass

    @property
    def label(self):
        return self._label

    @cached_property
    def posterior_samples(self):
        return {k: v for k, v in self._mcmc.get_samples().items() if "base" in k}

    @cached_property
    def posterior_predictive_samples(self):
        return Predictive(
            model=self._model,
            posterior_samples=self.posterior_samples,
        )(
            jax.random.PRNGKey(0),
            template=self._template,
            wave=self._wave,
            flux=None,
            flux_err=None,
        )

    # @cached_property
    # def posterior_samples_transformed(self):
    #     return {
    #         k: v
    #         for k, v in self._mcmc.get_samples().items()
    #         if "base" not in k and "flux" not in k and "obs" not in k
    #     }

    @cached_property
    def posterior_samples_transformed(self):
        return jax.vmap(
            self._postprocess_fn(self._template, self._wave, self._flux, self._flux_err)
        )(self.posterior_samples)

    @cached_property
    def posterior_predictive_samples_transformed(self):
        return Predictive(
            model=self._model,
            posterior_samples=self.posterior_samples_transformed,
        )(
            jax.random.PRNGKey(0),
            template=self._template,
            wave=self._wave,
            flux=None,
            flux_err=None,
        )

    @cached_property
    def _idata_transformed(self):
        return az.from_dict(
            posterior={
                k: np.expand_dims(a=np.asarray(v), axis=0)
                for k, v in self.posterior_samples_transformed.items()
            },
            posterior_predictive={
                k: np.expand_dims(a=np.asarray(v), axis=0)
                for k, v in self.posterior_predictive_samples_transformed.items()
            },
        )

    @cached_property
    def fixed_fields(self):
        return [
            f"{prof.name}_{param.name}"
            for prof in self._template.disk_profiles + self._template.line_profiles
            for param in prof.fixed
        ]

    def summary(self, write=True):
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

        posterior_summary = summarize_posterior(self.posterior_samples_transformed)

        for k, v in posterior_summary.items():
            if "disk_flux" in k or "line_flux" in k:
                continue

            param_dict.setdefault("label", []).append(self.label)
            param_dict.setdefault("param", []).append(k)
            param_dict.setdefault("value", []).append(v["median"])
            param_dict.setdefault("err_lo", []).append(v["median"] - v["16%"])
            param_dict.setdefault("err_hi", []).append(v["84%"] - v["median"])
            param_dict.setdefault("n_eff", []).append(
                fit_summary.get(f"{k}_base", {"n_eff": np.nan})["n_eff"]
            )
            param_dict.setdefault("r_hat", []).append(
                fit_summary.get(f"{k}_base", {"r_hat": np.nan})["r_hat"]
            )

        param_tab = Table(param_dict)
        param_tab.pprint_all()

        if write:
            param_tab.write(
                f"{self._output_dir}/disk_param_results.csv",
                format="ascii.csv",
                overwrite=True,
            )

    def write_results(self):
        with open(f"{self._output_dir}/{self.label}.pkl", "wb") as f:
            pickle.dump(self._mcmc, f)

        self.summary()

    def plot_results(self):
        plot_results(
            self._output_dir,
            self.posterior_predictive_samples_transformed,
            self._idata_transformed,
            self._wave,
            self._flux,
            self._flux_err,
            self.label,
        )


class NUTSSampler(Sampler):
    def sample(self, init_strategy=init_to_median(), dense_mass=True, **kwargs):
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
        rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

        converged = False
        conv_num = 0
        mcmc = self._mcmc

        if mcmc is not None:
            converged = check_convergence(mcmc)

        if converged:
            return

        # if not path_exists or not converged:
        while not converged:
            if not converged and conv_num > 0:
                print(f"R_hat values are not converged. Re-running MCMC ({conv_num})")
                mcmc.post_warmup_state = mcmc.last_state
                rng_key = mcmc.post_warmup_state.rng_key

            # if not path_exists:
            if not converged and conv_num == 0:
                mcmc = MCMC(
                    nuts_kernel,
                    num_warmup=self._num_warmup,
                    num_samples=self._num_samples,
                    num_chains=self._num_chains,
                    chain_method=(
                        "vectorized" if jax.local_device_count() == 1 else "parallel"
                    ),
                )

            mcmc.run(
                rng_key,
                template=self._template,
                wave=jnp.asarray(self._wave),
                flux=jnp.asarray(self._flux),
                flux_err=jnp.asarray(self._flux_err),
            )

            self._mcmc = mcmc
            self.write_results()
            mcmc.print_summary()

            converged = check_convergence(mcmc)
            converged = True
            conv_num += 1


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


def nuts_with_adaptation(
    model,
    template,
    wave,
    flux,
    flux_err,
    output_dir,
    label,
    num_warmup,
    num_samples,
    num_chains=1,
    learning_rate=0.5,
    initial_step_size=0.1,
):
    save_location = f"{output_dir}/{label}_save.pkl"
    rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

    if Path(save_location).exists():
        with open(save_location, "rb") as f:
            states, infos, postprocess_fn, template, wave, flux, flux_err = pickle.load(
                f
            )
    else:
        rng_key, init_key = jax.random.split(rng_key)
        init_params, potential_fn_gen, postprocess_fn, *_ = initialize_model(
            init_key,
            model,
            model_args=(template, wave, flux, flux_err),
            dynamic_args=True,
        )

        logdensity_fn = lambda position: -potential_fn_gen(
            template, wave, flux, flux_err
        )(position)
        initial_position = init_params.z

        adapt = blackjax.window_adaptation(
            blackjax.nuts, logdensity_fn, target_acceptance_rate=0.8, progress_bar=True
        )

        rng_key, warmup_key = jax.random.split(rng_key)
        (last_state, parameters), _ = adapt.run(
            warmup_key, initial_position, num_warmup
        )
        kernel = blackjax.nuts(logdensity_fn, **parameters).step

        rng_key, sample_key = jax.random.split(rng_key)
        states, infos = inference_loop(sample_key, kernel, last_state, num_samples)
        _ = states.position["halpha_disk_center_base"].block_until_ready()

        # Save the outputs
        with open(f"{output_dir}/{label}_save.pkl", "wb") as f:
            pickle.dump(
                (states, infos, postprocess_fn, template, wave, flux, flux_err), f
            )

    acceptance_rate = np.mean(infos[0])
    num_divergent = np.mean(infos[1])

    print(f"Average acceptance rate: {acceptance_rate:.2f}")
    print(f"There were {100 * num_divergent:.2f}% divergent transitions")

    # samples_constrained = {}
    #
    # for i in range(num_samples):
    #     pp_res = postprocess_fn(template, wave, flux, flux_err)(
    #         {k: v[i] for k, v in states.position.items()}
    #     )
    #
    #     for k, v in pp_res.items():
    #         samples_constrained.setdefault(k, []).append(v)
    #
    # samples_constrained = {k: np.array(v) for k, v in samples_constrained.items()}

    posterior_samples_transformed = jax.vmap(
        postprocess_fn(template, wave, flux, flux_err)
    )(states.position)

    posterior_predictive_samples_transformed = Predictive(
        model=model,
        posterior_samples=posterior_samples_transformed,
    )(rng_key, template=template, wave=wave, flux=None, flux_err=None)

    fixed_fields = [
        f"{prof.name}_{param.name}"
        for prof in template.disk_profiles + template.line_profiles
        for param in prof.fixed
    ]

    idata_transformed = az.from_dict(
        posterior={
            k: np.expand_dims(a=np.asarray(v), axis=0)
            for k, v in posterior_samples_transformed.items()
            if k not in fixed_fields
        },
        posterior_predictive={
            k: np.expand_dims(a=np.asarray(v), axis=0)
            for k, v in posterior_predictive_samples_transformed.items()
            if k not in fixed_fields
        },
    )

    plot_results(
        output_dir,
        posterior_predictive_samples_transformed,
        idata_transformed,
        wave,
        flux,
        flux_err,
        label,
    )


def nuts_with_adaptation_multi(
    model,
    template,
    wave,
    flux,
    flux_err,
    output_dir,
    label,
    num_warmup,
    num_samples,
    num_chains=2,
):
    num_devices = jax.local_device_count()  # Number of available devices
    if num_chains % num_devices != 0:
        raise ValueError(
            f"num_chains ({num_chains}) must be a multiple of num_devices ({num_devices}) for pmap."
        )

    rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

    _, potential_fn_gen, postprocess_fn, *_ = initialize_model(
        rng_key,
        model,
        model_args=(template, wave, flux, flux_err),
        dynamic_args=True,
    )

    def logdensity_fn(position):
        return -potential_fn_gen(template, wave, flux, flux_err)(position)

    adapt = blackjax.window_adaptation(
        blackjax.nuts, logdensity_fn, target_acceptance_rate=0.9, progress_bar=False
    )

    rng_keys = jax.random.split(rng_key, num_chains)

    def single_chain_run(init_key):
        """Runs NUTS sampling for a single chain."""
        init_params, _, _, *_ = initialize_model(
            init_key,
            model,
            model_args=(template, wave, flux, flux_err),
            dynamic_args=True,
        )

        initial_position = init_params.z

        rng_key, warmup_key = jax.random.split(init_key)

        (last_state, parameters), _ = adapt.run(
            warmup_key, initial_position, num_warmup
        )

        kernel = blackjax.nuts(logdensity_fn, **parameters).step

        rng_key, sample_key = jax.random.split(rng_key)
        states, infos = inference_loop(sample_key, kernel, last_state, num_samples)

        return states.position  # Return sampled positions

    # Convert inputs into per-device arrays
    # replicated_keys = jnp.array(rng_keys).reshape(
    #     num_devices, num_chains // num_devices, -1
    # )

    # Run multiple chains in parallel across devices
    results = jax.pmap(single_chain_run, in_axes=(0,))(rng_keys)

    # Ensure computation completes before returning
    _ = results["halpha_disk_center_base"].block_until_ready()

    # Save the outputs
    # with open(f"{output_dir}/{label}_save.pkl", "wb") as f:
    #     pickle.dump((states, infos, postprocess_fn, template, wave, flux, flux_err), f)

    # acceptance_rate = np.mean(infos[0])
    # num_divergent = np.mean(infos[1])
    #
    # print(f"Average acceptance rate: {acceptance_rate:.2f}")
    # print(f"There were {100 * num_divergent:.2f}% divergent transitions")

    # samples_constrained = {}
    #
    # for i in range(num_samples):
    #     pp_res = postprocess_fn(template, wave, flux, flux_err)(
    #         {k: v[i] for k, v in states.position.items()}
    #     )
    #
    #     for k, v in pp_res.items():
    #         samples_constrained.setdefault(k, []).append(v)
    #
    # samples_constrained = {k: np.array(v) for k, v in samples_constrained.items()}

    # print(results)

    results = {k: v.flatten() for k, v in results.items()}

    posterior_samples_transformed = jax.vmap(
        postprocess_fn(template, wave, flux, flux_err)
    )(results)

    posterior_predictive_samples_transformed = Predictive(
        model=model,
        posterior_samples=posterior_samples_transformed,
    )(rng_key, template=template, wave=wave, flux=None, flux_err=None)

    idata_transformed = az.from_dict(
        posterior={
            k: np.expand_dims(a=np.asarray(v), axis=0)
            for k, v in posterior_samples_transformed.items()
        },
        posterior_predictive={
            k: np.expand_dims(a=np.asarray(v), axis=0)
            for k, v in posterior_predictive_samples_transformed.items()
        },
    )

    plot_results(
        output_dir,
        posterior_predictive_samples_transformed,
        idata_transformed,
        wave,
        flux,
        flux_err,
        label,
    )
