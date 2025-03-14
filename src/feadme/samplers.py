import pickle
from datetime import date
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

from .plotting import plot_corner, plot_fit

finfo = np.finfo(float)


def output_results(mcmc, posterior_samples, label, output_dir):
    fit_summary = summary(mcmc.get_samples(), group_by_chain=False)

    param_dict = {}

    def summarize_posterior(samples):
        summary = {}
        for param, values in samples.items():
            median = jnp.median(values)
            lower_bound = jnp.percentile(values, 16)
            upper_bound = jnp.percentile(values, 86)
            summary[param] = {"median": median, "16%": lower_bound, "84%": upper_bound}
        return summary

    posterior_summary = summarize_posterior(posterior_samples)

    for k, v in posterior_summary.items():
        if "disk_flux" in k or "line_flux" in k:
            continue

        param_dict.setdefault("label", []).append(label)
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

    Table(param_dict).write(
        f"{output_dir}/disk_param_results.csv", format="ascii.csv", overwrite=True
    )


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
    return 0.99 <= pivot_prior_summary["r_hat"].mean() < 1.01


def initialize_to_nuts(
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
):

    nuts_kernel = NUTS(
        model,
        init_strategy=init_to_median(),
        # find_heuristic_step_size=True,
        dense_mass=True,
        # max_tree_depth=20,
        # adapt_step_size=True,
        # target_accept_prob=0.9,
    )
    rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
    converged = False
    conv_num = 0
    path_exists = Path(f"{output_dir}/{label}.pkl").exists()

    if path_exists:
        with open(f"{output_dir}/{label}.pkl", "rb") as f:
            mcmc = pickle.load(f)

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
                num_warmup=num_warmup,
                num_samples=num_samples,
                num_chains=num_chains,
                chain_method=(
                    "vectorized" if jax.local_device_count() == 1 else "parallel"
                ),
                # chain_method="vectorized",
            )

        with numpyro.validation_enabled():
            mcmc.run(
                rng_key,
                template=template,
                wave=jnp.asarray(wave),
                flux=jnp.asarray(flux),
                flux_err=jnp.asarray(flux_err),
            )

        with open(f"{output_dir}/{label}.pkl", "wb") as f:
            pickle.dump(mcmc, f)

        mcmc.print_summary()
        converged = check_convergence(mcmc)
        conv_num += 1

    # output_results(mcmc, label, output_dir)
    # plot_fit(mcmc, model, wave, flux, flux_err, output_dir, rng_key)
    # plot_corner(mcmc, output_dir)

    posterior_samples = mcmc.get_samples()

    posterior_predictive_samples_transformed = Predictive(
        model=model,
        posterior_samples=posterior_samples,
    )(rng_key, template=template, wave=wave, flux=None, flux_err=None)

    fixed_fields = [
        f"{prof.name}_{param.name}"
        for prof in template.disk_profiles + template.line_profiles
        for param in prof.fixed
    ]

    idata_transformed = az.from_dict(
        posterior={
            k: np.expand_dims(a=np.asarray(v), axis=0)
            for k, v in posterior_samples.items()
            if k not in fixed_fields
        },
        posterior_predictive={
            k: np.expand_dims(a=np.asarray(v), axis=0)
            for k, v in posterior_predictive_samples_transformed.items()
            if k not in fixed_fields
        },
    )

    output_results(mcmc, posterior_predictive_samples_transformed, label, output_dir)

    plot_results(
        output_dir,
        posterior_predictive_samples_transformed,
        idata_transformed,
        wave,
        flux,
        flux_err,
        label,
    )


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


def plot_results(
    output_dir,
    posterior_predictive_samples_transformed,
    idata_transformed,
    wave,
    flux,
    flux_err,
    label,
):
    # axes = az.plot_trace(
    #     idata_transformed,
    #     var_names=[x for x in idata_transformed.posterior.keys() if "_base" not in x],
    #     compact=True,
    #     backend_kwargs={"layout": "constrained"},
    # )
    #
    # fig = axes.ravel()[0].figure
    # fig.tight_layout()
    # fig.savefig(f"{output_dir}/trace_plot.png")

    axes = az.plot_trace(
        idata_transformed,
        var_names=[x for x in idata_transformed.posterior.keys() if "_base" in x],
        compact=True,
        backend_kwargs={"layout": "constrained"},
    )

    fig = axes.ravel()[0].figure
    fig.tight_layout()
    fig.savefig(f"{output_dir}/base_trace_plot.png")

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")

    ax.plot(wave, flux)
    az.plot_hdi(
        ax=ax,
        x=wave,
        y=idata_transformed["posterior_predictive"]["obs"],
        fill_kwargs={"alpha": 0.5},
        color="C1",
    )
    fig.savefig(f"{output_dir}/hdi_plot.png")

    fig, ax = plt.subplots()
    ax.errorbar(
        wave,
        flux,
        yerr=flux_err,
        fmt="o",
        color="grey",
        # markeredgecolor="grey",
        # ecolor="grey",
        zorder=-10,
        alpha=0.25,
    )

    for var in ["disk_flux", "line_flux"]:
        var_dist = unc.Distribution(posterior_predictive_samples_transformed[var].T)
        _, median, _ = var_dist.pdf_percentiles([16, 50, 84])
        ax.plot(wave, median, label=f"{var}")

    real_dist = unc.Distribution(posterior_predictive_samples_transformed["obs"].T)
    lower_lim, median, upper_lim = real_dist.pdf_percentiles([16, 50, 84])
    # lower_lim = median - lower_lim
    # upper_lim = upper_lim - median

    ax.plot(wave, median, color="C3", label="Model Fit")
    ax.fill_between(wave, lower_lim, upper_lim, alpha=0.5, color="C3")

    ax.set_ylabel("Flux [mJy]")
    ax.set_xlabel("Wavelength [AA]")
    ax.set_title(f"{label} Model Fit")

    ax.legend()
    fig.savefig(f"{output_dir}/model_fit.png")

    fig = corner.corner(
        idata_transformed,
        var_names=[
            x
            for x in idata_transformed.posterior.keys()
            if "_base" not in x and "_flux" not in x
        ],
        labels=[
            x
            for x in idata_transformed.posterior.keys()
            if "_base" not in x and "_flux" not in x
        ],
        quantiles=[0.16, 0.5, 0.84],
        smooth=1,
        show_titles=True,
    )
    fig.savefig(f"{output_dir}/corner_plot.png")
