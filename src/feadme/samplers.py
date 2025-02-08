import pickle
from datetime import date
from pathlib import Path

import arviz as az
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import optax
from numpyro.infer import MCMC, NUTS, init_to_uniform
from numpyro.infer.util import initialize_model, Predictive
from astropy.table import Table
import corner
import matplotlib.pyplot as plt
import astropy.uncertainty as unc

from .plotting import plot_corner, plot_fit

finfo = np.finfo(float)


def output_results(mcmc, label, output_dir):
    posterior_samples = mcmc.get_samples()
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
        param_dict.setdefault("label", []).append(label)
        param_dict.setdefault("param", []).append(k)
        param_dict.setdefault("value", []).append(v["median"])
        param_dict.setdefault("err_lo", []).append(v["median"] - v["16%"])
        param_dict.setdefault("err_hi", []).append(v["84%"] - v["median"])

    Table(param_dict).write(
        f"{output_dir}/disk_param_results.csv", format="ascii.csv", overwrite=True
    )


def initialize_to_nuts(
    model, wave, flux, flux_err, output_dir, label, num_warmup, num_samples, num_chains
):

    nuts_kernel = NUTS(
        model,
        init_strategy=init_to_uniform(),
        # find_heuristic_step_size=True,
        # dense_mass=True,
        # max_tree_depth=(20, 10),
        # adapt_step_size=True,
        # target_accept_prob=0.9,
    )
    rng_key = jax.random.PRNGKey(0)

    if Path(f"{output_dir}/{label}.pkl").exists():
        with open(f"{output_dir}/{label}.pkl", "rb") as f:
            mcmc = pickle.load(f)
    else:
        mcmc = MCMC(
            nuts_kernel,
            num_warmup=num_warmup,
            num_samples=num_samples,
            num_chains=num_chains,
            # chain_method=(
            #     "vectorized" if jax.local_device_count() == 1 else "parallel"
            # ),
            chain_method="vectorized",
        )

        with numpyro.validation_enabled():
            mcmc.run(
                rng_key,
                wave=jnp.asarray(wave),
                flux=jnp.asarray(flux),
                flux_err=jnp.asarray(flux_err),
            )

        with open(f"{output_dir}/{label}.pkl", "wb") as f:
            pickle.dump(mcmc, f)

    mcmc.print_summary()

    output_results(mcmc, label, output_dir)
    plot_fit(mcmc, model, wave, flux, flux_err, output_dir, rng_key)
    plot_corner(mcmc, output_dir)


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


def initialize_to_chees(
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
    rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
    rng_key, init_key = jax.random.split(rng_key)
    init_params, potential_fn_gen, postprocess_fn, *_ = initialize_model(
        init_key,
        model,
        model_args=(template, wave, flux, flux_err),
        # model_kwargs={"wave": wave, "flux": flux, "flux_err": flux_err},
        dynamic_args=True,
    )

    logdensity_fn = lambda position: -potential_fn_gen(template, wave, flux, flux_err)(
        position
    )
    initial_position = init_params.z

    adapt = blackjax.window_adaptation(
        blackjax.nuts, logdensity_fn, target_acceptance_rate=0.8
    )
    # learning_rate = optax.cosine_decay_schedule(1e-3, 1e4, 1e-5)
    # initial_step_size = 1e-3
    # optim = optax.adam(learning_rate)
    # adapt = blackjax.chees_adaptation(
    #     logdensity_fn, num_chains=1, target_acceptance_rate=0.8
    # )

    rng_key, warmup_key = jax.random.split(rng_key)
    (last_state, parameters), _ = adapt.run(
        warmup_key, initial_position, num_warmup  #  initial_step_size, optim,
    )
    kernel = blackjax.nuts(logdensity_fn, **parameters).step

    rng_key, sample_key = jax.random.split(rng_key)
    states, infos = inference_loop(sample_key, kernel, last_state, num_samples)
    _ = states.position["halpha_disk_center_base"].block_until_ready()

    acceptance_rate = np.mean(infos[0])
    num_divergent = np.mean(infos[1])

    print(f"Average acceptance rate: {acceptance_rate:.2f}")
    print(f"There were {100 * num_divergent:.2f}% divergent transitions")

    # res = {}
    # for prof in template.disk_profiles + template.line_profiles:
    #     for param in prof.independent:
    #         samp_name = f"{prof.name}_{param.name}"
    #         res[samp_name] = param.transform(states.position[f"{samp_name}_base"])

    samples_constrained = jax.vmap(postprocess_fn(template, wave, flux, flux_err))(
        states.position
    )

    idata = az.from_dict(
        posterior={
            k: v[None, ...] for k, v in samples_constrained.items() if "_base" not in k
        }
        # posterior=res
    )

    axes = az.plot_posterior(
        idata,
        var_names=[x for x in idata.posterior.keys()],
    )

    fig = axes.ravel()[0].figure
    fig.tight_layout()
    fig.savefig(
        "/home/nmearl/research/disk_comparison/results/ZTF18aaaotwe/posterior_plot.png"
    )

    axes = az.plot_trace(
        idata,
        var_names=[x for x in idata.posterior.keys()],
    )

    fig = axes.ravel()[0].figure
    fig.tight_layout()
    fig.savefig(
        "/home/nmearl/research/disk_comparison/results/ZTF18aaaotwe/trace_plot.png"
    )

    fig = corner.corner(
        idata.posterior,
        var_names=[x for x in idata.posterior.keys() if "_base" not in x],
        labels=[x for x in idata.posterior.keys() if "_base" not in x],
        quantiles=[0.16, 0.5, 0.84],
        smooth=1,
    )
    fig.savefig(
        "/home/nmearl/research/disk_comparison/results/ZTF18aaaotwe/corner_plot.png"
    )

    predictive = Predictive(model, posterior_samples=samples_constrained)
    y_pred = predictive(
        rng_key, template=template, wave=wave, flux=None, flux_err=flux_err
    )["obs"]

    real_dist = unc.Distribution(y_pred.T)
    lower_lim, median, upper_lim = real_dist.pdf_percentiles([16, 50, 84])

    fig, ax = plt.subplots()

    ax.plot(wave, median)
    ax.fill_between(wave, lower_lim, upper_lim, alpha=0.5)
    ax.plot(wave, flux)

    fig.savefig(
        f"/home/nmearl/research/disk_comparison/results/ZTF18aaaotwe/model_fit.png"
    )
