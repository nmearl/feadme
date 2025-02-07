import jax

from numpyro.infer import Predictive
import matplotlib.pyplot as plt
import astropy.uncertainty as unc

import corner
import arviz as az


def plot_fit(mcmc, model, wave, flux, flux_err, output_dir, rng_key=None):
    rng_key = rng_key if rng_key is not None else jax.random.PRNGKey(0)

    posterior_samples = mcmc.get_samples()
    predictive = Predictive(model, posterior_samples=posterior_samples)
    y_pred = predictive(rng_key, wave=wave, flux=None, flux_err=flux_err)["obs"]

    real_dist = unc.Distribution(y_pred.T)
    lower_lim, median, upper_lim = real_dist.pdf_percentiles([16, 50, 84])

    fig, ax = plt.subplots()

    ax.plot(wave, median)
    ax.fill_between(wave, lower_lim, upper_lim, alpha=0.5)
    ax.plot(wave, flux)

    fig.savefig(f"{output_dir}/model_fit.png")

    return fig, ax


def plot_corner(mcmc, output_dir):
    posterior_samples = mcmc.get_samples()
    idata = az.from_numpyro(mcmc)

    fig = corner.corner(
        posterior_samples,
        var_names=[x for x in idata.posterior.keys() if "_base" not in x],
        labels=[x for x in idata.posterior.keys() if "_base" not in x],
        quantiles=[0.16, 0.5, 0.84],
        smooth=1,
    )

    fig.savefig(f"{output_dir}/corner_plot.png")

    return fig
