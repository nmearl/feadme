from jax.scipy.stats import norm


def truncnorm_ppf(u, mu, sigma, a, b):
    """JAX-compatible PPF for truncated normal using NumPy's scipy.stats.norm.ppf."""
    Fa = norm.cdf((a - mu) / sigma)  # CDF at lower bound
    Fb = norm.cdf((b - mu) / sigma)  # CDF at upper bound

    # Transform uniform sample to truncated normal sample
    return mu + sigma * norm.ppf(Fa + u * (Fb - Fa))
