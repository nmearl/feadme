from jax.scipy.stats import norm


def truncnorm_ppf(q, loc, scale, lower_limit, upper_limit):
    a = (lower_limit - loc) / scale
    b = (upper_limit - loc) / scale

    # Compute CDF bounds
    cdf_a = norm.cdf(a)
    cdf_b = norm.cdf(b)

    # Compute the truncated normal PPF
    return norm.ppf(cdf_a + q * (cdf_b - cdf_a)) * scale + loc
