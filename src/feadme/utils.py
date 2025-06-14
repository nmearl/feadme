from jax.scipy.stats import norm
import jax.numpy as jnp
from collections import namedtuple
import numpy as np

from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform

from numpyro.handlers import trace, seed
import jax
from scipy import stats

from scipy.optimize import minimize_scalar


def dict_to_namedtuple(name, d):
    """
    Recursively convert a nested dictionary into a namedtuple.
    """
    if isinstance(d, list):
        return tuple(dict_to_namedtuple(name, item) for item in d)

    if not isinstance(d, dict):
        return d

    fields = {k: dict_to_namedtuple(k.capitalize(), v) for k, v in d.items()}
    NT = namedtuple(name, fields.keys())
    return NT(**fields)


def truncnorm_ppf(q, loc, scale, lower_limit, upper_limit):
    """
    Compute the percent point function (PPF) of a truncated normal distribution.
    """
    a = (lower_limit - loc) / scale
    b = (upper_limit - loc) / scale

    # Compute CDF bounds
    cdf_a = norm.cdf(a)
    cdf_b = norm.cdf(b)

    # Compute the truncated normal PPF
    return norm.ppf(cdf_a + q * (cdf_b - cdf_a)) * scale + loc


class BaseTenTransform(Transform):
    sign = 1

    # TODO: refine domain/codomain logic through setters, especially when
    # transforms for inverses are supported
    def __init__(self, domain=constraints.real):
        self.domain = domain

    @property
    def codomain(self):
        if self.domain is constraints.ordered_vector:
            return constraints.positive_ordered_vector
        elif self.domain is constraints.real:
            return constraints.positive
        elif isinstance(self.domain, constraints.greater_than):
            return constraints.greater_than(self.__call__(self.domain.lower_bound))
        elif isinstance(self.domain, constraints.interval):
            return constraints.interval(
                self.__call__(self.domain.lower_bound),
                self.__call__(self.domain.upper_bound),
            )
        else:
            raise NotImplementedError

    def __call__(self, x):
        # XXX consider to clamp from below for stability if necessary
        return 10**x

    def _inverse(self, y):
        return jnp.log10(y)

    def log_abs_det_jacobian(self, x, y, intermediates=None):
        return x

    def tree_flatten(self):
        return (self.domain,), (("domain",), dict())

    def __eq__(self, other):
        if not isinstance(other, BaseTenTransform):
            return False
        return self.domain == other.domain


def circular_rhat(samples):
    """
    Estimate circular R-hat using vector average consistency across chains.

    Parameters
    ----------
    samples : ndarray
        Shape (chains, draws), in radians.

    Returns
    -------
    rhat : float
        Circular R-hat value.
    """
    chains, draws = samples.shape

    # Mean resultant vector per chain
    cos_means = np.mean(np.cos(samples), axis=1)
    sin_means = np.mean(np.sin(samples), axis=1)
    R_means = np.sqrt(cos_means**2 + sin_means**2)

    # Mean of mean directions
    mean_cos = np.mean(cos_means)
    mean_sin = np.mean(sin_means)
    R_total = np.sqrt(mean_cos**2 + mean_sin**2)

    # Between-chain variance (how different the chain means are)
    B = chains * (1 - R_total)

    # Within-chain variance (how dispersed each chain is)
    R_within = np.mean(
        [
            np.sqrt(np.mean(np.cos(samples[c])) ** 2 + np.mean(np.sin(samples[c])) ** 2)
            for c in range(chains)
        ]
    )
    W = 1 - R_within

    # Gelman-style circular Rhat
    rhat = np.sqrt((W + B) / W)
    return rhat


def get_sample_sites(model, model_args, model_kwargs, rng_key=jax.random.PRNGKey(0)):
    tr = trace(seed(model, rng_key)).get_trace(*model_args, **model_kwargs)
    return {k: site["fn"] for k, site in tr.items() if site["type"] == "sample"}


def transform_init_values(init_vals, sites):
    """
    Convert user-friendly param values to base values expected by TransformReparam sites.
    """
    base_vals = {}

    for site_name, dist_obj in sites.items():
        if not site_name.endswith("_base"):
            continue  # We only care about reparam sites

        param_name = site_name.removesuffix("_base")
        if param_name not in init_vals:
            continue

        value = init_vals[param_name]

        # Apply inverse of transform to get base value
        if hasattr(dist_obj, "transforms") and dist_obj.transforms:
            base_value = dist_obj.transforms[0].inv(value)
            base_vals[site_name] = base_value
        else:
            base_vals[site_name] = value  # fallback if not transformed

    return base_vals


def circular_median(angles):
    """
    Calculate the circular median of angles in radians.
    Uses the geometric median approach on the unit circle.
    """
    # Convert angles to unit vectors
    x = np.cos(angles)
    y = np.sin(angles)

    def objective(theta):
        # Sum of distances from candidate point to all data points
        candidate_x, candidate_y = np.cos(theta), np.sin(theta)
        distances = np.sqrt((x - candidate_x) ** 2 + (y - candidate_y) ** 2)
        return np.sum(distances)

    # Find the angle that minimizes the sum of distances
    result = minimize_scalar(objective, bounds=(0, 2 * np.pi), method="bounded")
    return result.x


def circular_percentiles(angles, percentiles=[16, 84]):
    """
    Calculate circular percentiles using the circular median as reference.

    Parameters:
    angles: array of angles in radians (0 to 2π)
    percentiles: list of percentiles to calculate (0-100)

    Returns:
    Array of percentile values in radians
    """
    # Get circular median as reference
    median = circular_median(angles)

    # Calculate signed angular distances from median
    def angular_distance(angle, reference):
        diff = angle - reference
        # Wrap to [-π, π]
        return np.arctan2(np.sin(diff), np.cos(diff))

    distances = np.array([angular_distance(angle, median) for angle in angles])

    # Sort distances
    sorted_distances = np.sort(distances)

    # Calculate percentiles
    percentile_distances = np.percentile(sorted_distances, percentiles)

    # Convert back to angles
    percentile_angles = []
    for dist in percentile_distances:
        angle = median + dist
        # Wrap to [0, 2π]
        angle = angle % (2 * np.pi)
        percentile_angles.append(angle)

    return np.array(percentile_angles)


# Usage with your pre-calculated angles
def circular_error_bars(median, p16, p84):
    """
    Calculate error bars from circular median and percentiles.

    Parameters:
    median: circular median in radians
    p16: 16th percentile in radians
    p84: 84th percentile in radians

    Returns:
    err_lo, err_hi: lower and upper error bar magnitudes
    """

    def angular_distance(angle1, angle2):
        """Calculate signed angular distance from angle1 to angle2"""
        diff = angle2 - angle1
        return np.arctan2(np.sin(diff), np.cos(diff))

    # Calculate signed distances
    dist_to_p16 = angular_distance(median, p16)
    dist_to_p84 = angular_distance(median, p84)

    # Error bars are absolute distances
    err_lo = abs(dist_to_p16)  # Distance to 16th percentile (lower error)
    err_hi = abs(dist_to_p84)  # Distance to 84th percentile (upper error)

    return err_lo, err_hi


def parse_circular_parameters(samples):
    """
    Analyze your pre-calculated apocenter angle samples.

    Parameters:
    samples: array of angles in radians (0 to 2π)

    Returns:
    Dictionary with circular mean, median, percentiles, and error bars
    """
    # Calculate circular statistics
    circular_mean = stats.circmean(samples)
    median = circular_median(samples)
    p16, p84 = circular_percentiles(samples, [16, 84])
    err_lo, err_hi = circular_error_bars(median, p16, p84)

    return {
        "circular_mean": circular_mean,
        "circular_median": median,
        "percentile_16th": p16,
        "percentile_84th": p84,
        "err_lo": err_lo,
        "err_hi": err_hi,
    }
