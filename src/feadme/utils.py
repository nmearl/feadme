from jax.scipy.stats import norm
import jax.numpy as jnp
from collections import namedtuple

from numpyro.distributions import constraints
from numpyro.distributions.transforms import Transform


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
        return 10 ** x

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