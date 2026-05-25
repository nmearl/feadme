import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.distributions import constraints


class BoundedLogNormal(dist.Distribution):
    arg_constraints = {}
    reparametrized_params = ["loc", "scale"]

    def __init__(self, loc, scale, low, high, *, validate_args=None):
        self.loc = jnp.asarray(loc)
        self.scale = jnp.asarray(scale)
        self.low = jnp.asarray(low)
        self.high = jnp.asarray(high)
        self._log_low = jnp.log(self.low)
        self._log_high = jnp.log(self.high)
        self._base_dist = dist.TruncatedNormal(
            self.loc,
            self.scale,
            low=self._log_low,
            high=self._log_high,
        )
        super().__init__(self._base_dist.batch_shape, validate_args=validate_args)

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.low, self.high)

    def sample(self, key, sample_shape=()):
        return jnp.exp(self._base_dist.sample(key, sample_shape))

    def log_prob(self, value):
        log_value = jnp.log(value)
        log_prob = self._base_dist.log_prob(log_value) - log_value
        in_bounds = (value >= self.low) & (value <= self.high)
        return jnp.where(in_bounds, log_prob, -jnp.inf)
