from jax.scipy.stats import norm
import numpyro.distributions as dist
import jax.numpy as jnp


def truncnorm_ppf(q, loc, scale, lower_limit, upper_limit):
    a = (lower_limit - loc) / scale
    b = (upper_limit - loc) / scale

    # Compute CDF bounds
    cdf_a = norm.cdf(a)
    cdf_b = norm.cdf(b)

    # Compute the truncated normal PPF
    return norm.ppf(cdf_a + q * (cdf_b - cdf_a)) * scale + loc


# Define a custom transform that clips the value within [log10(a), log10(b)]
class TruncatedAffineTransform(dist.transforms.AffineTransform):
    def __init__(self, loc, scale, lower, upper, *args, **kwargs):
        super().__init__(loc, scale, *args, **kwargs)
        self.lower = lower
        self.upper = upper

    def __call__(self, x):
        # Apply the affine transform
        x_transformed = self.loc + self.scale * x
        # Clip the value to stay within bounds
        return jnp.clip(x_transformed, self.lower, self.upper)


# Build re-parameterization configuration more efficiently
#     reparam_config = {}
#
#     for prof in all_profiles:
#         for param in prof.independent:
#             samp_name = f"{prof.name}_{param.name}"
#
#             if param.distribution in [
#                 Distribution.log_uniform,
#                 Distribution.log_normal,
#             ]:
#                 reparam_config[samp_name] = TransformReparam()
#             elif param.distribution == Distribution.normal:
#                 reparam_config[samp_name] = LocScaleReparam(centered=0)
#
#     # Sample parameters with optimized reparam
#     with numpyro.handlers.reparam(config=reparam_config):
#         for prof in all_profiles:
#             for param in prof.independent:
#                 samp_name = f"{prof.name}_{param.name}"
#
#                 if param.distribution == Distribution.uniform:
#                     param_mods[samp_name] = numpyro.sample(
#                         samp_name, dist.Uniform(param.low, param.high)
#                     )
#                 elif param.distribution == Distribution.log_uniform:
#                     # Create distribution using properly normalized base distribution
#                     base_dist = dist.Uniform(
#                         jnp.log10(param.low), jnp.log10(param.high)
#                     )
#                     transforms = [
#                         dist.transforms.PowerTransform(10.0),
#                     ]
#                     log_unif_param_dist = dist.TransformedDistribution(
#                         base_dist, transforms
#                     )
#                     param_mods[samp_name] = numpyro.sample(
#                         samp_name, log_unif_param_dist
#                     )
#
#                 elif param.distribution == Distribution.normal:
#                     param_mods[samp_name] = numpyro.sample(
#                         samp_name,
#                         dist.TruncatedNormal(
#                             param.loc, param.scale, low=param.low, high=param.high
#                         ),
#                     )
#                 elif param.distribution == Distribution.log_normal:
#                     # Create distribution using properly normalized base distribution
#                     base_dist = dist.Normal(0, 1)
#                     transforms = [
#                         dist.transforms.AffineTransform(param.loc, param.scale),
#                         log10_transform,
#                         exp_transform,
#                     ]
#                     log_norm_param_dist = dist.TransformedDistribution(
#                         base_dist, transforms
#                     )
#                     param_mods[samp_name] = numpyro.sample(
#                         samp_name, log_norm_param_dist
#                     )
#                 else:
#                     raise ValueError(
#                         f"Invalid distribution: {param.distribution} for parameter {param.name}"
#                     )
#
#             # Sample all shared parameters
#             for param in prof.shared:
#                 samp_name = f"{prof.name}_{param.name}"
#                 param_mods[samp_name] = numpyro.deterministic(
#                     samp_name, param_mods[f"{param.shared}_{param.name}"]
#                 )
#
#             # Include fixed fields
#             for param in prof.fixed:
#                 param_mods[f"{prof.name}_{param.name}"] = numpyro.deterministic(
#                     f"{prof.name}_{param.name}", param.value
#                 )
