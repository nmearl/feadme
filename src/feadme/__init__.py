import numpyro

numpyro.set_host_device_count(4)
numpyro.enable_x64()

import jax

print(f"""
JAX Stats:
    Backend: {jax.lib.xla_bridge.get_backend().platform}
    Device count: {jax.device_count()}
    Device: {jax.devices()[0].device_kind}
    Local device count: {jax.local_device_count()}
""")