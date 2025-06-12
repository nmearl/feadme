from loguru import logger
import numpyro
import sys
import os

logger.remove()

# logger.add(
#     "{time}.log",
#     colorize=False,
#     format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z}</green> | "
#     "<level>{level: <8}</level> | "
#     "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
# )

logger.add(
    sys.stderr,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>",
)

numpyro.set_host_device_count(int(os.getenv("FORCE_DEVICE_COUNT", "1")))
numpyro.enable_x64()

import jax

logger.debug("Initializing...")
logger.debug(f"Backend: {jax.lib.xla_bridge.get_backend().platform}")
logger.debug(f"Device count: {jax.device_count()}")
logger.debug(f"Device: {jax.devices()[0].device_kind}")
logger.debug(f"Local device count: {jax.local_device_count()}")
