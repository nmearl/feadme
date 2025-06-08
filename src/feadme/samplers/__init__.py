from typing import Optional, Callable

import numpy as np

from .base_sampler import Sampler, SamplerConfig, DataContainer
from .nuts_sampler import NUTSSampler
from ..parser import Template


# Factory function for easier instantiation
def create_sampler(
    sampler_type: str,
    model: Callable,
    template: Template,
    wave: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    output_dir: str,
    label: str,
    num_warmup: int = 1000,
    num_samples: int = 1000,
    num_chains: int = 1,
    progress_bar: bool = True,
    **kwargs,
) -> Sampler:
    """Factory function to create samplers."""
    data = DataContainer(wave=wave, flux=flux, flux_err=flux_err)
    config = SamplerConfig(
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        progress_bar=progress_bar,
    )

    samplers = {
        "nuts": NUTSSampler,
        # Add other sampler types here
    }

    if sampler_type.lower() not in samplers:
        raise ValueError(f"Unknown sampler type: {sampler_type}")

    return samplers[sampler_type.lower()](
        model=model,
        template=template,
        data=data,
        output_dir=output_dir,
        label=label,
        config=config,
        **kwargs,
    )
