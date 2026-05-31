from typing import Callable
import flax.struct
from ..core.parser import Config
from ..core.parser import Shape, Distribution, Parameter
from ..core.integrators import mixed_jax_integrate


@flax.struct.dataclass
class BaseModel:
    config: Config
    integrator: Callable = mixed_jax_integrate

    def setup(self, *args, **kwargs):
        return self

    def __call__(self, *args, **kwargs):
        raise NotImplementedError
