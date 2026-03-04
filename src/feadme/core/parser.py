import json
from enum import Enum
from pathlib import Path
from typing import Optional

import flax
import flax.struct
import jax.numpy as jnp
from dacite import from_dict, Config as DaciteConfig
from jax.tree_util import tree_map
from jax.typing import ArrayLike
import copy
import loguru
import numpy as np
from functools import cached_property

logger = loguru.logger.opt(colors=True)


def jax_array_hook(value, target_type):
    # If dacite sees a list for a field typed as ArrayLike, convert it
    if issubclass(target_type, jnp.ndarray) and isinstance(value, list):
        return jnp.array(value)
    return value


class Writable:
    """
    A mixin class for objects that can be serialized to JSON.
    """

    def to_json(self, path: str):
        """
        Serialize the object to a JSON file.
        """
        raw = flax.struct.dataclasses.asdict(self)

        serializable = tree_map(
            lambda v: (
                v.tolist()
                if hasattr(v, "tolist")
                else v.value if isinstance(v, Enum) else v
            ),
            # lambda v: v.value if isinstance(v, Enum) else v,
            raw,
        )

        with open(path, "w") as f:
            json.dump(serializable, f, indent=4)

        return serializable

    def to_dict(self):
        """
        Serialize the object to a JSON file.
        """
        raw = flax.struct.dataclasses.asdict(self)

        serializable = tree_map(
            lambda v: (
                v.tolist()
                if hasattr(v, "tolist")
                else v.value if isinstance(v, Enum) else v
            ),
            # lambda v: v.value if isinstance(v, Enum) else v,
            raw,
        )

        return serializable

    @classmethod
    def from_json(cls, path: str | Path):
        """
        Deserialize the object from a JSON file.
        """
        with open(path, "r") as f:
            raw = json.load(f)

        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict):
        """
        Deserialize the object from a dictionary.
        """
        instance = from_dict(
            data_class=cls,
            data=raw,
            config=DaciteConfig(
                type_hooks={
                    ArrayLike: lambda v: jnp.array(v),
                    Distribution: lambda v: Distribution(v),
                    Shape: lambda v: Shape(v),
                }
            ),
        )

        return instance


class Distribution(str, Enum):
    UNIFORM = "uniform"
    LOG_UNIFORM = "log_uniform"
    NORMAL = "normal"
    LOG_NORMAL = "log_normal"
    HALF_NORMAL = "half_normal"
    LOG_HALF_NORMAL = "log_half_normal"


DIST_MAP = {
    Distribution.UNIFORM: 0,
    Distribution.LOG_UNIFORM: 1,
    Distribution.NORMAL: 2,
    Distribution.LOG_NORMAL: 3,
    Distribution.HALF_NORMAL: 4,
    Distribution.LOG_HALF_NORMAL: 5,
}


@flax.struct.dataclass
class Parameter:
    distribution: Distribution = Distribution.UNIFORM
    value: Optional[float] = None
    fixed: Optional[bool] = False
    shared: Optional[str] = None
    low: Optional[float] = None
    high: Optional[float] = None
    loc: Optional[float] = None
    scale: Optional[float] = None
    circular: Optional[bool] = False


@flax.struct.dataclass
class Profile:
    name: Optional[str] = None

    @cached_property
    def independent(self) -> dict[str, Parameter]:
        return {
            field.name: getattr(self, field.name)
            for field in flax.struct.dataclasses.fields(self)
            if isinstance(getattr(self, field.name), Parameter)
            and not getattr(self, field.name).fixed
            and getattr(self, field.name).shared is None
        }

    @cached_property
    def sampler_independent(self) -> dict[str, Parameter]:
        return {f"{self.name}_{k}": v for k, v in self.independent.items()}

    @cached_property
    def shared(self) -> dict[str, Parameter]:
        return {
            field.name: getattr(self, field.name)
            for field in flax.struct.dataclasses.fields(self)
            if isinstance(getattr(self, field.name), Parameter)
            and not getattr(self, field.name).fixed
            and getattr(self, field.name).shared is not None
        }

    @cached_property
    def sampler_shared(self) -> dict[str, Parameter]:
        return {f"{self.name}_{k}": v for k, v in self.shared.items()}

    @cached_property
    def map_shared(self) -> dict[str, str]:
        return {f"{self.name}_{k}": f"{v.shared}_{k}" for k, v in self.shared.items()}

    @cached_property
    def fixed(self) -> dict[str, Parameter]:
        return {
            field.name: getattr(self, field.name)
            for field in flax.struct.dataclasses.fields(self)
            if isinstance(getattr(self, field.name), Parameter)
            and getattr(self, field.name).fixed
        }

    @cached_property
    def sampler_fixed(self) -> dict[str, Parameter]:
        return {f"{self.name}_{k}": v for k, v in self.fixed.items()}


@flax.struct.dataclass
class Disk(Profile, Writable):
    center: Optional[Parameter] = None
    inner_radius: Optional[Parameter] = None
    radius_ratio: Optional[Parameter] = None
    inclination: Optional[Parameter] = None
    sigma: Optional[Parameter] = None
    q: Optional[Parameter] = None
    eccentricity: Optional[Parameter] = None
    apocenter: Optional[Parameter] = None
    area: Parameter = Parameter(distribution=Distribution.UNIFORM, low=0, high=2)
    offset: Parameter = Parameter(distribution=Distribution.UNIFORM, low=0, high=2)


class Shape(str, Enum):
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"


@flax.struct.dataclass
class Line(Profile):
    center: Optional[float] = None
    offset: Optional[Parameter] = None
    area: Optional[Parameter] = None
    vel_width: Optional[Parameter] = None
    shape: Shape = Shape.GAUSSIAN


@flax.struct.dataclass
class Mask:
    lower_limit: float
    upper_limit: float


@flax.struct.dataclass
class Template(Writable):
    name: str = "default_template"
    disk_profiles: list[Disk] = flax.struct.field(default_factory=list)
    line_profiles: list[Line] = flax.struct.field(default_factory=list)
    redshift: Parameter = Parameter(distribution=Distribution.UNIFORM, low=0, high=1.0)
    obs_date: float = 0.0
    white_noise: Parameter = Parameter(
        distribution=Distribution.UNIFORM, low=-10, high=1
    )
    mask: list[Mask] | None = None

    @cached_property
    def parameters(self) -> dict[str, Parameter]:
        params = {"redshift": self.redshift, "white_noise": self.white_noise}

        for prof in self.disk_profiles + self.line_profiles:
            for field in flax.struct.dataclasses.fields(prof):
                field_value = getattr(prof, field.name)
                if isinstance(field_value, Parameter):
                    params[f"{prof.name}_{field.name}"] = field_value

        return params

    @cached_property
    def independent_parameters(self) -> dict[str, Parameter]:
        return {
            k: v for k, v in self.parameters.items() if not v.fixed and v.shared is None
        }

    @cached_property
    def fixed_parameters(self) -> dict[str, Parameter]:
        return {k: v for k, v in self.parameters.items() if v.fixed}

    @cached_property
    def shared_parameters(self) -> dict[str, Parameter]:
        return {k: v for k, v in self.parameters.items() if v.shared is not None}

    @cached_property
    def map_shared_parameters(self) -> dict[str, str]:
        shared_map = {}

        for prof in self.disk_profiles + self.line_profiles:
            shared_map.update(prof.map_shared)

        return shared_map

    @cached_property
    def parameter_names(self) -> list[str]:
        return list(self.parameters.keys())

    @cached_property
    def circular_parameter_names(self) -> list[str]:
        return [k for k, v in self.parameters.items() if v.circular]

    @cached_property
    def fixed_parameter_names(self) -> list[str]:
        return [k for k, v in self.parameters.items() if v.fixed]

    @cached_property
    def shared_parameter_names(self) -> list[str]:
        return [k for k, v in self.parameters.items() if v.shared is not None]

    def fitted_parameter_names(
        self, include_shared=True, include_circ=False
    ) -> list[str]:
        params = []

        for k, v in self.parameters.items():
            if v.fixed:
                continue
            if v.shared is not None and not include_shared:
                continue
            if v.circular and not include_circ:
                continue
            params.append(k)

        return params


@flax.struct.dataclass
class Data(Writable):
    wave: ArrayLike
    flux: ArrayLike
    flux_err: ArrayLike
    mask: ArrayLike
    masked_wave: ArrayLike
    masked_flux: ArrayLike
    masked_flux_err: ArrayLike

    @classmethod
    def create(cls, wave, flux, flux_err, mask=list[Mask] | None):
        mask_array = jnp.ones(len(wave), dtype=bool)

        if mask is not None:
            lower_limits = jnp.array([m.lower_limit for m in mask])
            upper_limits = jnp.array([m.upper_limit for m in mask])

            wave_expanded = wave[:, None]

            individual_masks = (wave_expanded >= lower_limits) & (
                wave_expanded <= upper_limits
            )
            mask_array = jnp.any(individual_masks, axis=1)

        return cls(
            wave=jnp.asarray(wave),
            flux=jnp.asarray(flux),
            flux_err=jnp.asarray(flux_err),
            mask=mask_array,
            masked_wave=jnp.asarray(wave)[mask_array],
            masked_flux=jnp.asarray(flux)[mask_array],
            masked_flux_err=jnp.asarray(flux_err)[mask_array],
        )


@flax.struct.dataclass
class Config(Writable):
    template: Template
    data: Data
    output_path: str
    data_path: str
    template_path: str | None = None
    skip_existing: bool = False
