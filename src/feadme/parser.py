import numpy as np
import numpyro
from pydantic import (
    BaseModel,
    root_validator,
    model_validator,
    field_validator,
    ConfigDict,
    computed_field,
)
from enum import Enum, auto
from typing import Optional, List, Callable, NamedTuple
import jax.numpy as jnp
from collections import namedtuple
from jax.scipy import stats
from .utils import truncnorm_ppf
from functools import cached_property


class Distribution(str, Enum):
    uniform = "uniform"
    log_uniform = "log_uniform"
    normal = "normal"
    log_normal = "log_normal"
    half_normal = "half_normal"


class Parameter(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    name: str
    distribution: Distribution = Distribution.uniform
    value: Optional[float] = None
    fixed: Optional[bool] = False
    tied: Optional[Callable] = None
    shared: Optional[str] = None
    low: Optional[float] = None
    high: Optional[float] = None
    loc: Optional[float] = None
    scale: Optional[float] = None

    @model_validator(mode="before")
    def validate_parameters(cls, values):
        dist = values.get("distribution")

        if dist in ["uniform", "log_uniform"]:
            if values.get("low") is None or values.get("high") is None:
                raise ValueError(
                    "For 'uniform' distribution, 'low' and 'high' parameters are required."
                )
        elif dist == "normal":
            missing_params = [
                param for param in ("loc", "scale") if values.get(param) is None
            ]
            if missing_params:
                raise ValueError(
                    f"For 'normal' distribution, {', '.join(missing_params)} are required."
                )
        elif dist == "log_normal":
            if values.get("loc") is None or values.get("scale") is None:
                raise ValueError(
                    "For 'lognormal' distribution, 'loc' and 'scale' parameters are required."
                )
        elif dist == "half_normal":
            if values.get("scale") is None:
                raise ValueError(
                    "For 'half_normal' distribution, 'scale' parameter is required."
                )

        return values


class Profile(BaseModel):
    name: str

    @cached_property
    def _parameter_fields(self):
        return [
            getattr(self, field_name)
            for field_name in self.model_fields
            if isinstance(getattr(self, field_name), Parameter)
        ]

    def _shared(self):
        return [
            field
            for field in self._parameter_fields
            if field.shared is not None and not field.fixed
        ]

    @cached_property
    def shared(self) -> List[Parameter]:
        return self._shared()

    def _independent(self):
        return [
            field
            for field in self._parameter_fields
            if field.shared is None and not field.fixed
        ]

    @cached_property
    def independent(self) -> List[Parameter]:
        return self._independent()

    def _fixed(self):
        return [field for field in self._parameter_fields if field.fixed]

    @cached_property
    def fixed(self) -> List[Parameter]:
        return self._fixed()


class Mask(BaseModel):
    lower_limit: float
    upper_limit: float


class Disk(Profile):
    center: Parameter
    inner_radius: Parameter
    delta_radius: Parameter
    inclination: Parameter
    sigma: Parameter
    q: Parameter
    eccentricity: Parameter
    apocenter: Parameter
    scale: Optional[Parameter] = Parameter(
        name="scale", distribution=Distribution.uniform, low=0, high=2
    )
    offset: Optional[Parameter] = Parameter(
        name="offset", distribution=Distribution.uniform, low=0, high=0.1
    )


class Shape(str, Enum):
    gaussian = "gaussian"
    lorentzian = "lorentzian"


class Line(Profile):
    shape: Optional[Shape] = Shape.gaussian
    center: Parameter
    amplitude: Parameter
    vel_width: Parameter


class Template(BaseModel):
    name: str
    data_path: Optional[str] = None
    mjd: Optional[int] = None
    redshift: Optional[float] = 0
    disk_profiles: List[Disk]
    line_profiles: List[Line]
    white_noise: Optional[Parameter] = Parameter(
        name="white_noise",
        distribution=Distribution.normal,
        low=0,
        high=0.5,
        loc=0,
        scale=0.1,
    )
    mask: Optional[List[Mask]] = []

    @cached_property
    def all_profiles(self) -> List[Profile]:
        return self.disk_profiles + self.line_profiles

    @field_validator("disk_profiles", mode="after")
    @classmethod
    def validate_disk_profiles(cls, value):
        disk_names = [disk.name for disk in value]
        if len(disk_names) != len(set(disk_names)):
            raise ValueError("Disk profile names must be unique.")
        return value

    @field_validator("line_profiles", mode="after")
    @classmethod
    def validate_line_profiles(cls, value):
        line_names = [line.name for line in value]
        if len(line_names) != len(set(line_names)):
            raise ValueError("Line profile names must be unique.")
        return value
