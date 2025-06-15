import json
from enum import Enum
from pathlib import Path
from typing import Optional

import flax
import flax.struct
import jax
import jax.numpy as jnp
from dacite import from_dict, Config as DaciteConfig
from jax.tree_util import tree_map


def jax_array_hook(value, target_type):
    # If dacite sees a list for a field typed as jnp.ndarray, convert it
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
            lambda v: v.tolist() if hasattr(v, "tolist") else v,
            raw,
        )

        with open(path, "w") as f:
            json.dump(serializable, f, indent=4)

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
                    jnp.ndarray: lambda v: jnp.array(v),
                    Distribution: lambda v: Distribution(v),
                    Shape: lambda v: Shape(v),
                }
            ),
        )

        # Process the instance to populate parameter lists
        return cls._process_profiles(instance)

    @classmethod
    def _process_profiles(cls, instance):
        """Process all Profile instances in the object tree"""
        if isinstance(instance, Profile):
            return instance.populate_param_lists()
        elif hasattr(instance, "__dataclass_fields__"):
            # Handle dataclass instances
            updates = {}
            for field_name, field in instance.__dataclass_fields__.items():
                field_value = getattr(instance, field_name)
                if isinstance(field_value, list):
                    # Process lists of profiles
                    processed_list = [
                        cls._process_profiles(item) for item in field_value
                    ]
                    if processed_list != field_value:
                        updates[field_name] = processed_list
                elif isinstance(field_value, Profile):
                    # Process single profile
                    processed_profile = cls._process_profiles(field_value)
                    if processed_profile != field_value:
                        updates[field_name] = processed_profile

            if updates:
                return instance.replace(**updates)

        return instance


class Distribution(str, Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"
    LOG_UNIFORM = "log_uniform"
    LOG_NORMAL = "log_normal"


@flax.struct.dataclass
class Parameter:
    name: str
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

    # Computed parameter lists
    _independent_params: list[Parameter] = flax.struct.field(default_factory=list)
    _shared_params: list[Parameter] = flax.struct.field(default_factory=list)
    _fixed_params: list[Parameter] = flax.struct.field(default_factory=list)

    def populate_param_lists(self):
        """
        Populate parameter lists - returns a new instance with populated lists.
        This should be called immediately after deserialization.
        """
        if self._independent_params or self._shared_params or self._fixed_params:
            return self  # Already populated

        # Get all Parameter fields from this instance
        param_kwargs = {}
        for field_name in self.__dataclass_fields__:
            if not field_name.startswith("_") and field_name != "name":
                field_value = getattr(self, field_name)
                if isinstance(field_value, Parameter):
                    param_kwargs[field_name] = field_value

        independent = []
        shared = []
        fixed = []

        # First pass: categorize fixed vs non-fixed parameters
        for field_name, field_value in param_kwargs.items():
            if field_value.fixed:
                fixed.append(field_value)
            else:
                independent.append(field_value)

        # Second pass: handle shared parameters
        shared_candidates = []
        for field_name, field_value in param_kwargs.items():
            if field_value.shared is not None:
                shared_candidates.append(field_value)

        for shared_param in shared_candidates:
            # if shared_param.shared in [p.name for p in independent]:
            if shared_param in independent:
                independent.remove(shared_param)
            shared.append(shared_param)

        return self.replace(
            _independent_params=independent, _shared_params=shared, _fixed_params=fixed
        )

    @classmethod
    def create(cls, name, **kwargs):
        param_kwargs = {k: v for k, v in kwargs.items() if not k.startswith("_")}

        independent = []
        shared = []
        fixed = []

        for field_name, field_value in param_kwargs.items():
            if isinstance(field_value, Parameter):
                if field_value.fixed:
                    fixed.append(field_value)
                else:
                    independent.append(field_value)

        for field_name, field_value in param_kwargs.items():
            if isinstance(field_value, Parameter):
                if field_value.shared is not None:
                    if field_value.shared in [p.name for p in independent]:
                        shared.append(field_value)
                    else:
                        independent.append(field_value)

        return cls(
            name=name,
            _independent_params=independent,
            _shared_params=shared,
            _fixed_params=fixed,
            **param_kwargs,
        )

    @property
    def independent(self) -> list[Parameter]:
        return self._independent_params

    @property
    def shared(self) -> list[Parameter]:
        return self._shared_params

    @property
    def fixed(self) -> list[Parameter]:
        return self._fixed_params


@flax.struct.dataclass
class Disk(Profile, Writable):
    center: Optional[Parameter] = None
    inner_radius: Optional[Parameter] = None
    delta_radius: Optional[Parameter] = None
    inclination: Optional[Parameter] = None
    sigma: Optional[Parameter] = None
    q: Optional[Parameter] = None
    eccentricity: Optional[Parameter] = None
    apocenter: Optional[Parameter] = None
    scale: Parameter = Parameter(
        name="scale", distribution=Distribution.UNIFORM, low=0, high=2
    )
    offset: Parameter = Parameter(
        name="offset", distribution=Distribution.UNIFORM, low=0, high=2
    )


class Shape(str, Enum):
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"


@flax.struct.dataclass
class Line(Profile):
    center: Optional[Parameter] = None
    amplitude: Optional[Parameter] = None
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
    redshift: float = 0.0
    obs_date: float = 0.0
    white_noise: Parameter = Parameter(
        name="white_noise", distribution=Distribution.UNIFORM, low=0, high=0.1
    )
    mask: list[Mask] | None = None


@flax.struct.dataclass
class Data(Writable):
    wave: jnp.ndarray
    flux: jnp.ndarray
    flux_err: jnp.ndarray
    mask: jnp.ndarray
    masked_wave: jnp.ndarray
    masked_flux: jnp.ndarray
    masked_flux_err: jnp.ndarray

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
class Sampler(Writable):
    sampler_type: str
    num_warmup: int = 1000
    num_samples: int = 1000
    num_chains: int = 1
    progress_bar: bool = True
    # TODO: Currently only NUTS is supported
    target_accept_prob: float = 0.8
    max_tree_depth: int = 10
    dense_mass: bool = True

    @property
    def chain_method(self) -> str:
        return "vectorized" if jax.local_device_count() == 1 else "parallel"


@flax.struct.dataclass
class Config(Writable):
    template: Template
    data: Data
    sampler: Sampler
    output_path: str
    template_path: str
    data_path: str
