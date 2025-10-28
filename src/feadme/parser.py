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

        # Process the instance to populate parameter lists and names
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

    # Internal fields set during Profile initialization
    _field_name: Optional[str] = None
    _qualified_name: Optional[str] = None

    def with_names(self, field_name: str, qualified_name: str):
        """Return a new Parameter with both field name and qualified name set."""
        return self.replace(_field_name=field_name, _qualified_name=qualified_name)

    @property
    def name(self) -> str:
        """Get the parameter's field name."""
        if self._field_name is None:
            raise ValueError(
                "Parameter name not set. This parameter hasn't been properly initialized in a Profile."
            )
        return self._field_name

    @property
    def qualified_name(self) -> str:
        """Get the fully qualified parameter name (profile_name_field_name)."""
        if self._qualified_name is None:
            raise ValueError(
                "Parameter qualified name not set. This parameter hasn't been properly initialized in a Profile."
            )
        return self._qualified_name


@flax.struct.dataclass
class Profile:
    name: Optional[str] = None

    # Computed parameter lists
    _independent_params: list[Parameter] = flax.struct.field(default_factory=list)
    _shared_params: list[Parameter] = flax.struct.field(default_factory=list)
    _fixed_params: list[Parameter] = flax.struct.field(default_factory=list)

    def populate_param_lists(self):
        """
        Populate parameter lists and set both field names and qualified names
        for all parameters. Returns a new instance with populated lists and
        properly named parameters.
        """
        if self._independent_params or self._shared_params or self._fixed_params:
            return self  # Already populated

        if self.name is None:
            raise ValueError("Profile must have a name before populating parameters")

        # Get all Parameter fields from this instance
        param_fields = {}
        updates = {}

        for field_name in self.__dataclass_fields__:
            if not field_name.startswith("_") and field_name != "name":
                field_value = getattr(self, field_name)
                if isinstance(field_value, Parameter):
                    # Set both field name and qualified name
                    if (
                        field_value._field_name is None
                        or field_value._qualified_name is None
                    ):
                        qualified_name = f"{self.name}_{field_name}"
                        updated_param = field_value.with_names(
                            field_name, qualified_name
                        )
                        param_fields[field_name] = updated_param
                        updates[field_name] = updated_param
                    else:
                        param_fields[field_name] = field_value

        independent = []
        shared = []
        fixed = []

        # First pass: categorize fixed vs non-fixed parameters
        for field_name, param in param_fields.items():
            if param.fixed:
                fixed.append(param)
            else:
                independent.append(param)

        # Second pass: handle shared parameters
        shared_candidates = []
        for field_name, param in param_fields.items():
            if param.shared is not None and not param.fixed:
                shared_candidates.append(param)

        for shared_param in shared_candidates:
            if shared_param in independent:
                independent.remove(shared_param)
            shared.append(shared_param)

        updates.update(
            {
                "_independent_params": independent,
                "_shared_params": shared,
                "_fixed_params": fixed,
            }
        )

        return self.replace(**updates)

    @classmethod
    def create(cls, name, **kwargs):
        if name is None:
            raise ValueError("Profile must have a name")

        # Separate parameters from other kwargs
        param_kwargs = {}
        other_kwargs = {}

        for k, v in kwargs.items():
            if not k.startswith("_"):
                if isinstance(v, Parameter):
                    # Set both field name and qualified name
                    qualified_name = f"{name}_{k}"
                    param_kwargs[k] = v.with_names(k, qualified_name)
                else:
                    other_kwargs[k] = v
            else:
                other_kwargs[k] = v

        # Create instance
        instance = cls(name=name, **param_kwargs, **other_kwargs)

        # Populate parameter lists
        return instance.populate_param_lists()

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
    radius_ratio: Optional[Parameter] = None
    inclination: Optional[Parameter] = None
    sigma: Optional[Parameter] = None
    q: Optional[Parameter] = None
    eccentricity: Optional[Parameter] = None
    apocenter: Optional[Parameter] = None
    scale: Parameter = Parameter(distribution=Distribution.UNIFORM, low=0, high=2)
    offset: Parameter = Parameter(distribution=Distribution.UNIFORM, low=0, high=2)


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
    redshift: Parameter = Parameter(
        distribution=Distribution.UNIFORM, low=0, high=1.0, _field_name="redshift"
    )
    obs_date: float = 0.0
    white_noise: Parameter = Parameter(
        distribution=Distribution.UNIFORM, low=-10, high=1, _field_name="white_noise"
    )
    mask: list[Mask] | None = None

    @property
    def all_parameters(self) -> list[Parameter]:
        params = [self.redshift, self.white_noise]

        for prof in self.disk_profiles + self.line_profiles:
            params.extend(prof.independent)
            params.extend(prof.shared)
            params.extend(prof.fixed)

        return params


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
    target_accept_prob: float = 0.9
    max_tree_depth: int = 10
    dense_mass: bool = True
    use_prefit: bool = True
    use_neutra: bool = False

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
