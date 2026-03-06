import json
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import flax.struct
import jax.numpy as jnp
from dacite import Config as DaciteConfig
from dacite import from_dict as dacite_from_dict
from jax.tree_util import tree_map
from jax.typing import ArrayLike


def _to_serializable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Enum):
        return value.value
    return value


class Writable:
    """
    Mixin for JSON serialization. Subclasses may define
    `_EXCLUDE_FROM_SERIALIZATION` as a frozenset of field names to omit
    when writing to disk.
    """

    _EXCLUDE_FROM_SERIALIZATION: frozenset[str] = frozenset()

    def to_dict(self) -> dict[str, Any]:
        raw = flax.struct.dataclasses.asdict(self)
        raw = {
            k: v for k, v in raw.items() if k not in self._EXCLUDE_FROM_SERIALIZATION
        }
        return tree_map(_to_serializable, raw)

    def to_json(self, path: str | Path) -> dict[str, Any]:
        serializable = self.to_dict()
        with open(path, "w") as f:
            json.dump(serializable, f, indent=4)
        return serializable

    @classmethod
    def from_json(cls, path: str | Path):
        with open(path, "r") as f:
            raw = json.load(f)
        return cls.from_dict(raw)

    @classmethod
    def _after_from_dict(cls, instance):
        return instance

    @classmethod
    def from_dict(cls, raw: dict):
        instance = dacite_from_dict(
            data_class=cls,
            data=raw,
            config=DaciteConfig(
                type_hooks={
                    ArrayLike: lambda v: jnp.asarray(v),
                    Distribution: lambda v: Distribution(v),
                    Shape: lambda v: Shape(v),
                }
            ),
        )
        return cls._after_from_dict(instance)


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


class Shape(str, Enum):
    GAUSSIAN = "gaussian"
    LORENTZIAN = "lorentzian"


@flax.struct.dataclass
class Parameter:
    distribution: Distribution = Distribution.UNIFORM
    value: Optional[float] = None
    fixed: bool = False
    shared: Optional[str] = None
    low: Optional[float] = None
    high: Optional[float] = None
    loc: Optional[float] = None
    scale: Optional[float] = None
    circular: bool = False


@flax.struct.dataclass
class ParameterRef:
    """
    Canonical reference to a single model parameter. `name` is the
    fully-qualified sampler name, e.g. ``disk1_inclination``. For
    profile-level parameters, `profile_name` and `field_name`
    are set. For top-level parameters (redshift, white_noise) they are None.
    """

    name: str
    param: Parameter
    profile_name: Optional[str] = None
    field_name: Optional[str] = None
    target_name: Optional[str] = None


@flax.struct.dataclass
class TemplateIndex:
    """
    Single precomputed index consumed by the NumPyro model.

    All four tuples are views over the same `ParameterRef` objects built once
    by `_build_template_index`.
    """

    parameters: tuple[ParameterRef, ...] = flax.struct.field(
        default_factory=tuple, pytree_node=False
    )
    independent: tuple[ParameterRef, ...] = flax.struct.field(
        default_factory=tuple, pytree_node=False
    )
    fixed: tuple[ParameterRef, ...] = flax.struct.field(
        default_factory=tuple, pytree_node=False
    )
    shared: tuple[ParameterRef, ...] = flax.struct.field(
        default_factory=tuple, pytree_node=False
    )
    circular: tuple[ParameterRef, ...] = flax.struct.field(
        default_factory=tuple, pytree_node=False
    )


@flax.struct.dataclass
class Profile:
    name: Optional[str] = None

    def iter_parameter_fields(self) -> tuple[tuple[str, Parameter], ...]:
        """All (field_name, Parameter) pairs on this profile."""
        return tuple(
            (field.name, getattr(self, field.name))
            for field in flax.struct.dataclasses.fields(self)
            if isinstance(getattr(self, field.name), Parameter)
        )

    @property
    def iter_independent(self) -> tuple[tuple[str, Parameter], ...]:
        return tuple(
            (n, p)
            for n, p in self.iter_parameter_fields()
            if not p.fixed and p.shared is None
        )

    @property
    def iter_fixed(self) -> tuple[tuple[str, Parameter], ...]:
        return tuple((n, p) for n, p in self.iter_parameter_fields() if p.fixed)

    @property
    def iter_shared(self) -> tuple[tuple[str, Parameter], ...]:
        return tuple(
            (n, p) for n, p in self.iter_parameter_fields() if p.shared is not None
        )


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
    area: Optional[Parameter] = None
    offset: Parameter = flax.struct.field(
        default_factory=lambda: Parameter(
            distribution=Distribution.UNIFORM, low=0.0, high=2.0
        )
    )


@flax.struct.dataclass
class Line(Profile, Writable):
    center: Optional[float] = None
    offset: Optional[Parameter] = None
    area: Optional[Parameter] = None
    vel_width: Optional[Parameter] = None
    shape: Shape = Shape.GAUSSIAN


@flax.struct.dataclass
class Mask:
    lower_limit: float
    upper_limit: float


def _profile_refs(profile: Profile) -> tuple[ParameterRef, ...]:
    return tuple(
        ParameterRef(
            name=f"{profile.name}_{field_name}",
            param=param,
            profile_name=profile.name,
            field_name=field_name,
            target_name=(
                f"{param.shared}_{field_name}" if param.shared is not None else None
            ),
        )
        for field_name, param in profile.iter_parameter_fields()
    )


def _build_template_index(template: "Template") -> TemplateIndex:
    refs: list[ParameterRef] = [
        ParameterRef(name="redshift", param=template.redshift),
        ParameterRef(name="white_noise", param=template.white_noise),
    ]

    for disk in template.disk_profiles:
        refs.extend(_profile_refs(disk))

    for line in template.line_profiles:
        refs.extend(_profile_refs(line))

    all_parameters = tuple(refs)

    return TemplateIndex(
        parameters=all_parameters,
        independent=tuple(
            r for r in all_parameters if not r.param.fixed and r.param.shared is None
        ),
        fixed=tuple(r for r in all_parameters if r.param.fixed),
        shared=tuple(r for r in all_parameters if r.param.shared is not None),
        circular=tuple(r for r in all_parameters if r.param.circular),
    )


@flax.struct.dataclass
class Template(Writable):
    name: str = "default_template"
    disk_profiles: list[Disk] = flax.struct.field(default_factory=list)
    line_profiles: list[Line] = flax.struct.field(default_factory=list)
    redshift: Parameter = flax.struct.field(
        default_factory=lambda: Parameter(
            distribution=Distribution.UNIFORM, low=0.0, high=1.0
        )
    )
    obs_date: float = 0.0
    white_noise: Parameter = flax.struct.field(
        default_factory=lambda: Parameter(
            distribution=Distribution.UNIFORM, low=-10.0, high=1.0
        )
    )
    mask: Optional[list[Mask]] = None

    index: TemplateIndex = flax.struct.field(
        default_factory=TemplateIndex,
        pytree_node=False,
    )

    _EXCLUDE_FROM_SERIALIZATION: frozenset[str] = frozenset({"index"})

    def refresh(self) -> "Template":
        """Rebuild the index from current profile/parameter state."""
        return self.replace(index=_build_template_index(self))

    @classmethod
    def create(
        cls,
        name: str = "default_template",
        disk_profiles: Optional[list[Disk]] = None,
        line_profiles: Optional[list[Line]] = None,
        redshift: Optional[Parameter] = None,
        obs_date: float = 0.0,
        white_noise: Optional[Parameter] = None,
        mask: Optional[list[Mask]] = None,
    ) -> "Template":
        instance = cls(
            name=name,
            disk_profiles=[] if disk_profiles is None else disk_profiles,
            line_profiles=[] if line_profiles is None else line_profiles,
            redshift=(
                redshift
                if redshift is not None
                else Parameter(distribution=Distribution.UNIFORM, low=0.0, high=1.0)
            ),
            obs_date=obs_date,
            white_noise=(
                white_noise
                if white_noise is not None
                else Parameter(distribution=Distribution.UNIFORM, low=-10.0, high=1.0)
            ),
            mask=mask,
        )
        return instance.refresh()

    @classmethod
    def _after_from_dict(cls, instance: "Template") -> "Template":
        return instance.refresh()

    @property
    def iter_independent(self) -> tuple[ParameterRef, ...]:
        return self.index.independent

    @property
    def iter_fixed(self) -> tuple[ParameterRef, ...]:
        return self.index.fixed

    @property
    def iter_shared(self) -> tuple[ParameterRef, ...]:
        return self.index.shared

    @property
    def iter_all(self) -> tuple[ParameterRef, ...]:
        return self.index.parameters

    def fitted_parameter_names(
        self, include_shared: bool = True, include_circ: bool = False
    ) -> list[str]:
        return [
            ref.name
            for ref in self.index.parameters
            if not ref.param.fixed
            and (include_shared or ref.param.shared is None)
            and (include_circ or not ref.param.circular)
        ]


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
    def create(
        cls,
        wave,
        flux,
        flux_err,
        mask: Optional[list[Mask]] = None,
    ):
        wave = jnp.asarray(wave)
        flux = jnp.asarray(flux)
        flux_err = jnp.asarray(flux_err)

        mask_array = jnp.ones(len(wave), dtype=bool)

        if mask is not None and len(mask) > 0:
            lower_limits = jnp.asarray([m.lower_limit for m in mask])
            upper_limits = jnp.asarray([m.upper_limit for m in mask])
            wave_expanded = wave[:, None]
            individual_masks = (wave_expanded >= lower_limits) & (
                wave_expanded <= upper_limits
            )
            mask_array = jnp.any(individual_masks, axis=1)

        return cls(
            wave=wave,
            flux=flux,
            flux_err=flux_err,
            mask=mask_array,
            masked_wave=wave[mask_array],
            masked_flux=flux[mask_array],
            masked_flux_err=flux_err[mask_array],
        )


@flax.struct.dataclass
class Config(Writable):
    template: Template
    data: Data
    output_path: str
    data_path: str
    template_path: Optional[str] = None
    skip_existing: bool = False

    @classmethod
    def _after_from_dict(cls, instance: "Config") -> "Config":
        return instance.replace(template=instance.template.refresh())
