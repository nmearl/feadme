from typing import Callable, Dict

import astropy.constants as const
import astropy.units as u
import flax.struct
import numpy as np
from astropy.modeling import Fittable1DModel, Parameter, CompoundModel
from astropy.modeling.models import Const1D, RedshiftScaleFactor

from ..base_model import BaseModel
from ...core.evaluators import (
    _compute_line_flux_vectorized,
    _compute_disk_flux_vectorized,
)
from ...core.integrators import quad_jax_integrate
from ...core.parser import Template, Shape

FLOAT_EPSILON = 1e-6
c_kms = const.c.to(u.km / u.s).value


class DiskProfileModel(Fittable1DModel):
    center = Parameter()
    inner_radius = Parameter()
    outer_radius = Parameter()
    inclination = Parameter()
    sigma = Parameter()
    q = Parameter()
    eccentricity = Parameter()
    apocenter = Parameter()
    area = Parameter()
    offset = Parameter()

    def __init__(
        self,
        integrator: Callable = quad_jax_integrate,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._integrator = integrator

    def evaluate(self, x, *args):
        pars = {}

        for i, pn in enumerate(self.param_names):
            pars[f"{pn}"] = np.atleast_1d(args[i])

            if "log" in self.meta["distributions"].get(pn, ""):
                pars[f"{pn}"] = 10 ** pars[f"{pn}"]

        if not np.all(np.isfinite(list(pars.values()))):
            raise ValueError(f"Invalid parameters for {self.name}: {pars}")

        res = _compute_disk_flux_vectorized(
            x,
            **pars,
            integrator=self._integrator,
        )

        if not np.all(np.isfinite(res)):
            raise ValueError(f"Invalid model evaluation for {self.name}: {pars}")

        return res


class LineProfileModel(Fittable1DModel):
    center = Parameter(fixed=True)
    offset = Parameter()
    area = Parameter()
    vel_width = Parameter()

    def evaluate(self, x, *args):
        pars = {}

        for i, pn in enumerate(self.param_names):
            pars[f"{pn}"] = np.atleast_1d(args[i])

            if "log" in self.meta["distributions"].get(pn, ""):
                pars[f"{pn}"] = 10 ** pars[f"{pn}"]

        res = _compute_line_flux_vectorized(x, **pars, shape=self.meta["shape"])

        if not np.all(np.isfinite(list(pars.values()))):
            print(f"Invalid parameters for {self.name}: {pars}")

        if not np.all(np.isfinite(res)):
            print(f"Invalid model evaluation for {self.name}")
            from pprint import pprint

            pprint(pars)

        return res


def _compose_model(
    template: Template, integrator: Callable = quad_jax_integrate, redshift: float = 0.0
) -> CompoundModel:
    full_model = Const1D(amplitude=0, fixed={"amplitude": True}, name="base")

    for prof in template.disk_profiles:
        in_par_values = {}
        in_par_bounds = {}
        in_par_fixed = {}
        in_par_tied = {}

        for param_name, param in prof.iter_independent:
            param_low = param.low
            param_high = param.high

            if "log" in param.distribution.value:
                param_low = np.log10(param_low)
                param_high = np.log10(param_high)

            in_par_bounds[param_name] = (
                param_low,
                param_high,
            )

            in_par_values[param_name] = (param_high + param_low) / 2

        for param_name, param in prof.iter_fixed:
            in_par_values[param_name] = param.value
            in_par_fixed[param_name] = True

        for param_name, param in prof.iter_shared:
            param_low = param.low
            param_high = param.high

            if "log" in param.distribution.value:
                param_low = np.log10(param_low)
                param_high = np.log10(param_high)

            in_par_values[param_name] = (param_high + param_low) / 2
            in_par_tied[param_name] = lambda m, mn=param.shared, pn=param_name: getattr(
                m[mn], pn
            )

        disk_mod = DiskProfileModel(
            **in_par_values,
            name=prof.name,
            bounds=in_par_bounds,
            fixed=in_par_fixed,
            tied=in_par_tied,
            integrator=integrator,
            meta={
                "distributions": {
                    param_name: param.distribution.value
                    for param_name, param in prof.iter_independent
                    + prof.iter_shared
                    + prof.iter_fixed
                },
            },
        )

        full_model += disk_mod

    for prof in template.line_profiles:
        in_par_values = {}
        in_par_bounds = {}
        in_par_fixed = {}
        in_par_tied = {}

        for param_name, param in prof.iter_independent:
            param_low = param.low
            param_high = param.high

            if "log" in param.distribution.value:
                param_low = np.log10(param_low)
                param_high = np.log10(param_high)

            in_par_bounds[param_name] = (
                param_low,
                param_high,
            )

            in_par_values[param_name] = (param_high + param_low) / 2

        for param_name, param in prof.iter_fixed:
            in_par_values[param_name] = param.value
            in_par_fixed[param_name] = True

        for param_name, param in prof.iter_shared:
            param_low = param.low
            param_high = param.high

            if "log" in param.distribution.value:
                param_low = np.log10(param_low)
                param_high = np.log10(param_high)

            in_par_values[param_name] = (param_high + param_low) / 2
            in_par_tied[param_name] = lambda m, mn=param.shared, pn=param_name: getattr(
                m[mn], pn
            )

        line_mod = LineProfileModel(
            center=prof.center,
            **in_par_values,
            name=prof.name,
            bounds=in_par_bounds,
            fixed=in_par_fixed,
            tied=in_par_tied,
            meta={
                "distributions": {
                    param_name: param.distribution.value
                    for param_name, param in prof.iter_independent
                    + prof.iter_shared
                    + prof.iter_fixed
                },
                "shape": np.array([prof.shape == Shape.GAUSSIAN]),
            },
        )

        full_model += line_mod

    # Redshift
    z_lower_limit = 1 / (1 + template.redshift.high) - 1
    z_upper_limit = 1 / (1 + template.redshift.low) - 1
    z_value = (
        template.redshift.value or 1 / (1 + (z_lower_limit + z_upper_limit) / 2) - 1
    )

    full_model = (
        RedshiftScaleFactor(
            z=z_value,
            fixed={"z": template.redshift.fixed},
            bounds={"z": (z_lower_limit, z_upper_limit)},
            name="redshift",
        ).inverse
        | full_model
    )

    # Fix NII line ratio if both lines are present
    if (
        "niir_narrow" in full_model.submodel_names
        and "niil_narrow" in full_model.submodel_names
    ):
        niil_sm = full_model["niil_narrow"]
        niil_sm.area.tied = lambda m: m["niir_narrow"].area / 3.0

    # Add distribution metadata from sub models
    full_model.meta["distributions"] = {
        f"{sm.name}_{pn}": dist
        for sm in full_model
        for pn, dist in sm.meta.get("distributions", {}).items()
    }

    return full_model


@flax.struct.dataclass
class LSQModel(BaseModel):
    model: Callable = lambda *args, **kwargs: None

    def setup(self, *args, **kwargs):
        model = _compose_model(self.config.template, integrator=self.integrator, redshift=self.config.template.redshift.value)
        return self.replace(config=self.config, integrator=self.integrator, model=model)

    def __call__(self, wave, param_mods, *args, **kwargs):
        return self.model(wave, **param_mods)
