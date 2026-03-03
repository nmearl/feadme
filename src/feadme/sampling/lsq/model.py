from typing import Callable, Dict

import astropy.constants as const
import astropy.units as u
import flax.struct
import numpy as np
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Const1D, RedshiftScaleFactor

from ..base_model import BaseModel
from ...core.evaluators import (
    _compute_line_flux_vectorized,
    _compute_disk_flux_vectorized,
)
from ...core.integrators import quad_jax_integrate
from ...core.parser import Template

FLOAT_EPSILON = 1e-6
c_kms = const.c.to(u.km / u.s).value


class DiskProfileModel(Fittable1DModel):
    center = Parameter()
    inner_radius = Parameter()
    radius_ratio = Parameter()
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

        radius_ratio = pars.pop("radius_ratio")
        pars["outer_radius"] = pars["inner_radius"] * radius_ratio

        if pars["outer_radius"] > 2e4:
            return np.full_like(x, 1e10)

        if radius_ratio > 2e4 / pars["inner_radius"]:
            return np.full_like(x, 1e10)

        # if radius_ratio <= 1 or radius_ratio > 2e4 / 5e2:
        #     return np.full_like(x, 1e10)

        if pars["outer_radius"] <= pars["inner_radius"]:
            return np.full_like(x, 1e10)

        res = _compute_disk_flux_vectorized(x, **pars, integrator=self._integrator)

        if not np.all(np.isfinite(list(pars.values()))):
            raise ValueError(f"Invalid parameters for {self.name}: {pars}")

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

        res = _compute_line_flux_vectorized(x, **pars, shape=np.array([1]).astype(bool))

        if not np.all(np.isfinite(list(pars.values()))):
            print(f"Invalid parameters for {self.name}: {pars}")

        if not np.all(np.isfinite(res)):
            print(f"Invalid model evaluation for {self.name}")
            from pprint import pprint

            pprint(pars)

        return res


def _compose_model(
    template: Template,
    integrator: Callable = quad_jax_integrate,
    force_values: Dict | None = None,
    remove_broad: bool = False,
    max_outer_radius: float = 2e4,
):
    full_model = Const1D(amplitude=0, fixed={"amplitude": True}, name="base")

    for prof in template.disk_profiles:
        in_par_values = {}
        in_par_bounds = {}
        in_par_fixed = {}

        for param in prof.independent:
            param_low = param.low
            param_high = param.high

            if "log" in param.distribution.value:
                param_low = np.log10(param_low)
                param_high = np.log10(param_high)

            in_par_bounds[param.name] = (
                param_low,
                param_high,
            )

            in_par_values[param.name] = (param_high + param_low) / 2

        # Handle radius ratio explicitly
        # for param in prof.independent:
        #     if param.name == "radius_ratio":
        #         max_radius_ratio = max_outer_radius / in_par_values["inner_radius"]
        #         param_low = np.log10(param.low)
        #         param_high = np.log10(max_radius_ratio)
        #         in_par_values[param.name] = (param_high + param_low) / 2
        #         print(
        #             f"Adjusted radius_ratio upper bound: {param_low} ({10 ** param_low}), {param_high} ({10 ** param_high}): {in_par_values[param.name]}"
        #         )

        for param in prof.fixed:
            in_par_values[param.name] = param.value
            in_par_fixed[param.name] = True

        disk_mod = DiskProfileModel(
            **in_par_values,
            name=prof.name,
            bounds=in_par_bounds,
            fixed=in_par_fixed,
            integrator=integrator,
            meta={
                "distributions": {
                    param.name: param.distribution.value for param in prof.independent
                }
            },
        )

        full_model += disk_mod

    for prof in template.line_profiles:
        in_par_values = {}
        in_par_bounds = {}
        in_par_fixed = {}
        in_par_tied = {}

        for param in prof.independent:
            param_low = param.low
            param_high = param.high

            if "log" in param.distribution.value:
                param_low = np.log10(param_low)
                param_high = np.log10(param_high)

            in_par_bounds[param.name] = (
                param_low,
                param_high,
            )

            in_par_values[param.name] = (param_high + param_low) / 2

        for param in prof.fixed:
            in_par_values[param.name] = param.value
            in_par_fixed[param.name] = True

        for param in prof.shared:
            param_low = param.low
            param_high = param.high

            if "log" in param.distribution:
                param_low = np.log10(param_low)
                param_high = np.log10(param_high)

            in_par_values[param.name] = (param_high + param_low) / 2
            in_par_tied[param.name] = lambda m, mn=param.shared, pn=param.name: getattr(
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
                    param.name: param.distribution.value for param in prof.independent
                }
            },
        )

        full_model += line_mod

    # Redshift
    full_model = (
        RedshiftScaleFactor(
            z=template.redshift.value,
            fixed={"z": template.redshift.fixed},
            bounds={
                "z": (
                    1 / (1 + template.redshift.high) - 1,
                    1 / (1 + template.redshift.low) - 1,
                )
            },
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
        model = _compose_model(self.config.template, integrator=self.integrator)
        return self.replace(config=self.config, integrator=self.integrator, model=model)

    def __call__(self, wave, param_mods, *args, **kwargs):
        return self.model(wave, **param_mods)
