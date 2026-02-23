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
    flux = Parameter()
    offset = Parameter()

    def __init__(
        self,
        log_dist: dict = None,
        integrator: Callable = quad_jax_integrate,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._integrator = integrator
        self._log_dist = log_dist or {}

    def evaluate(self, x, *args):
        pars = {}

        for i, pn in enumerate(self.param_names):
            pars[f"{pn}"] = np.atleast_1d(args[i])

            if self._log_dist.get(pn, False):
                pars[f"{pn}"] = 10 ** pars[f"{pn}"]

        # Check for unphysical parameters
        # if (pars["outer_radius"] - pars["inner_radius"]) <= 10:
        #     return np.ones_like(x) * -99

        radius_ratio = pars.pop("radius_ratio")
        pars["outer_radius"] = pars["inner_radius"] * radius_ratio

        if pars["outer_radius"] > 5e4:
            return np.ones_like(x) * -99

        res = _compute_disk_flux_vectorized(x, **pars, integrator=self._integrator)

        if not np.all(np.isfinite(list(pars.values()))):
            raise ValueError(f"Invalid parameters for {self.name}: {pars}")

        if not np.all(np.isfinite(res)):
            raise ValueError(f"Invalid model evaluation for {self.name}: {pars}")

        return res


class LineProfileModel(Fittable1DModel):
    center = Parameter(fixed=True)
    offset = Parameter()
    flux = Parameter()
    vel_width = Parameter()

    def __init__(self, log_dist: dict = None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._log_dist = log_dist or {}

    def evaluate(self, x, *args):
        pars = {}

        for i, pn in enumerate(self.param_names):
            pars[f"{pn}"] = np.atleast_1d(args[i])

            if self._log_dist.get(pn, False):
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
):
    full_model = Const1D(amplitude=0, fixed={"amplitude": True}, name="base")
    full_in_par_log_dist = {}

    for prof in template.disk_profiles:
        in_par_values = {}
        in_par_bounds = {}
        in_par_fixed = {}
        in_par_log_dist = {}

        for param in prof.independent:
            param_low = param.low
            param_high = param.high

            if "log" in param.distribution:
                param_low = np.log10(param_low)
                param_high = np.log10(param_high)
                in_par_log_dist[param.name] = True
                full_in_par_log_dist[f"{prof.name}_{param.name}"] = True

            in_par_bounds[param.name] = (
                param_low,
                param_high,
            )

            if force_values is not None and f"{prof.name}_{param.name}" in force_values:
                param_val = force_values[f"{prof.name}_{param.name}"]
                param_val = (
                    np.log10(param_val) if "log" in param.distribution else param_val
                )
                in_par_values[param.name] = param_val
            else:
                in_par_values[param.name] = (param_high + param_low) / 2

        for param in prof.fixed:
            in_par_values[param.name] = param.value
            in_par_fixed[param.name] = True

        disk_mod = DiskProfileModel(
            log_dist=in_par_log_dist,
            **in_par_values,
            name=prof.name,
            bounds=in_par_bounds,
            fixed=in_par_fixed,
            integrator=integrator,
        )

        full_model += disk_mod

    for prof in template.line_profiles:
        if remove_broad:
            if "broad" in prof.name.lower():
                continue

        in_par_values = {}
        in_par_bounds = {}
        in_par_fixed = {}
        in_par_tied = {}
        in_par_log_dist = {}

        for param in prof.independent:
            param_low = param.low
            param_high = param.high

            if "log" in param.distribution:
                param_low = np.log10(param_low)
                param_high = np.log10(param_high)
                in_par_log_dist[param.name] = True
                full_in_par_log_dist[f"{prof.name}_{param.name}"] = True

            in_par_bounds[param.name] = (
                param_low,
                param_high,
            )

            if force_values is not None and f"{prof.name}_{param.name}" in force_values:
                param_val = force_values[f"{prof.name}_{param.name}"]
                param_val = (
                    np.log10(param_val) if "log" in param.distribution else param_val
                )
                in_par_values[param.name] = param_val
            else:
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
            log_dist=in_par_log_dist,
            center=prof.center,
            **in_par_values,
            name=prof.name,
            bounds=in_par_bounds,
            fixed=in_par_fixed,
            tied=in_par_tied,
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

    full_model.meta["log_dist"] = full_in_par_log_dist

    if (
        "niir_narrow" in full_model.submodel_names
        and "niil_narrow" in full_model.submodel_names
    ):
        niil_sm = full_model["niil_narrow"]
        niil_sm.flux.tied = lambda m: m["niir_narrow"].flux / 3.0

    return full_model


@flax.struct.dataclass
class LSQModel(BaseModel):
    model: Callable = lambda *args, **kwargs: None

    def setup(self, *args, **kwargs):
        model = _compose_model(self.config.template, integrator=self.integrator)
        return self.replace(config=self.config, integrator=self.integrator, model=model)

    def __call__(self, wave, param_mods, *args, **kwargs):
        return self.model(wave, **param_mods)
