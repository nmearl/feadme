from uuid import uuid4

from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Gaussian1D, Const1D, Shift
from astropy.modeling.fitting import (
    LMLSQFitter,
    TRFLSQFitter,
    model_to_fit_params,
    LevMarLSQFitter,
)
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from astropy.table import Table, vstack
from ..parser import Template, Mask
from ..compose import evaluate_disk_model  # , jax_evaluate_disk_model
from collections import namedtuple
from itertools import product
from ..utils import dict_to_namedtuple


class DiskProfileModel(Fittable1DModel):
    center = Parameter()
    inner_radius = Parameter()
    delta_radius = Parameter()
    inclination = Parameter()
    sigma = Parameter()
    q = Parameter()
    eccentricity = Parameter()
    apocenter = Parameter()
    scale = Parameter()
    offset = Parameter()

    def __init__(self, template, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._template = dict_to_namedtuple("NTemplate", template.model_dump())

    def evaluate(self, x, *args):
        pars = {}
        for i, pn in enumerate(self.param_names):
            pars[f"{self._name}_{pn}"] = args[i].item()

            if pn in ["inner_radius", "delta_radius", "sigma"]:
                pars[f"{self._name}_{pn}"] = 10 ** pars[f"{self._name}_{pn}"]

        pars[f"{self.name}_outer_radius"] = (
            pars[f"{self.name}_inner_radius"] + pars[f"{self.name}_delta_radius"]
        )
        del pars[f"{self.name}_delta_radius"]

        # pars = dict_to_namedtuple("NTParamMods", pars)
        # print(pars)

        res = evaluate_disk_model(self._template, x, pars)[0]
        if np.any(np.isnan(res)) or np.any(np.isinf(res)):
            print(pars)
        return res


class LineProfileModel(Fittable1DModel):
    center = Parameter()
    amplitude = Parameter()
    vel_width = Parameter()

    def __init__(self, template, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._template = dict_to_namedtuple("NTemplate", template.model_dump())

    def evaluate(self, x, *args):
        pars = {}

        for i, pn in enumerate(self.param_names):
            pars[f"{self.name}_{pn}"] = args[i].squeeze()

        # pars = dict_to_namedtuple("NTParamMods", pars)
        # print(pars)

        res = evaluate_disk_model(self._template, x, pars)[0]
        # print(res)
        return res


def lsq_model_fitter(template, rest_wave, flux, flux_err):
    # Construct masks
    mask = [
        np.bitwise_and(rest_wave > m.lower_limit, rest_wave < m.upper_limit)
        for m in template.mask
    ]

    if len(mask) > 1:
        mask = np.bitwise_or(*mask)
    else:
        mask = mask[0]

    # Apply masks to data
    rest_wave = rest_wave[mask]
    flux = flux[mask]
    flux_err = flux_err[mask]

    full_model = Const1D(amplitude=0, fixed={"amplitude": True}, name="base")

    for prof in template.disk_profiles:
        in_par_values = {}
        in_par_bounds = {}
        in_par_fixed = {}

        disk_temp = template.model_copy()
        disk_temp.disk_profiles = [prof]
        disk_temp.line_profiles = []

        for param in prof._independent():
            param_low = param.low
            param_high = param.high

            if param.name in ["inner_radius", "delta_radius", "sigma"]:
                param_low = np.log10(param_low)
                param_high = np.log10(param_high)

            in_par_bounds[param.name] = (
                param_low,
                param_high,
            )

            in_par_values[param.name] = (param_high + param_low) / 2

        for param in prof._fixed():
            in_par_values[param.name] = param.value
            in_par_fixed[param.name] = True

        disk_mod = DiskProfileModel(
            disk_temp,
            **in_par_values,
            name=prof.name,
            bounds=in_par_bounds,
            fixed=in_par_fixed,
        )

        full_model += disk_mod

    for prof in template.line_profiles:
        in_par_values = {}
        in_par_bounds = {}
        in_par_fixed = {}
        in_par_tied = {}

        line_temp = template.model_copy()
        line_temp.disk_profiles = []
        line_temp.line_profiles = [prof]

        for param in prof._independent():
            in_par_bounds[param.name] = (
                param.low,
                param.high,
            )

            in_par_values[param.name] = (param.high + param.low) / 2

        for param in prof._fixed():
            in_par_values[param.name] = param.value
            in_par_fixed[param.name] = True

        for param in prof._shared():
            in_par_values[param.name] = (param.high + param.low) / 2
            in_par_tied[param.name] = lambda m, mn=param.shared, pn=param.name: getattr(
                m[mn], pn
            )

        line_mod = LineProfileModel(
            line_temp,
            **in_par_values,
            name=prof.name,
            bounds=in_par_bounds,
            fixed=in_par_fixed,
            tied=in_par_tied,
        )

        full_model += line_mod

    full_model = (
        Shift(
            offset=0, bounds={"offset": (-20, 5)}, name="shift", fixed={"offset": True}
        )
        | full_model
    )

    _, indices, _ = model_to_fit_params(full_model)

    fitter = TRFLSQFitter()
    fit_mod = fitter(full_model, rest_wave, flux, weights=1 / flux_err, maxiter=10000)

    starters = {}

    indep_params = [
        f"{prof.name}_{param.name}"
        for prof in template.all_profiles
        for param in prof._independent()
    ]

    for sm in fit_mod:
        if sm.name in ["shift", "base"]:
            continue

        _, inds, _ = model_to_fit_params(sm)

        for pn, pv in zip(np.array(sm.param_names)[inds], sm.parameters[inds]):
            samp_name = f"{sm.name}_{pn}"

            if samp_name in indep_params:
                if pn in ['apocenter']:
                    starters[f"{samp_name}_x"] = np.cos(pv)
                    starters[f"{samp_name}_y"] = np.sin(pv)

                if pn in ["inner_radius", "delta_radius", "sigma"]:
                    pv = 10**pv

                starters[samp_name] = pv

    return starters
