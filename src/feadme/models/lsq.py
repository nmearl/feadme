from uuid import uuid4

from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.models import Gaussian1D, Const1D, Shift, Polynomial1D
from astropy.modeling.fitting import (
    LMLSQFitter,
    TRFLSQFitter,
    DogBoxLSQFitter,
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
import astropy.uncertainty as unc
import uncertainties.unumpy as unp

FLOAT_EPSILON = 1e-6


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

    def __init__(self, template, use_quad=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._use_quad = use_quad

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

        res = evaluate_disk_model(self._template, x, pars, use_quad=self._use_quad)[0]

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

            if pn in ["vel_width"]:
                pars[f"{self._name}_{pn}"] = 10 ** pars[f"{self._name}_{pn}"]

        res = evaluate_disk_model(self._template, x, pars)[0]

        return res


def lsq_model_fitter(template, rest_wave, flux, flux_err, use_quad=False, force_values=None, show_plot=False):
    # Construct masks
    mask = [
        np.bitwise_and(rest_wave > m.lower_limit, rest_wave < m.upper_limit)
        for m in template.mask
    ]

    if len(mask) > 1:
        mask = np.bitwise_or.reduce(mask)
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

            if force_values is not None and f"{prof.name}_{param.name}" in force_values:
                param_val = force_values[f"{prof.name}_{param.name}"]
                param_val = np.log10(param_val) if param.name in ["inner_radius", "delta_radius", "sigma"] else param_val
                in_par_values[param.name] = param_val
            else:
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
            use_quad=use_quad,
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

        # Find shared profiles whose parent doesn't exist; make them independent
        _shared_orphans = [
            param
            for param in prof._shared()
            if param.shared not in [p.name for p in template.all_profiles]
        ]

        for param in prof._independent() + _shared_orphans:
            param_low = param.low
            param_high = param.high

            if param.name in ["vel_width"]:
                param_low = np.log10(param_low)
                param_high = np.log10(param_high)

            in_par_bounds[param.name] = (
                param_low,
                param_high,
            )

            if force_values is not None and f"{prof.name}_{param.name}" in force_values:
                param_val = force_values[f"{prof.name}_{param.name}"]
                param_val = np.log10(param_val) if param.name in ["vel_width"] else param_val
                in_par_values[param.name] = param_val
            else:
                in_par_values[param.name] = (param_high + param_low) / 2

        for param in prof._fixed():
            in_par_values[param.name] = param.value
            in_par_fixed[param.name] = True

        for param in [x for x in prof._shared() if x not in _shared_orphans]:
            param_low = param.low
            param_high = param.high

            if param.name in ["vel_width"]:
                param_low = np.log10(param_low)
                param_high = np.log10(param_high)

            in_par_values[param.name] = (param_high + param_low) / 2
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

    fitter = TRFLSQFitter(calc_uncertainties=True)

    fit_mod = fitter(full_model, rest_wave, flux, weights=1 / flux_err, maxiter=10000)
    cov = fitter.fit_info["param_cov"]

    # Parameter uncertainties = sqrt of diagonal
    param_uncerts = np.sqrt(np.diag(cov))

    if show_plot:
        fig, ax = plt.subplots()

        new_rest = np.linspace(
            rest_wave.min(),
            rest_wave.max(),
            1000,
        )

        ax.errorbar(
            rest_wave,
            flux,
            yerr=flux_err,
            fmt="o",
            color="grey",
            zorder=-10,
            alpha=0.25,
        )
        ax.plot(
            new_rest,
            fit_mod(new_rest),
            label="Model Fit",
            color="C3",
        )

        ax.set_title(f"LSQ Fit to {template.name} {template.mjd} ({template.redshift})")

        for sm in fit_mod:
            if sm.name in ["shift", "base"]:
                continue

            ax.plot(new_rest, sm(new_rest), label=f"{sm.name}")

        txt = ""
        for pn, pv, pe in zip(
            np.array(fit_mod.param_names)[indices],
            fit_mod.parameters[indices],
            param_uncerts,
        ):
            txt += f"{pn:15}: {pv:.3f}\n"# ± {pe:.3f}\n"

        ax.text(
            0.05,
            0.95,
            txt,
            transform=ax.transAxes,
            fontsize=8,
            family="monospace",
            verticalalignment="top",
            bbox=dict(facecolor="white", alpha=0.5, edgecolor="black"),
        )

        ax.legend()
        # fig.savefig(f"/Users/nmearl/Downloads/lsq_fit_{template.name}_{template.mjd}.png")

    starters = {}

    indep_params = [
        f"{prof.name}_{param.name}"
        for prof in template.all_profiles
        for param in prof._independent()
    ]

    _, inds, _ = model_to_fit_params(fit_mod)

    for pn, pv, pe in zip(
        np.array(fit_mod.param_names)[inds], fit_mod.parameters[inds], param_uncerts
    ):
        sm_idx = int(pn.split("_")[-1])
        pn = "_".join(pn.split("_")[:-1])
        sm = fit_mod[sm_idx]

        if sm.name in ["shift", "base"]:
            continue

        upv = unp.uarray(pv, pe)

        samp_name = f"{sm.name}_{pn}"

        print(f"{samp_name:25}: {pv:.3f} ± {pe:.3f}")

        if samp_name in indep_params:
            if pn in ["apocenter"]:
                ux = unp.cos(upv)
                x = unp.nominal_values(ux)
                xe = unp.std_devs(ux)

                uy = unp.sin(upv)
                y = unp.nominal_values(uy)
                ye = unp.std_devs(uy)

                starters[f"{samp_name}_x"] = (x, 5 * xe)
                starters[f"{samp_name}_y"] = (y, 5 * ye)

            if pn in ["inner_radius", "delta_radius", "sigma", "vel_width"]:
                upv = 10**upv
                pv = unp.nominal_values(upv)
                pe = unp.std_devs(upv)

                print(f"{samp_name:25}: {pv:.3f} ± {pe:.3f}")

            if pe < FLOAT_EPSILON:
                pe = 1

            starters[samp_name] = (pv, pe * 5)

    return starters
