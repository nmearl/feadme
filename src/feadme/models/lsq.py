import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.fitting import (
    TRFLSQFitter,
    model_to_fit_params,
)
from astropy.modeling.models import Const1D, Shift, RedshiftScaleFactor
from pathlib import Path
import astropy.uncertainty as unc

from ..compose import evaluate_model
from ..parser import Template, Data

FLOAT_EPSILON = 1e-6


class DiskProfileModel(Fittable1DModel):
    center = Parameter()
    inner_radius = Parameter()
    # delta_radius = Parameter()
    # outer_radius = Parameter()
    radius_ratio = Parameter()
    inclination = Parameter()
    sigma = Parameter()
    q = Parameter()
    eccentricity = Parameter()
    apocenter = Parameter()
    # apocenter_x = Parameter()
    # apocenter_y = Parameter()
    scale = Parameter()
    offset = Parameter()

    def __init__(self, template: Template, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._template = template

    def evaluate(self, x, *args):
        pars = {}
        for i, pn in enumerate(self.param_names):
            pars[f"{self._name}_{pn}"] = args[i].item()

            if pn in ["inner_radius", "delta_radius", "sigma", "radius_ratio"]:
                pars[f"{self._name}_{pn}"] = 10 ** pars[f"{self._name}_{pn}"]

        # pars[f"{self.name}_outer_radius"] = (
        #     pars[f"{self.name}_inner_radius"] + pars[f"{self.name}_delta_radius"]
        # )
        # del pars[f"{self.name}_delta_radius"]

        pars[f"{self._name}_outer_radius"] = (
            pars[f"{self._name}_inner_radius"] * pars[f"{self._name}_radius_ratio"]
        )
        del pars[f"{self._name}_radius_ratio"]

        # if pars[f"{self._name}_inner_radius"] >= pars[f"{self._name}_outer_radius"]:
        #     raise np.zeros(x.shape)

        res = evaluate_model(self._template, x, pars)[0]

        if np.any(np.isnan(list(pars.values()))) or np.any(
            np.isinf(list(pars.values()))
        ):
            print(f"Invalid parameters for {self.name}: {pars}")
            raise ValueError()

        if np.any(np.isnan(res)) or np.any(np.isinf(res)):
            print(f"Invalid model evaluation for {self.name}")
            from pprint import pprint

            pprint(pars)
            raise ValueError()

        return res


class LineProfileModel(Fittable1DModel):
    center = Parameter()
    amplitude = Parameter()
    vel_width = Parameter()

    def __init__(self, template: Template, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._template = template

    def evaluate(self, x, *args):
        pars = {}

        for i, pn in enumerate(self.param_names):
            pars[f"{self.name}_{pn}"] = args[i].squeeze()

            if pn in ["vel_width"]:
                pars[f"{self._name}_{pn}"] = 10 ** pars[f"{self._name}_{pn}"]

        res = evaluate_model(self._template, x, pars)[0]

        if np.any(np.isnan(list(pars.values()))) or np.any(
            np.isinf(list(pars.values()))
        ):
            print(f"Invalid parameters for {self.name}: {pars}")
            raise ValueError()

        if np.any(np.isnan(res)) or np.any(np.isinf(res)):
            print(f"Invalid model evaluation for {self.name}")
            from pprint import pprint

            pprint(pars)
            raise ValueError()

        return res


def lsq_model_fitter(
    template: Template, data: Data, force_values=None, show_plot=False, out_dir=None
):
    """
    Fit a least-squares model to the provided template and data.
    This function constructs a model based on the disk and line profiles defined in the template,
    applies the necessary masks to the data, and performs a fit using the TRFLSQFitter.

    Parameters
    ----------
    template : Template
        The template object containing disk and line profiles.
    data : Data
        The data object containing wavelength, flux, and flux error.
    force_values : dict, optional
        A dictionary of parameter names and values to force during the fit.
        The keys should be in the format "<profile_name>_<parameter_name>".
    show_plot : bool, optional
        If True, display a plot of the fit results. Defaults to False.

    Returns
    -------
    dict
        A dictionary containing the fitted parameters and their uncertainties.
        The keys are in the format "<profile_name>_<parameter_name>".
    """
    # Apply masks to data
    rest_wave = data.masked_wave
    flux = data.masked_flux
    flux_err = data.masked_flux_err

    full_model = Const1D(amplitude=0, fixed={"amplitude": True}, name="base")

    for prof in template.disk_profiles:
        in_par_values = {}
        in_par_bounds = {}
        in_par_fixed = {}

        for param in prof.independent:
            param_low = param.low
            param_high = param.high

            if param.name in ["inner_radius", "delta_radius", "sigma", "radius_ratio"]:
                param_low = np.log10(param_low)
                param_high = np.log10(param_high)

            in_par_bounds[param.name] = (
                param_low,
                param_high,
            )

            if force_values is not None and f"{prof.name}_{param.name}" in force_values:
                param_val = force_values[f"{prof.name}_{param.name}"]
                param_val = (
                    np.log10(param_val)
                    if param.name
                    in ["inner_radius", "delta_radius", "sigma", "radius_ratio"]
                    else param_val
                )
                in_par_values[param.name] = param_val
            else:
                in_par_values[param.name] = (param_high + param_low) / 2

        for param in prof.fixed:
            in_par_values[param.name] = param.value
            in_par_fixed[param.name] = True

        disk_mod = DiskProfileModel(
            template,
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

        for param in prof.independent:
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
                param_val = (
                    np.log10(param_val) if param.name in ["vel_width"] else param_val
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

            if param.name in ["vel_width"]:
                param_low = np.log10(param_low)
                param_high = np.log10(param_high)

            in_par_values[param.name] = (param_high + param_low) / 2
            in_par_tied[param.name] = lambda m, mn=param.shared, pn=param.name: getattr(
                m[mn], pn
            )

        line_mod = LineProfileModel(
            template,
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

    _, indices, _ = model_to_fit_params(full_model)

    fitter = TRFLSQFitter(calc_uncertainties=True)

    fit_mod = fitter(full_model, rest_wave, flux, weights=1 / flux_err, maxiter=10000)
    cov = fitter.fit_info["param_cov"]

    # Parameter uncertainties = sqrt of diagonal
    param_uncerts = np.sqrt(np.diag(cov))

    # Get real redshift
    fit_z = 1 / (1 + fit_mod["redshift"].z) - 1

    if out_dir is not None:
        fig, ax = plt.subplots()

        new_rest = np.linspace(
            rest_wave.min(),
            rest_wave.max(),
            1000,
        )

        ax.errorbar(
            rest_wave / (1 + fit_z),
            flux,
            yerr=flux_err,
            fmt="o",
            color="grey",
            zorder=-10,
            alpha=0.25,
        )
        ax.plot(
            new_rest / (1 + fit_z),
            fit_mod(new_rest),
            label="Model Fit",
            color="C3",
        )

        ax.set_title(
            f"LSQ Fit to {template.name} ({fit_z:.3f}, {template.redshift.value:.3f})"
        )

        for sm in fit_mod:
            if sm.name in ["shift", "base", "redshift"]:
                continue

            ax.plot(
                new_rest / (1 + fit_z), (fit_mod[0] | sm)(new_rest), label=f"{sm.name}"
            )

        txt = ""
        for pn, pv, pe in zip(
            np.array(fit_mod.param_names)[indices],
            fit_mod.parameters[indices],
            param_uncerts,
        ):
            pn = "_".join([fit_mod[int(pn.split("_")[-1])].name] + pn.split("_")[:-1])
            txt += f"{pn:15}: {pv:.3f}\n"  # ± {pe:.3f}\n"

        ax.text(
            0.05,
            0.95,
            txt[:-2],
            transform=ax.transAxes,
            fontsize=8,
            family="monospace",
            verticalalignment="top",
            # bbox=dict(facecolor="white", alpha=0.5, edgecolor="black"),
        )

        ax.legend()
        fig.savefig(Path(out_dir or "") / "lsq_model_fit.png")

        if not show_plot:
            plt.close(fig)

    starters = {}

    indep_params = [
        f"{prof.name}_{param.name}"
        for prof in template.disk_profiles + template.line_profiles
        for param in prof.independent
    ]

    _, inds, _ = model_to_fit_params(fit_mod)

    for pn, pv, pe, (plb, pub) in zip(
        np.array(fit_mod.param_names)[inds],
        fit_mod.parameters[inds],
        param_uncerts,
        np.array(list(fit_mod.bounds.values()))[inds],
    ):
        sm_idx = int(pn.split("_")[-1])
        pn = "_".join(pn.split("_")[:-1])
        sm = fit_mod[sm_idx]

        if sm.name in ["shift", "base"]:
            continue

        upv = unp.uarray(pv, pe)

        samp_name = f"{sm.name}_{pn}"

        # print(f"{samp_name:25}: {pv:.3f} ± {pe:.3f}")

        std_scale = 1

        if samp_name in indep_params:
            if pn in ["apocenter"]:
                ux = unp.cos(upv)
                x = unp.nominal_values(ux)
                xe = unp.std_devs(ux)

                uy = unp.sin(upv)
                y = unp.nominal_values(uy)
                ye = unp.std_devs(uy)

                starters[f"{samp_name}_x_base"] = (x, std_scale * xe, plb, pub)
                starters[f"{samp_name}_y_base"] = (y, std_scale * ye, plb, pub)

            if pn in [
                "inner_radius",
                "delta_radius",
                "sigma",
                "vel_width",
                "radius_ratio",
            ]:
                upv = 10**upv
                pv = unp.nominal_values(upv)
                pe = unp.std_devs(upv)
                plb = 10**plb
                pub = 10**pub

                # print(f"{samp_name:25}: {pv:.3f} ± {pe:.3f}")

            if pe < FLOAT_EPSILON:
                pe = 1

            starters[samp_name] = (pv, pe * std_scale, plb, pub)

    for prof in template.disk_profiles:
        starters[f"{prof.name}_outer_radius"] = (
            starters[f"{prof.name}_inner_radius"][0]
            * starters[f"{prof.name}_radius_ratio"][0],
            0,
            1e6,
        )

        if False:
            radius_ratio_dist = unc.normal(
                starters[f"{prof.name}_radius_ratio"][0],
                std=starters[f"{prof.name}_radius_ratio"][1],
                n_samples=10000,
            )

            inner_radius_dist = unc.normal(
                starters[f"{prof.name}_inner_radius"][0],
                std=starters[f"{prof.name}_inner_radius"][1],
                n_samples=10000,
            )

            starters[f"{prof.name}_outer_radius"] = (
                (inner_radius_dist * radius_ratio_dist).pdf_median(),
                (inner_radius_dist * radius_ratio_dist).pdf_std(),
                0,
                1e6,
            )

    return starters
