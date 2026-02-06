import matplotlib.pyplot as plt
import numpy as np
import uncertainties.unumpy as unp
from astropy.modeling import Fittable1DModel, Parameter
from astropy.modeling.fitting import (
    TRFLSQFitter,
    model_to_fit_params,
    LMLSQFitter,
    LevMarLSQFitter,
    DogBoxLSQFitter,
)
from astropy.modeling.models import Const1D, Shift, RedshiftScaleFactor
from pathlib import Path
import astropy.uncertainty as unc

from ..compose import (
    evaluate_model,
    _compute_line_flux_vectorized,
    _compute_disk_flux_vectorized,
)
from ..parser import Template, Data

FLOAT_EPSILON = 1e-6


class DiskProfileModel(Fittable1DModel):
    center = Parameter()
    inner_radius = Parameter()
    outer_radius = Parameter()
    inclination = Parameter()
    sigma = Parameter()
    q = Parameter()
    eccentricity = Parameter()
    apocenter = Parameter()
    scale = Parameter()
    offset = Parameter()

    def __init__(self, template: Template, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._template = template

    def evaluate(self, x, *args):
        pars = {}
        for i, pn in enumerate(self.param_names):
            pars[f"{pn}"] = np.atleast_1d(args[i])

            if pn in [
                "inner_radius",
                "outer_radius",
                "delta_radius",
                "sigma",
                "radius_ratio",
                "scale",
            ]:
                pars[f"{pn}"] = 10 ** pars[f"{pn}"]

        if pars["outer_radius"] - pars["inner_radius"] <= 100:
            return np.zeros_like(x)

        res = _compute_disk_flux_vectorized(x, **pars)

        if not np.all(np.isfinite(list(pars.values()))):
            print(f"Invalid parameters for {self.name}: {pars}")

        if not np.all(np.isfinite(res)):
            print(f"Invalid model evaluation for {self.name}")
            from pprint import pprint

            pprint(pars)

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
            pars[f"{pn}"] = np.atleast_1d(args[i])

            if pn in ["vel_width"]:
                pars[f"{pn}"] = 10 ** pars[f"{pn}"]

        res = _compute_line_flux_vectorized(x, **pars, shape=np.array([1]))

        if not np.all(np.isfinite(list(pars.values()))):
            print(f"Invalid parameters for {self.name}: {pars}")

        if not np.all(np.isfinite(res)):
            print(f"Invalid model evaluation for {self.name}")
            from pprint import pprint

            pprint(pars)

        return res


def _construct_starters_dict(
    fit_mod: Fittable1DModel,
    param_uncerts: np.ndarray,
    template: Template,
):
    # Prepare starters dictionary
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
                "outer_radius",
                "delta_radius",
                "sigma",
                "vel_width",
                "radius_ratio",
                "scale",
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

    starters = {k: v[0].item() for k, v in starters.items()}

    # Fixed and shared parameters
    fixed_vars = {
        f"{prof.name}_{param.name}": param.value
        for prof in template.disk_profiles + template.line_profiles
        for param in prof.fixed
    }

    shared_vars = {
        f"{prof.name}_{param.name}": starters[f"{param.shared}_{param.name}"]
        for prof in template.disk_profiles + template.line_profiles
        for param in prof.shared
        if f"{param.shared}_{param.name}" in starters
    }

    orphaned_vars = {
        f"{prof.name}_{param.name}": fixed_vars[f"{param.shared}_{param.name}"]
        for prof in template.disk_profiles + template.line_profiles
        for param in prof.shared
        if f"{param.shared}_{param.name}" in fixed_vars
    }

    starters.update(fixed_vars)
    starters.update(shared_vars)
    starters.update(orphaned_vars)

    starters.update(
        {
            f"{prof.name}_inclination_base": np.cos(
                starters[f"{prof.name}_inclination"]
            )
            for prof in template.disk_profiles
        }
    )

    return starters


def _plot_fit_results(
    rest_wave: np.ndarray,
    flux: np.ndarray,
    flux_err: np.ndarray,
    fit_z: float,
    fit_mod: Fittable1DModel,
    indices: np.ndarray,
    param_uncerts: np.ndarray,
    template: Template,
    starters: dict,
    out_dir: str | None = None,
    show_plot: bool = False,
):
    if out_dir is None:
        return

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
        f"LSQ Fit to {template.name} ({fit_z:.5f}, {template.redshift.value:.5f})"
    )

    for sm in fit_mod:
        if sm.name in ["shift", "base", "redshift"]:
            continue

        ax.plot(new_rest / (1 + fit_z), (fit_mod[0] | sm)(new_rest), label=f"{sm.name}")

    txt = ""
    for pn, pv, pe in zip(
        np.array(fit_mod.param_names)[indices],
        fit_mod.parameters[indices],
        param_uncerts,
    ):
        if "redshift_z" in pn:
            pn = "redshift"
            pv = fit_z
            start_val = template.redshift.value
        else:
            pn = "_".join([fit_mod[int(pn.split("_")[-1])].name] + pn.split("_")[:-1])
            start_val = starters.get(pn, np.nan)

        pn = pn.replace("halpha_", "")

        txt += f"{pn:15}: {pv:.3f} ({start_val:.3f}) \n"  # ± {pe:.3f}\n"

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

    # ax.legend()
    fig.savefig(Path(out_dir or "") / "lsq_model_fit.png")

    if not show_plot:
        plt.close(fig)


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

            if param.name in [
                "inner_radius",
                "outer_radius",
                "delta_radius",
                "sigma",
                "radius_ratio",
                "scale",
            ]:
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
                    in [
                        "inner_radius",
                        "outer_radius",
                        "delta_radius",
                        "sigma",
                        "radius_ratio",
                        "scale",
                    ]
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

    fit_mod = fitter(
        full_model,
        rest_wave,
        flux,
        maxiter=10_000,
        weights=1 / flux_err,
        filter_non_finite=True,
    )
    cov = fitter.fit_info["param_cov"]

    # Parameter uncertainties = sqrt of diagonal
    param_uncerts = np.sqrt(np.diag(cov))

    # Get real redshift
    fit_z = 1 / (1 + fit_mod["redshift"].z) - 1

    # Prepare starters dictionary
    starters = _construct_starters_dict(fit_mod, param_uncerts, template)

    # Plotting
    _plot_fit_results(
        rest_wave,
        flux,
        flux_err,
        fit_z,
        fit_mod,
        indices,
        param_uncerts,
        template,
        starters,
        out_dir,
        show_plot,
    )

    # Separate disk and line models
    disk_submodels = [
        fit_mod[sm_idx]
        for sm_idx in range(fit_mod.n_submodels)
        if fit_mod[sm_idx].name in [prof.name for prof in template.disk_profiles]
    ]
    disk_model = (
        np.sum(disk_submodels) if len(disk_submodels) > 0 else Const1D(amplitude=0)
    )

    line_submodels = [
        fit_mod[sm_idx]
        for sm_idx in range(fit_mod.n_submodels)
        if fit_mod[sm_idx].name in [prof.name for prof in template.line_profiles]
    ]
    line_model = (
        np.sum(line_submodels) if len(line_submodels) > 0 else Const1D(amplitude=0)
    )

    return (
        starters,
        rest_wave / (1 + fit_z),
        fit_mod(rest_wave),
        disk_model(rest_wave / (1 + fit_z)),
        line_model(rest_wave / (1 + fit_z)),
        fit_mod,
    )
