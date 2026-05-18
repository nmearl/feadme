import astropy.constants as const
import astropy.units as u
import numpy as np
from astropy.modeling.core import CompoundModel
import operator
import loguru

logger = loguru.logger.opt(colors=True)

c_kms = const.c.to(u.km / u.s).value


def estimate_native_dv(wave):
    """
    Estimate the native logarithmic velocity sampling of a spectrum.

    Parameters
    ----------
    wave : array_like
        Observed-frame wavelength array in Angstroms.

    Returns
    -------
    dv_native : float
        Median native sampling in km/s.
    """
    wave = np.asarray(wave, dtype=float)
    good = np.isfinite(wave) & (wave > 0)
    wave = wave[good]

    if wave.size < 2:
        raise ValueError(
            "Need at least two valid wavelength points to estimate native dv."
        )

    dloglam = np.diff(np.log(wave))
    dloglam = dloglam[np.isfinite(dloglam) & (dloglam > 0)]

    if dloglam.size == 0:
        raise ValueError("Could not estimate native dv from wavelength array.")

    return c_kms * np.median(dloglam)


def suggested_dv_from_resolution(R=2000.0, samples_per_resel=3.0):
    if R <= 0 or samples_per_resel <= 0:
        raise ValueError("R and samples_per_resel must be positive.")
    return c_kms / R / samples_per_resel


def rebin_spectrum_logdv(
    wave,
    flux,
    flux_err,
    dv=None,
    *,
    R=None,
    samples_per_resel=3.0,
    min_points_per_bin=1,
    verbose=True,
):
    """
    Rebin a spectrum onto a constant-dv grid using inverse-variance weighted
    averaging of flux density. All inputs and outputs are in the observed frame.

    Parameters
    ----------
    wave : array_like
        Observed-frame wavelength array in Angstroms.
    flux : array_like
        Flux density array.
    flux_err : array_like
        1-sigma uncertainty array on flux density.
    dv : float, optional
        Desired bin width in km/s. If None, inferred from R and samples_per_resel.
    R : float, optional
        Instrumental resolving power. Used to infer dv if dv is None, and for
        sampling diagnostics when verbose=True.
    samples_per_resel : float, optional
        Desired number of samples per resolution element if dv is not supplied.
    min_points_per_bin : int, optional
        Minimum number of original points required to keep a bin.
    verbose : bool, optional
        If True, logger.debug a summary including rebin factor and sampling diagnostics.

    Returns
    -------
    wave_bin : ndarray
        Observed-frame bin-center wavelengths (ivar-weighted mean within each bin).
    flux_bin : ndarray
        Inverse-variance weighted mean flux density per bin.
    flux_err_bin : ndarray
        Propagated 1-sigma uncertainty per bin.
    info : dict
        Summary metadata.
    """
    wave = np.asarray(wave, dtype=float)
    flux = np.asarray(flux, dtype=float)
    flux_err = np.asarray(flux_err, dtype=float)

    if not (wave.shape == flux.shape == flux_err.shape):
        raise ValueError("wave, flux, and flux_err must have the same shape.")

    good = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & np.isfinite(flux_err)
        & (wave > 0)
        & (flux_err > 0)
    )

    wave_obs = wave[good]
    flux = flux[good]
    flux_err = flux_err[good]

    if wave_obs.size < 2:
        raise ValueError("Not enough valid spectral points after filtering.")

    order = np.argsort(wave_obs)
    wave_obs = wave_obs[order]
    flux = flux[order]
    flux_err = flux_err[order]

    dv_native = estimate_native_dv(wave_obs)

    if dv is None:
        if R is None:
            raise ValueError("Provide either dv or R.")
        dv = suggested_dv_from_resolution(R=R, samples_per_resel=samples_per_resel)

    if dv <= 0:
        raise ValueError("dv must be positive.")

    if dv < dv_native:
        logger.warning(
            f"Warning: requested dv ({dv:.1f} km/s) < native dv ({dv_native:.1f} km/s). "
            "This upsamples rather than bins — output pixels will not be independent."
        )

    # Build a uniform log-lambda grid in the observed frame
    dloglam = dv / c_kms
    loglam = np.log(wave_obs)
    loglam_min = loglam.min()
    loglam_max = loglam.max()

    nbins = int(np.ceil((loglam_max - loglam_min) / dloglam))
    loglam_edges = loglam_min + dloglam * np.arange(nbins + 1)

    inds = np.clip(np.digitize(loglam, loglam_edges) - 1, 0, nbins - 1)

    # Vectorized inverse-variance accumulation
    ivar = 1.0 / np.square(flux_err)

    wsum = np.bincount(inds, weights=ivar, minlength=nbins).astype(float)
    flux_num = np.bincount(inds, weights=ivar * flux, minlength=nbins).astype(float)
    wave_num = np.bincount(inds, weights=ivar * wave_obs, minlength=nbins).astype(float)
    counts_bin = np.bincount(inds, minlength=nbins)

    # A bin is valid if it has enough points and a finite positive ivar sum.
    # wave_bin is the ivar-weighted mean observed wavelength, which differs
    # slightly from the geometric log-lambda center for wide bins; this choice
    # keeps the output wavelengths tied directly to the input data.
    valid = (wsum > 0) & np.isfinite(wsum) & (counts_bin >= min_points_per_bin)

    safe_wsum = np.where(valid, wsum, 1.0)
    flux_bin = np.where(valid, flux_num / safe_wsum, np.nan)
    flux_err_bin = np.where(valid, np.sqrt(1.0 / safe_wsum), np.nan)
    wave_bin = np.where(valid, wave_num / safe_wsum, np.nan)

    keep = valid
    wave_bin = wave_bin[keep]
    flux_bin = flux_bin[keep]
    flux_err_bin = flux_err_bin[keep]

    dv_resolution = None
    samples_per_resel_effective = None
    if R is not None and R > 0:
        dv_resolution = c_kms / R
        samples_per_resel_effective = dv_resolution / dv

    rebin_factor = dv / dv_native

    if verbose:
        msg = (
            f"Native dv ≈ {dv_native:.2f} km/s; new dv = {dv:.2f} km/s "
            f"(rebin factor ≈ {rebin_factor:.2f}x); "
            f"{wave_obs.size} -> {wave_bin.size} bins."
        )
        if dv_resolution is not None:
            msg += (
                f" Instrumental Δv ≈ {dv_resolution:.2f} km/s "
                f"(~{samples_per_resel_effective:.2f} samples/resel)."
            )
        logger.debug(msg)

        if R is not None and samples_per_resel_effective is not None:
            if samples_per_resel_effective < 2.0:
                logger.debug(
                    "Warning: requested dv gives < 2 samples per resolution element; "
                    "this may undersample narrow features."
                )
            elif samples_per_resel_effective < 3.0:
                logger.debug(
                    "Note: requested dv gives < 3 samples per resolution element; "
                    "probably okay for broad-line work, but check narrow lines."
                )

    info = {
        "dv_native_obs": dv_native,
        "dv_new": dv,
        "rebin_factor": rebin_factor,
        "R": R,
        "dv_resolution": dv_resolution,
        "samples_per_resel_effective": samples_per_resel_effective,
        "n_input": wave_obs.size,
        "n_output": wave_bin.size,
    }

    return wave_bin, flux_bin, flux_err_bin, info


def convert_to_model_set(model, param_array):
    """
    Convert an existing compound model to a model set, preserving all operators.

    Parameters
    ----------
    model : astropy Model
        Existing model (compound or single)
    param_array : ndarray
        Parameter array of shape (n_models, n_params)

    Returns
    -------
    model_set : astropy Model
        Model set that can evaluate all parameter sets efficiently
    """
    n_models = param_array.shape[0]

    # Map operator symbols to actual operators
    OP_MAP = {
        "+": operator.add,
        "-": operator.sub,
        "*": operator.mul,
        "/": operator.truediv,
        "**": operator.pow,
        "|": operator.or_,
        "&": operator.and_,
    }

    def rebuild_model_tree(m, param_idx_ref):
        """
        Recursively rebuild the model tree with model sets.

        Parameters
        ----------
        m : Model
            Current model node
        param_idx_ref : list
            Mutable reference to current parameter index [idx]

        Returns
        -------
        new_model : Model
            Rebuilt model (set) for this subtree
        """
        if not isinstance(m, CompoundModel):
            # Leaf model - convert to model set
            n_params = len(m.param_names)
            submodel_params = param_array[
                :, param_idx_ref[0] : param_idx_ref[0] + n_params
            ]

            # Create parameter dict
            param_dict = {}
            for i, pname in enumerate(m.param_names):
                param_dict[pname] = submodel_params[:, i]

            # Create new model set
            model_class = type(m)
            new_submodel = model_class(n_models=n_models, **param_dict)

            param_idx_ref[0] += n_params
            return new_submodel
        else:
            # Compound model - recursively rebuild left and right
            # Access the internal structure
            # CompoundModel stores left, right, and operator

            # Try different attribute names (varies by astropy version)
            if hasattr(m, "_left"):
                left = m._left
                right = m._right
                op_symbol = m._operator
            elif hasattr(m, "left"):
                left = m.left
                right = m.right
                op_symbol = m.op
            else:
                # Fallback: use traverse to get submodels
                # This is less ideal but works
                submodels = [
                    sm
                    for sm in m.traverse_postorder()
                    if not isinstance(sm, CompoundModel)
                ]

                # For simple binary operations, assume first two are the operands
                if len(submodels) >= 2:
                    left_new = rebuild_model_tree(submodels[0], param_idx_ref)
                    right_new = rebuild_model_tree(submodels[1], param_idx_ref)
                    # Assume addition if we can't determine operator
                    return left_new + right_new
                else:
                    raise ValueError("Cannot determine compound model structure")

            # Recursively rebuild left and right subtrees
            left_new = rebuild_model_tree(left, param_idx_ref)
            right_new = rebuild_model_tree(right, param_idx_ref)

            # Apply the operator
            if isinstance(op_symbol, str):
                op_func = OP_MAP.get(op_symbol, operator.add)
            else:
                # op_symbol might already be a function
                op_func = op_symbol

            return op_func(left_new, right_new)

    # Start rebuilding from root with parameter index tracker
    param_idx_ref = [0]
    return rebuild_model_tree(model, param_idx_ref)
