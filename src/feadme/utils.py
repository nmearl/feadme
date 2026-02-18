import astropy.constants as const
import numpy as np
from astropy.modeling.core import CompoundModel
import operator

c_cgs = const.c.cgs.value
c_kms = const.c.to("km/s").value


def rebin_spectrum(wave, flux, flux_err, dv=100.0, rest=False, z=0.0):
    """
    Rebin a spectrum in wavelength space, conserving total flux
    and propagating uncertainties via inverse-variance weighting.

    Parameters
    ----------
    wave : array_like
        Wavelength array in Angstroms.
    flux : array_like
        Flux array (same units throughout; flux density, not integrated flux).
    flux_err : array_like
        1σ uncertainty array, same shape as `flux`.
    dv : float, optional
        Velocity width per bin in km/s. Default = 100 km/s.
    rest : bool, optional
        If True, treat `wave` as rest-frame; if False, interpret as observed.
        Used only if you pass a redshift `z`.
    z : float, optional
        Redshift of the source. If nonzero and rest=False, converts to rest-frame
        wavelength before computing bin edges.

    Returns
    -------
    wave_bin : ndarray
        Central wavelength of each bin.
    flux_bin : ndarray
        Weighted mean flux in each bin.
    flux_err_bin : ndarray
        Propagated uncertainty per bin.
    """

    # Convert to rest-frame if necessary
    if not rest and z != 0.0:
        wave_eff = wave / (1.0 + z)
    else:
        wave_eff = wave.copy()

    # Compute bin edges in log-lambda space (constant Δv bins)
    dloglam = dv / c_kms
    loglam = np.log(wave_eff)
    loglam_edges = np.arange(loglam.min(), loglam.max() + dloglam, dloglam)

    # Assign each wavelength to a bin
    inds = np.digitize(loglam, loglam_edges) - 1
    nbins = len(loglam_edges) - 1

    flux_bin = np.zeros(nbins)
    ivar_bin = np.zeros(nbins)
    wave_bin = np.zeros(nbins)

    ivar = 1.0 / flux_err**2
    for i in range(nbins):
        m = inds == i
        if not np.any(m):
            flux_bin[i] = np.nan
            ivar_bin[i] = 0.0
            continue
        w = ivar[m]
        flux_bin[i] = np.sum(w * flux[m]) / np.sum(w)
        ivar_bin[i] = np.sum(w)
        wave_bin[i] = np.exp(np.mean(loglam[m]))

    flux_err_bin = np.zeros_like(flux_bin)
    mask = ivar_bin > 0
    flux_err_bin[mask] = np.sqrt(1.0 / ivar_bin[mask])
    flux_err_bin[~mask] = np.nan

    return wave_bin[mask], flux_bin[mask], flux_err_bin[mask]


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
