import jax
import jax.numpy as jnp
import numpy as np
from numpyro.infer.initialization import init_to_value
from numpyro.infer.util import initialize_model
import jax.random as random
import loguru

logger = loguru.logger.opt(colors=True)


def _resolve_param_value(init_params, config, profile_name, field_name):
    full_name = f"{profile_name}_{field_name}"
    if full_name in init_params:
        return float(init_params[full_name])

    param_ref = next((p for p in config.template.iter_all if p.name == full_name), None)
    if param_ref is None:
        return None

    if param_ref.param.shared is not None:
        target = param_ref.target_name
        if target in init_params:
            return float(init_params[target])
        target_ref = next((p for p in config.template.iter_all if p.name == target), None)
        if target_ref is not None and target_ref.param.value is not None:
            return float(target_ref.param.value)

    if param_ref.param.value is not None:
        return float(param_ref.param.value)

    return None


def _log_term_stats(name, arr):
    arr = np.asarray(arr, dtype=float)
    finite = np.isfinite(arr)
    finite_count = int(np.sum(finite))
    total = int(arr.size)

    if finite_count == 0:
        logger.warning(f"    {name}: all values are non-finite")
        return

    finite_arr = arr[finite]
    nonpos = np.sum(finite_arr <= 0.0)
    logger.warning(
        f"    {name}: min={np.min(finite_arr):.4g} max={np.max(finite_arr):.4g} "
        f"nonfinite_frac={(total - finite_count) / total:.3%} "
        f"nonpos_frac={nonpos / finite_count:.3%}"
    )


def diagnose_disk_failure(init_params, config, *, n_xi=64, n_phi=128, n_x=256):
    if not config.template.disk_profiles:
        return

    redshift = float(init_params.get("redshift", config.template.redshift.value))
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)

    for disk_idx, disk in enumerate(config.template.disk_profiles):
        profile_name = disk.name
        inner_radius = _resolve_param_value(init_params, config, profile_name, "inner_radius")
        outer_radius = _resolve_param_value(init_params, config, profile_name, "outer_radius")
        inclination = _resolve_param_value(init_params, config, profile_name, "inclination")
        sigma = _resolve_param_value(init_params, config, profile_name, "sigma")
        q = _resolve_param_value(init_params, config, profile_name, "q")
        eccentricity = _resolve_param_value(init_params, config, profile_name, "eccentricity")
        apocenter = _resolve_param_value(init_params, config, profile_name, "apocenter")
        center = float(disk.center) if disk.center is not None else None

        if any(v is None for v in [inner_radius, outer_radius, inclination, sigma, q, eccentricity, apocenter, center]):
            logger.warning(f"  Disk diagnostics skipped for {profile_name}: missing resolved parameter(s)")
            continue

        center_obs = float(center) * (1.0 + redshift)
        wave_min = min(m.lower_limit for m in config.template.mask) if config.template.mask else center_obs * 0.9
        wave_max = max(m.upper_limit for m in config.template.mask) if config.template.mask else center_obs * 1.1
        x_lo = (center_obs - wave_max) / center_obs
        x_hi = (center_obs - wave_min) / center_obs
        x_grid = np.linspace(x_lo, x_hi, n_x)
        xi_log = np.linspace(np.log10(float(inner_radius)), np.log10(float(outer_radius)), n_xi)
        xi_tilde = 10.0 ** xi_log

        phi_3d = phi[None, :, None]
        xi_3d = xi_tilde[:, None, None]
        x_3d = x_grid[None, None, :]

        sini = np.sin(float(inclination))
        sinphi = np.sin(phi_3d)
        cosphi = np.cos(phi_3d)
        phi_diff = phi_3d - float(apocenter)
        sinphiphinot = np.sin(phi_diff)
        cosphiphinot = np.cos(phi_diff)

        sini_cosphi = sini * cosphi
        one_minus_sinisq_cosphisq = 1.0 - sini_cosphi**2
        one_minus_e_cosphiphinot = 1.0 - float(eccentricity) * cosphiphinot
        trans_fac = (1.0 + float(eccentricity)) / one_minus_e_cosphiphinot
        xi = xi_3d * trans_fac
        xi_recip = 1.0 / xi
        scale = 1.0 - 2.0 * xi_recip

        one_plus_sini_cosphi = 1.0 + sini_cosphi
        one_minus_sini_cosphi = 1.0 - sini_cosphi
        psi_geom = one_minus_sini_cosphi / one_plus_sini_cosphi
        b_div_r = np.sqrt(one_minus_sinisq_cosphisq) * (1.0 + xi_recip * psi_geom)

        e_sq_sin_sq = float(eccentricity) ** 2 * sinphiphinot**2
        one_minus_e_cosphiphinot_sq = one_minus_e_cosphiphinot**2
        gamma_raw = 1.0 - (e_sq_sin_sq + scale * one_minus_e_cosphiphinot_sq) / (
            xi * scale**2 * one_minus_e_cosphiphinot
        )

        term_raw = 1.0 - b_div_r * b_div_r * scale
        sqrt_xi = np.sqrt(xi)
        sqrt_scale = np.sqrt(scale)
        sqrt_one_minus_e_cosphiphinot = np.sqrt(one_minus_e_cosphiphinot)
        sqrt_one_minus_sinisq_cosphisq = np.sqrt(one_minus_sinisq_cosphisq)

        dc_val = sqrt_xi * scale * sqrt_scale * sqrt_one_minus_e_cosphiphinot
        de_val = sqrt_xi * sqrt_scale * sqrt_one_minus_sinisq_cosphisq
        gamma = np.sqrt(1.0 / gamma_raw)
        db_num = np.sqrt(term_raw) * float(eccentricity) * sinphiphinot
        dd_num = b_div_r * sqrt_one_minus_e_cosphiphinot * sini * sinphi
        inv_dop_raw = gamma * ((1.0 / sqrt_scale) - db_num / dc_val + dd_num / de_val)
        D = 1.0 / inv_dop_raw
        exponent = -((1.0 + x_3d - D) ** 2) / (2.0 * D * D) * ((299792.458 / float(sigma)) ** 2)

        logger.warning(
            f"  Disk algebra diagnostics for {profile_name}: "
            f"center_obs={center_obs:.3f} inner={float(inner_radius):.4g} "
            f"outer={float(outer_radius):.4g} inc={float(inclination):.4g} "
            f"e={float(eccentricity):.4g} apo={float(apocenter):.4g}"
        )
        _log_term_stats("one_minus_sinisq_cosphisq", one_minus_sinisq_cosphisq)
        _log_term_stats("one_minus_e_cosphiphinot", one_minus_e_cosphiphinot)
        _log_term_stats("xi", xi)
        _log_term_stats("scale", scale)
        _log_term_stats("one_plus_sini_cosphi", one_plus_sini_cosphi)
        _log_term_stats("gamma_raw", gamma_raw)
        _log_term_stats("term_raw", term_raw)
        _log_term_stats("dc_val", dc_val)
        _log_term_stats("de_val", de_val)
        _log_term_stats("inv_dop_raw", inv_dop_raw)
        _log_term_stats("exponent", exponent)


def jitter_unconstrained(init_params, key, scale=0.05):
    return jax.tree.map(
        lambda x, k: x + scale * jax.random.normal(k, x.shape),
        init_params,
        jax.tree.unflatten(
            jax.tree.structure(init_params),
            jax.random.split(key, len(jax.tree.leaves(init_params))),
        ),
    )


def make_init_params(model, model_args, model_kwargs, base_values, rng_keys):
    """
    Convert constrained starting point(s) to unconstrained init_params for
    mcmc.run(), with small jitter applied to each chain.

    Parameters
    ----------
    base_values : dict | list[dict]
        Either a single constrained parameter dict (all chains share it) or
        a list of dicts of length num_chains (one starting point per chain).
    rng_keys : array
        Per-chain RNG keys, shape (num_chains, 2).
    """
    if isinstance(base_values, dict):
        base_values = [base_values] * len(rng_keys)

    inits = []
    for values, k in zip(base_values, rng_keys):
        init_params, *_ = initialize_model(
            k,
            model,
            model_args=model_args,
            model_kwargs=model_kwargs,
            init_strategy=init_to_value(values=values),
        )
        inits.append(jitter_unconstrained(init_params, k, scale=0.02))

    return jax.tree.map(lambda *xs: jnp.stack(xs, 0), *inits)


def diagnose_init_params(model, init_params, config):
    rng = random.PRNGKey(0)
    model_kwargs = dict(
        wave=config.data.masked_wave,
        flux=config.data.masked_flux,
        flux_err=config.data.masked_flux_err,
    )

    # First check if the full set fails
    try:
        initialize_model(
            rng,
            model,
            model_kwargs=model_kwargs,
            init_strategy=init_to_value(values=init_params),
        )
        logger.info("Full init_params: OK")
        return
    except Exception as e:
        logger.warning(f"Full init_params failed: {e}")
        diagnose_disk_failure(init_params, config)

    # Binary-search: try dropping each param one at a time
    bad_params = []
    for key in init_params:
        reduced = {k: v for k, v in init_params.items() if k != key}
        try:
            initialize_model(
                rng,
                model,
                model_kwargs=model_kwargs,
                init_strategy=init_to_value(values=reduced),
            )
            bad_params.append(key)
            logger.warning(f"  Removing '{key}' fixes initialization — likely culprit")
        except Exception:
            pass  # Still broken without this param, not the sole cause

    if not bad_params:
        logger.warning("No single parameter identified — may be an interaction effect")

    # For each suspected bad param, log its value vs prior bounds
    for key in bad_params:
        val = init_params[key]
        param_ref = next((p for p in config.template.iter_all if p.name == key), None)
        if param_ref and param_ref.param.low is not None and param_ref.param.high is not None:
            logger.warning(
                f"  {key} = {val:.4g}  bounds=[{param_ref.param.low:.4g}, {param_ref.param.high:.4g}]"
                f"  {'OUT OF BOUNDS' if val <= param_ref.param.low or val >= param_ref.param.high else 'in bounds'}"
            )
        else:
            logger.warning(
                f"  {key} = {val:.4g}  (no matching template param found — may be a _base site)"
            )
