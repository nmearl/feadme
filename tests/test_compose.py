import jax.numpy as jnp

from feadme.core.evaluators import (
    compose_param_arrays,
    _compute_disk_flux_vectorized,
    _compute_line_flux_vectorized,
)
from feadme.core.parser import Disk, Distribution, Parameter, Template
from feadme.core.integrators import mixed_jax_integrate, mixed_scalar_integrate


def test_computes_line_flux_correctly():
    wave = jnp.linspace(6400, 6700, 100)
    centers = jnp.array([6562.8, 6583.5])
    offsets = jnp.array([0.0, 0.0])
    vel_widths = jnp.array([300.0, 200.0])
    areas = jnp.array([1.0, 0.8])
    shapes = jnp.array([True, False])  # Gaussian and Lorentzian

    result = _compute_line_flux_vectorized(
        wave, centers, offsets, vel_widths, areas, shapes
    )

    assert result.shape == wave.shape
    assert (result >= 0).all()


def test_handles_empty_line_flux():
    wave = jnp.linspace(6400, 6700, 100)
    centers = jnp.array([])
    offsets = jnp.array([])
    vel_widths = jnp.array([])
    areas = jnp.array([])
    shapes = jnp.array([])

    result = _compute_line_flux_vectorized(
        wave, centers, offsets, vel_widths, areas, shapes
    )

    assert result.shape == wave.shape
    assert (result == 0).all()


def test_compose_derives_outer_radius_from_radius_ratio():
    template = Template.create(
        disk_profiles=[
            Disk(
                name="halpha_disk",
                center=6562.8,
                inner_radius=Parameter(
                    distribution=Distribution.LOG_UNIFORM, low=100.0, high=5000.0
                ),
                radius_ratio=Parameter(
                    distribution=Distribution.LOG_NORMAL,
                    low=2.0,
                    high=22.0,
                    loc=10.0,
                    scale=6.0,
                ),
                offset=Parameter(value=0.0, fixed=True),
                inclination=Parameter(
                    distribution=Distribution.UNIFORM, low=0.0, high=1.5
                ),
                sigma=Parameter(
                    distribution=Distribution.LOG_UNIFORM, low=200.0, high=3000.0
                ),
                q=Parameter(
                    distribution=Distribution.NORMAL,
                    low=1.0,
                    high=4.0,
                    loc=2.5,
                    scale=1.0,
                ),
                eccentricity=Parameter(
                    distribution=Distribution.UNIFORM, low=0.0, high=0.95
                ),
                apocenter=Parameter(
                    distribution=Distribution.UNIFORM, low=0.0, high=6.3
                ),
                area=Parameter(
                    distribution=Distribution.LOG_UNIFORM, low=1.0, high=100.0
                ),
                baseline=Parameter(value=0.0, fixed=True),
            )
        ]
    )
    mods = {
        "halpha_disk_offset": 0.0,
        "halpha_disk_inner_radius": 300.0,
        "halpha_disk_radius_ratio": 10.0,
        "halpha_disk_sigma": 500.0,
        "halpha_disk_inclination": 0.5,
        "halpha_disk_q": 2.0,
        "halpha_disk_eccentricity": 0.2,
        "halpha_disk_apocenter": 1.0,
        "halpha_disk_area": 10.0,
        "halpha_disk_baseline": 0.0,
    }

    disk_arrays, _ = compose_param_arrays(template, mods, redshift=0.0)

    assert disk_arrays["outer_radius"][0] == 3000.0


def test_computes_disk_flux_correctly():
    wave = jnp.linspace(6400, 6700, 100)
    centers = jnp.array([6562.8])
    inner_radii = jnp.array([400.0])
    outer_radii = jnp.array([5000.0])
    sigmas = jnp.array([500.0])
    inclinations = jnp.array([jnp.pi / 4])
    qs = jnp.array([2.0])
    eccentricities = jnp.array([0.1])
    apocenters = jnp.array([jnp.pi / 3])
    areas = jnp.array([1.0])
    offsets = jnp.array([0.0])
    baselines = jnp.array([0.0])

    result = _compute_disk_flux_vectorized(
        wave,
        centers,
        inner_radii,
        outer_radii,
        sigmas,
        inclinations,
        qs,
        eccentricities,
        apocenters,
        areas,
        offsets,
        baselines,
        mixed_jax_integrate,
        mixed_scalar_integrate,
    )

    assert result.shape == wave.shape
    assert jnp.isfinite(result).all()
    assert (result >= 0).all()
