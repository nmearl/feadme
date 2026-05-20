import jax.numpy as jnp

from feadme.core.evaluators import (
    _compute_disk_flux_vectorized,
    _compute_line_flux_vectorized,
)
from feadme.core.integrators import mixed_jax_integrate, mixed_scalar_integrate


def test_computes_line_flux_correctly():
    wave = jnp.linspace(6400, 6700, 100)
    centers = jnp.array([6562.8, 6583.5])
    offsets = jnp.array([0.0, 0.0])
    vel_widths = jnp.array([300.0, 200.0])
    areas = jnp.array([1.0, 0.8])
    shapes = jnp.array([True, False])  # Gaussian and Lorentzian

    result = _compute_line_flux_vectorized(wave, centers, offsets, vel_widths, areas, shapes)

    assert result.shape == wave.shape
    assert (result >= 0).all()


def test_handles_empty_line_flux():
    wave = jnp.linspace(6400, 6700, 100)
    centers = jnp.array([])
    offsets = jnp.array([])
    vel_widths = jnp.array([])
    areas = jnp.array([])
    shapes = jnp.array([])

    result = _compute_line_flux_vectorized(wave, centers, offsets, vel_widths, areas, shapes)

    assert result.shape == wave.shape
    assert (result == 0).all()


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
