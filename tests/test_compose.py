import jax.numpy as jnp

from feadme.compose import _compute_line_flux_vectorized, _compute_disk_flux_vectorized


def computes_line_flux_correctly():
    wave = jnp.linspace(4000, 7000, 100)
    centers = jnp.array([5000, 6000])
    vel_widths = jnp.array([300, 400])
    amplitudes = jnp.array([1.0, 0.8])
    shapes = jnp.array([True, False])  # Gaussian and Lorentzian

    result = _compute_line_flux_vectorized(
        wave, centers, vel_widths, amplitudes, shapes
    )

    assert result.shape == wave.shape
    assert (result >= 0).all()


def handles_empty_line_flux():
    wave = jnp.linspace(4000, 7000, 100)
    centers = jnp.array([])
    vel_widths = jnp.array([])
    amplitudes = jnp.array([])
    shapes = jnp.array([])

    result = _compute_line_flux_vectorized(
        wave, centers, vel_widths, amplitudes, shapes
    )

    assert result.shape == wave.shape
    assert (result == 0).all()


def computes_disk_flux_correctly():
    wave = jnp.linspace(4000, 7000, 100)
    centers = jnp.array([5000])
    inner_radii = jnp.array([1.0])
    outer_radii = jnp.array([2.0])
    sigmas = jnp.array([0.5])
    inclinations = jnp.array([jnp.pi / 4])
    qs = jnp.array([2.0])
    eccentricities = jnp.array([0.1])
    apocenters = jnp.array([jnp.pi / 3])
    scales = jnp.array([1.0])
    offsets = jnp.array([0.0])

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
        scales,
        offsets,
    )

    assert result.shape == wave.shape
    assert (result >= 0).all()


def handles_empty_disk_flux():
    wave = jnp.linspace(4000, 7000, 100)
    centers = jnp.array([])
    inner_radii = jnp.array([])
    outer_radii = jnp.array([])
    sigmas = jnp.array([])
    inclinations = jnp.array([])
    qs = jnp.array([])
    eccentricities = jnp.array([])
    apocenters = jnp.array([])
    scales = jnp.array([])
    offsets = jnp.array([])

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
        scales,
        offsets,
    )

    assert result.shape == wave.shape
    assert (result == 0).all()
