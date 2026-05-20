import jax.numpy as jnp

from feadme.core.parser import (
    Data,
    Disk,
    Distribution,
    Line,
    Mask,
    Parameter,
    Template,
)


def test_creates_data_with_correct_masking():
    wave = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    flux = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    flux_err = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mask = [Mask(lower_limit=2.0, upper_limit=4.0)]

    data = Data.create(wave, flux, flux_err, mask)

    assert jnp.array_equal(data.mask, jnp.array([False, True, True, True, False]))
    assert jnp.array_equal(data.masked_wave, jnp.array([2.0, 3.0, 4.0]))
    assert jnp.array_equal(data.masked_flux, jnp.array([20.0, 30.0, 40.0]))
    assert jnp.array_equal(data.masked_flux_err, jnp.array([0.2, 0.3, 0.4]))


def test_handles_empty_mask_correctly():
    wave = jnp.array([1.0, 2.0, 3.0])
    flux = jnp.array([10.0, 20.0, 30.0])
    flux_err = jnp.array([0.1, 0.2, 0.3])

    data = Data.create(wave, flux, flux_err, mask=None)

    assert jnp.array_equal(data.mask, jnp.array([True, True, True]))
    assert jnp.array_equal(data.masked_wave, wave)
    assert jnp.array_equal(data.masked_flux, flux)
    assert jnp.array_equal(data.masked_flux_err, flux_err)


def test_serializes_and_deserializes_template_correctly(tmp_path):
    template = Template.create(
        name="test_template",
        disk_profiles=[Disk(name="halpha_disk", center=6562.8)],
        line_profiles=[Line(name="halpha_narrow", center=6562.8)],
    )
    file_path = tmp_path / "template.json"
    template.to_json(file_path)

    loaded = Template.from_json(file_path)

    assert loaded.name == template.name
    assert len(loaded.disk_profiles) == len(template.disk_profiles)
    assert len(loaded.line_profiles) == len(template.line_profiles)
