import pytest
import jax.numpy as jnp
from pathlib import Path
from src.feadme.parser import (
    Data,
    Mask,
    Parameter,
    Distribution,
    Profile,
    Disk,
    Line,
    Template,
    Sampler,
    Config,
)


def creates_data_with_correct_masking():
    wave = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    flux = jnp.array([10.0, 20.0, 30.0, 40.0, 50.0])
    flux_err = jnp.array([0.1, 0.2, 0.3, 0.4, 0.5])
    mask = [Mask(lower_limit=2.0, upper_limit=4.0)]

    data = Data.create(wave, flux, flux_err, mask)

    assert jnp.array_equal(data.mask, jnp.array([False, True, True, True, False]))
    assert jnp.array_equal(data.masked_wave, jnp.array([2.0, 3.0, 4.0]))
    assert jnp.array_equal(data.masked_flux, jnp.array([20.0, 30.0, 40.0]))
    assert jnp.array_equal(data.masked_flux_err, jnp.array([0.2, 0.3, 0.4]))


def handles_empty_mask_correctly():
    wave = jnp.array([1.0, 2.0, 3.0])
    flux = jnp.array([10.0, 20.0, 30.0])
    flux_err = jnp.array([0.1, 0.2, 0.3])

    data = Data.create(wave, flux, flux_err, mask=None)

    assert jnp.array_equal(data.mask, jnp.array([True, True, True]))
    assert jnp.array_equal(data.masked_wave, wave)
    assert jnp.array_equal(data.masked_flux, flux)
    assert jnp.array_equal(data.masked_flux_err, flux_err)


def serializes_and_deserializes_template_correctly(tmp_path):
    template = Template(
        name="test_template",
        disk_profiles=[Disk(center=Parameter(name="center", value=1.0))],
        line_profiles=[Line(center=Parameter(name="line_center", value=2.0))],
        redshift=0.1,
    )
    file_path = tmp_path / "template.json"
    template.to_json(file_path)

    loaded_template = Template.from_json(file_path)

    assert loaded_template.name == template.name
    assert len(loaded_template.disk_profiles) == len(template.disk_profiles)
    assert len(loaded_template.line_profiles) == len(template.line_profiles)
    assert loaded_template.redshift == template.redshift


def creates_sampler_with_correct_chain_method():
    sampler = Sampler(sampler_type="NUTS", num_chains=2)

    assert sampler.chain_method == "parallel"

    sampler_single_chain = Sampler(sampler_type="NUTS", num_chains=1)

    assert sampler_single_chain.chain_method == "vectorized"
