"""Tests for qpmix.circuit."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.constants as sc

from qpmix.circuit import EmbeddingCircuit, read_circuit


def test_more_than_four_tones_is_allowed():
    """The four-tone cap belongs to the multi-dimensional kernels, not to
    the circuit; qpmix.multitone handles the rest."""
    cct = EmbeddingCircuit(num_f=9, num_p=1, vb_npts=11)
    assert cct.num_f == 9
    assert cct.freq.shape == (10,)


def test_shapes_follow_the_tone_harmonic_convention():
    cct = EmbeddingCircuit(num_f=3, num_p=2, vb_npts=77)
    assert cct.num_n == 6
    assert cct.freq.shape == (4,)
    assert cct.vt.shape == (4, 3)
    assert cct.zt.shape == (4, 3)
    assert cct.vb.shape == (77,)
    assert cct.initialize_vj().shape == (4, 3, 77)


def test_bias_sweep_spans_the_requested_range():
    cct = EmbeddingCircuit(vb_min=-1.5, vb_max=3.5, vb_npts=11)
    assert cct.vb[0] == pytest.approx(-1.5)
    assert cct.vb[-1] == pytest.approx(3.5)


def test_gap_frequency_and_voltage_are_consistent():
    cct = EmbeddingCircuit(vgap=2.8e-3, rn=14.0)
    assert cct.fgap == pytest.approx(2.8e-3 * sc.e / sc.h)
    assert cct.igap == pytest.approx(2.8e-3 / 14.0)

    other = EmbeddingCircuit(fgap=cct.fgap, rn=14.0)
    assert other.vgap == pytest.approx(2.8e-3)


def test_junction_properties_default_to_none():
    cct = EmbeddingCircuit()
    assert (cct.vgap, cct.fgap, cct.rn, cct.igap) == (None, None, None, None)


@pytest.mark.parametrize(
    ("value", "units", "expected_hz"),
    [
        (230e9, "Hz", 230e9),
        (230e3, "MHz", 230e9),
        (230, "GHz", 230e9),
        (0.23, "THz", 230e9),
    ],
)
def test_set_freq_frequency_units(value, units, expected_hz):
    cct = EmbeddingCircuit(vgap=2.8e-3, rn=14.0)
    cct.set_freq(value, units=units)
    assert cct.freq[1] * cct.fgap == pytest.approx(expected_hz)


def test_set_freq_voltage_units():
    cct = EmbeddingCircuit(vgap=2.8e-3, rn=14.0)
    cct.set_freq(1.4, units="mV")
    assert cct.freq[1] == pytest.approx(0.5)
    cct.set_freq(1.4e-3, units="V")
    assert cct.freq[1] == pytest.approx(0.5)


def test_set_freq_normalized_units_needs_no_junction():
    cct = EmbeddingCircuit()
    cct.set_freq(0.42, units="norm")
    assert cct.freq[1] == pytest.approx(0.42)


def test_set_freq_requires_a_gap_for_physical_units():
    cct = EmbeddingCircuit()
    with pytest.raises(ValueError, match="Gap frequency"):
        cct.set_freq(230, units="GHz")
    with pytest.raises(ValueError, match="Gap voltage"):
        cct.set_freq(1.0, units="mV")


def test_set_freq_rejects_unknown_units():
    cct = EmbeddingCircuit(vgap=2.8e-3, rn=14.0)
    with pytest.raises(ValueError, match="not recognized"):
        cct.set_freq(1.0, units="furlongs")


@pytest.mark.parametrize("units", ["W", "mW", "uW", "nW", "pW", "fW", "dBm", "dBW"])
def test_available_power_round_trips_through_every_unit(units):
    cct = EmbeddingCircuit(vgap=2.8e-3, rn=14.0)
    cct.zt[1, 1] = 0.3 - 0.3j
    cct.set_available_power(50e-9, units="W")
    target = cct.available_power(units=units)
    cct.vt[1, 1] = 0
    cct.set_available_power(target, units=units)
    assert cct.available_power(units="W") == pytest.approx(50e-9, rel=1e-9)


def test_available_power_is_zero_for_a_reactive_source():
    cct = EmbeddingCircuit(vgap=2.8e-3, rn=14.0)
    cct.zt[1, 1] = 1j
    cct.vt[1, 1] = 0.5
    assert cct.available_power() == 0.0


def test_power_helpers_require_junction_properties():
    cct = EmbeddingCircuit()
    cct.zt[1, 1] = 0.3
    with pytest.raises(ValueError, match="Gap voltage"):
        cct.available_power()
    with pytest.raises(ValueError, match="Gap voltage"):
        cct.set_available_power(1e-9)


def test_set_available_power_requires_an_embedding_impedance():
    cct = EmbeddingCircuit(vgap=2.8e-3, rn=14.0)
    with pytest.raises(ValueError, match="impedance not set"):
        cct.set_available_power(1e-9)


def test_power_helpers_reject_unknown_units():
    cct = EmbeddingCircuit(vgap=2.8e-3, rn=14.0)
    cct.zt[1, 1] = 0.3
    with pytest.raises(ValueError, match="unit for power"):
        cct.available_power(units="horsepower")
    with pytest.raises(ValueError, match="unit for power"):
        cct.set_available_power(1.0, units="horsepower")


def test_set_alpha_matches_its_own_definition():
    cct = EmbeddingCircuit(vgap=2.8e-3, rn=14.0)
    cct.freq[1] = 0.4
    cct.zt[1, 1] = 0.3
    cct.set_alpha(1.2, zj=0.66)
    # vj = vt * zj / (zj + zt), and alpha = |vj| / vph.
    vj = cct.vt[1, 1] * 0.66 / (0.66 + cct.zt[1, 1])
    assert abs(vj) / cct.freq[1] == pytest.approx(1.2)


def test_set_alpha_needs_frequency_and_impedance():
    cct = EmbeddingCircuit()
    with pytest.raises(ValueError, match="impedance"):
        cct.set_alpha(1.0)
    cct.zt[1, 1] = 0.3
    with pytest.raises(ValueError, match="Frequency"):
        cct.set_alpha(1.0)


def test_vph_is_an_alias_for_freq():
    cct = EmbeddingCircuit()
    cct.freq[1] = 0.5
    assert cct.vph is cct.freq


def test_lock_makes_arrays_read_only():
    cct = EmbeddingCircuit()
    cct.lock()
    with pytest.raises(ValueError):
        cct.freq[1] = 0.5
    cct.unlock()
    cct.freq[1] = 0.5
    assert cct.freq[1] == 0.5


def test_set_name_and_str():
    cct = EmbeddingCircuit(2, 1, name="my mixer")
    cct.set_name("LO", f=1, p=1)
    assert cct.comment[1][1] == "LO"
    assert "my mixer" in str(cct)
    assert "Tones:2" in str(cct)


def test_summary_mentions_every_signal():
    cct = EmbeddingCircuit(2, 2, vgap=2.8e-3, rn=14.0)
    cct.freq[1], cct.freq[2] = 0.3, 0.32
    cct.zt[1:, 1:] = 0.3
    text = cct.summary()
    for f in (1, 2):
        for p in (1, 2):
            assert f"f={f}, p={p}" in text


def test_summary_works_without_junction_properties():
    cct = EmbeddingCircuit()
    cct.freq[1] = 0.3
    assert "f=1, p=1" in cct.summary()


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_f": 0}, "int >= 1"),
        ({"num_f": 2.5}, "int >= 1"),
        ({"num_p": 0}, "int >= 1"),
        ({"num_p": 1.5}, "int >= 1"),
        ({"vb_npts": 0}, "int >= 1"),
        ({"vb_min": 3, "vb_max": 1}, "greater than or equal"),
    ],
)
def test_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        EmbeddingCircuit(**kwargs)


def _populated(tmp_path):
    cct = EmbeddingCircuit(2, 2, vb_npts=31, vb_max=3, vgap=2.8e-3, rn=14.0, name="x")
    cct.freq[1], cct.freq[2] = 0.3, 0.32
    for f in (1, 2):
        for p in (1, 2):
            cct.vt[f, p] = 0.4 / p + 0.1j
            cct.zt[f, p] = 0.3 - 0.2j
    cct.set_name("LO", 1, 1)
    return cct


def test_json_round_trip(tmp_path):
    cct = _populated(tmp_path)
    path = tmp_path / "cct.json"
    cct.to_json(path)
    back = read_circuit(path)
    assert back.num_f == cct.num_f
    assert back.num_p == cct.num_p
    assert back.name == cct.name
    assert back.comment[1][1] == "LO"
    assert np.allclose(back.freq, cct.freq)
    assert np.allclose(back.vt, cct.vt)
    assert np.allclose(back.zt, cct.zt)
    assert np.allclose(back.vb, cct.vb)
    assert back.vgap == cct.vgap


def test_legacy_text_round_trip(tmp_path):
    cct = _populated(tmp_path)
    path = tmp_path / "cct.txt"
    cct.save_info(path)
    back = read_circuit(path)
    assert back.num_f == 2
    assert back.num_p == 2
    assert np.allclose(back.freq, cct.freq, atol=1e-4)
    assert np.allclose(back.vt, cct.vt, atol=1e-4)
    assert np.allclose(back.zt, cct.zt, atol=1e-2)


def test_read_circuit_rejects_garbage(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("not a circuit file at all\n")
    with pytest.raises(ValueError, match="Could not parse"):
        read_circuit(path)
