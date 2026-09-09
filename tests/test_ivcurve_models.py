"""Tests for qpmix.mathfn.ivcurve_models."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from qpmix.mathfn import ivcurve_models as iv

MODELS = [iv.perfect, iv.polynomial, iv.exponential, iv.expanded]


@pytest.mark.parametrize("model", MODELS)
def test_models_are_odd_functions(model):
    v = np.linspace(0.05, 5, 501)
    assert np.abs(model(v) + model(-v)).max() < 1e-9


@pytest.mark.parametrize("model", MODELS)
def test_models_are_ohmic_well_above_the_gap(model):
    """Slope 1 in normalized units.  ``expanded`` adds a constant offset by
    design (the reduced current amplitude seen above the gap), so the slope
    is the invariant, not the value."""
    v = np.linspace(5, 30, 251)
    slope = np.gradient(model(v), v)
    assert np.abs(slope - 1.0).max() < 1e-6


def test_expanded_has_a_constant_current_offset_above_the_gap():
    v = np.linspace(5, 30, 251)
    offset = iv.expanded(v) - v
    assert np.ptp(offset) < 1e-9
    assert offset[0] == pytest.approx(-0.1, abs=1e-6)


@pytest.mark.parametrize("model", MODELS)
def test_models_are_monotonic(model):
    v = np.linspace(-6, 6, 4001)
    assert np.all(np.diff(model(v)) >= -1e-9)


@pytest.mark.parametrize("model", MODELS)
def test_scalar_input_returns_a_float(model):
    out = model(2.0)
    assert isinstance(out, float)
    assert out == pytest.approx(float(model(np.array([2.0]))[0]))


def test_perfect_is_a_step():
    assert iv.perfect(np.array([0.0, 0.5, 0.999])).tolist() == [0.0, 0.0, 0.0]
    assert iv.perfect(1.0) == 0.5
    assert iv.perfect(-1.0) == -0.5
    assert iv.perfect(3.0) == 3.0


def test_perfect_kk_is_even():
    v = np.linspace(0.05, 20, 401)
    assert np.abs(iv.perfect_kk(v) - iv.perfect_kk(-v)).max() < 1e-12


def test_perfect_kk_uses_the_sentinel_at_the_singularity():
    assert iv.perfect_kk(1.0, max_kk=42.0) == 42.0
    assert iv.perfect_kk(-1.0, max_kk=42.0) == 42.0
    assert np.isnan(iv.perfect_kk(1.0, max_kk=np.nan))


def test_perfect_kk_decays_far_from_the_gap():
    assert abs(iv.perfect_kk(1000.0)) < 0.01


@pytest.mark.parametrize("order", [10, 30, 50, 100, 200])
def test_polynomial_never_overflows(order):
    """The textbook form overflows float64 past order ~50; this one cannot."""
    v = np.linspace(-35, 35, 2001)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = iv.polynomial(v, order)
    assert np.all(np.isfinite(out))


def test_polynomial_matches_the_textbook_form_where_that_is_safe():
    v = np.linspace(-4, 4, 1001)
    order = 20
    expected = v ** (2 * order + 1) / (1 + v ** (2 * order))
    assert np.abs(iv.polynomial(v, order) - expected).max() < 1e-9


def test_higher_polynomial_order_sharpens_the_transition():
    v = np.linspace(0.8, 1.2, 401)
    width = [np.abs(np.gradient(iv.polynomial(v, n))).max() for n in (20, 60)]
    assert width[1] > width[0]


def test_polynomial_passes_through_the_gap_at_one_half():
    assert iv.polynomial(1.0, 50) == pytest.approx(0.5)


def test_exponential_subgap_resistance_is_as_specified():
    """The fixed model must reproduce the requested subgap resistance."""
    vgap, rn, rsg = 2.8e-3, 14.0, 300.0
    v = np.linspace(0.2, 0.6, 201)
    current = iv.exponential(v, vgap, rn, rsg, model="fixed")
    # Normalized slope * rn is the conductance in units of 1/rsg.
    slope = np.polyfit(v, current, 1)[0]
    assert slope * rsg / rn == pytest.approx(1.0, rel=0.05)


def test_original_exponential_model_has_half_the_subgap_resistance():
    v = np.linspace(0.2, 0.6, 201)
    fixed = np.polyfit(v, iv.exponential(v, model="fixed"), 1)[0]
    original = np.polyfit(v, iv.exponential(v, model="original"), 1)[0]
    assert original / fixed == pytest.approx(2.0, rel=0.05)


def test_exponential_rejects_unknown_model():
    with pytest.raises(ValueError, match="not recognized"):
        iv.exponential(np.linspace(0, 2, 10), model="quantum")


def test_expanded_leakage_current_shows_up_near_zero_bias():
    v = np.linspace(0.05, 0.3, 101)
    with_leak = iv.expanded(v, ileak=5e-5)
    without = iv.expanded(v, ileak=0.0)
    assert np.all(with_leak > without)


def test_models_do_not_mutate_numpy_error_state():
    """QMix calls np.seterr globally; QPMix must not."""
    before = np.geterr()
    v = np.linspace(-35, 35, 1001)
    iv.exponential(v)
    iv.expanded(v)
    assert np.geterr() == before


def test_logistic_is_stable_at_extreme_arguments():
    z = np.array([-1e6, -800.0, 0.0, 800.0, 1e6])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = iv._logistic(z)
    assert np.all(np.isfinite(out))
    assert out[0] == pytest.approx(1.0)
    assert out[2] == pytest.approx(0.5)
    assert out[-1] == pytest.approx(0.0)
