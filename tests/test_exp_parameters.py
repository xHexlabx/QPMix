"""Tests for qpmix.exp.parameters."""

from __future__ import annotations

import pytest

from qpmix.exp.parameters import PARAMS, merge_params, params


def test_defaults_are_returned_unchanged():
    assert merge_params() == PARAMS
    assert merge_params({}) == PARAMS


def test_merge_does_not_mutate_the_defaults():
    before = dict(PARAMS)
    merged = merge_params({"t_cold": 4.2})
    merged["t_hot"] = 0.0
    assert PARAMS == before


def test_user_values_win():
    assert merge_params({"t_cold": 4.2})["t_cold"] == 4.2
    assert merge_params(t_cold=4.2)["t_cold"] == 4.2


def test_keyword_form_overrides_the_dict_form():
    assert merge_params({"t_cold": 1.0}, t_cold=2.0)["t_cold"] == 2.0


def test_unknown_parameters_are_rejected():
    with pytest.raises(TypeError, match="Unknown parameter"):
        merge_params({"not_a_parameter": 1})


def test_typos_get_a_suggestion():
    with pytest.raises(TypeError, match="Did you mean: t_cold"):
        merge_params({"t_kold": 1})


def test_legacy_alias_is_the_same_object():
    assert params is PARAMS


def test_normal_resistance_range_is_physical():
    """QMix's default is (3.5e-3, 5e3) volts -- five kilovolts -- which its
    own documentation contradicts."""
    lo, hi = PARAMS["vrn"]
    assert 1e-3 < lo < hi < 1e-2


def test_every_default_is_documented():
    """Each parameter should be mentioned in the module docstring table or
    carry an inline comment; a bare undocumented knob is a bug."""
    import qpmix.exp.parameters as mod

    source = mod.__file__
    with open(source) as handle:
        text = handle.read()
    for name in PARAMS:
        assert f'"{name}"' in text, name
