"""Unit tests for the vectorized plot unit conversion helper.

``to_display_magnitudes`` replaces per-point ``Characteristic.to_unit`` in the
plot path with a single pint-pandas conversion. These tests assert it is a
faithful vectorized pint conversion (matching per-point pint on the shared
application registry, including an offset unit) and that it falls back to the
raw floats rather than raising.
"""

import numpy as np
import pytest

pytest.importorskip("pandas")
pytest.importorskip("pint_pandas")

import pint  # noqa: E402

from opensemantic.base.view._channel_utils import (  # noqa: E402
    align_pint_pandas_registry,
    to_display_magnitudes,
)

align_pint_pandas_registry()
_ureg = pint.get_application_registry()


def _per_point(values, src, dst):
    return [_ureg.Quantity(float(v), src).to(dst).magnitude for v in values]


def test_linear_conversion_matches_per_point():
    vals = [0.0, 1.5, 1000.0, -42.0]
    got = to_display_magnitudes(vals, "meter", "millimeter")
    ref = _per_point(vals, "meter", "millimeter")
    assert np.allclose(np.asarray(got, float), ref)


def test_offset_conversion_matches_per_point():
    # Kelvin -> degree_Celsius is affine, not a pure scale; values like ~293 K.
    vals = [273.15, 293.15, 300.0]
    got = to_display_magnitudes(vals, "kelvin", "degree_Celsius")
    ref = _per_point(vals, "kelvin", "degree_Celsius")
    assert np.allclose(np.asarray(got, float), ref)


def test_identity_and_missing_units_return_raw():
    vals = [1.0, 2.0, 3.0]
    assert np.allclose(to_display_magnitudes(vals, "meter", "meter"), vals)
    assert np.allclose(to_display_magnitudes(vals, None, "meter"), vals)
    assert np.allclose(to_display_magnitudes(vals, "meter", None), vals)


def test_unparseable_unit_falls_back_to_raw():
    vals = [1.0, 2.0, 3.0]
    # A bogus target unit must not raise; it degrades to the raw magnitudes.
    out = np.asarray(to_display_magnitudes(vals, "meter", "not_a_unit_xyz"), float)
    assert np.allclose(out, vals)


def test_returns_float_array_length_preserved():
    vals = [10, 20, 30, 40]
    out = to_display_magnitudes(vals, "kelvin", "degree_Celsius")
    out = np.asarray(out, float)
    assert out.shape == (4,)
    assert out.dtype == np.float64
