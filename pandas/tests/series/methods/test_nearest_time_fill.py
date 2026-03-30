"""Tests for Series.nearest_time_fill (GH#64074)."""
import numpy as np
import pytest

import pandas as pd
from pandas import (
    DataFrame,
    DatetimeIndex,
    Series,
    Timestamp,
)
import pandas._testing as tm


class TestNearestTimeFill:
    """Tests for Series.nearest_time_fill and DataFrame.nearest_time_fill."""

    def test_basic_series(self):
        # GH#64074 - basic nearest-in-time fill for Series
        idx = pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-06"]
        )
        s = Series([1.0, np.nan, np.nan, 4.0], index=idx)
        result = s.nearest_time_fill()
        expected = Series([1.0, 1.0, 4.0, 4.0], index=idx)
        tm.assert_series_equal(result, expected)

    def test_equidistant_prefers_forward(self):
        # When equidistant, the forward (earlier) value should be used
        idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        s = Series([1.0, np.nan, 3.0], index=idx)
        result = s.nearest_time_fill()
        # 2020-01-02 is 1 day from both neighbors; forward fill wins
        expected = Series([1.0, 1.0, 3.0], index=idx)
        tm.assert_series_equal(result, expected)

    def test_leading_nan(self):
        # Leading NaN has no forward neighbor; should be backward filled
        idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        s = Series([np.nan, 2.0, np.nan], index=idx)
        result = s.nearest_time_fill()
        expected = Series([2.0, 2.0, 2.0], index=idx)
        tm.assert_series_equal(result, expected)

    def test_all_nan_remains_nan(self):
        # All-NaN Series stays all-NaN
        idx = pd.to_datetime(["2020-01-01", "2020-01-02"])
        s = Series([np.nan, np.nan], index=idx)
        result = s.nearest_time_fill()
        expected = Series([np.nan, np.nan], index=idx)
        tm.assert_series_equal(result, expected)

    def test_no_nan(self):
        # Series without NaN values stays unchanged
        idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        s = Series([1.0, 2.0, 3.0], index=idx)
        result = s.nearest_time_fill()
        tm.assert_series_equal(result, s)

    def test_non_datetime_index_raises(self):
        # Should raise TypeError if the index is not a DatetimeIndex
        s = Series([1.0, np.nan, 3.0])
        with pytest.raises(TypeError, match="nearest_time_fill requires a DatetimeIndex"):
            s.nearest_time_fill()

    def test_inplace(self):
        # inplace=True should modify the Series in place
        idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-05"])
        s = Series([1.0, np.nan, 5.0], index=idx)
        original = s.copy()
        ret = s.nearest_time_fill(inplace=True)
        # inplace modifies self and returns self
        assert ret is s
        expected = Series([1.0, 1.0, 5.0], index=idx)
        tm.assert_series_equal(s, expected)

    def test_basic_dataframe(self):
        # GH#64074 - basic nearest-in-time fill for DataFrame
        idx = pd.to_datetime(
            ["2020-01-01", "2020-01-02", "2020-01-05", "2020-01-06"]
        )
        df = DataFrame(
            {
                "A": [1.0, np.nan, np.nan, 4.0],
                "B": [np.nan, 2.0, np.nan, 4.0],
            },
            index=idx,
        )
        result = df.nearest_time_fill()
        expected = DataFrame(
            {
                "A": [1.0, 1.0, 4.0, 4.0],
                "B": [2.0, 2.0, 4.0, 4.0],
            },
            index=idx,
        )
        tm.assert_frame_equal(result, expected)

    def test_irregular_index(self):
        # Verify behaviour on an irregular (non-uniform) datetime index
        idx = pd.to_datetime(
            ["2020-01-01", "2020-01-10", "2020-01-11", "2020-01-20"]
        )
        # NaN at position 1 is closer to position 0 (9 days) than to position 3 (10 days)
        # NaN at position 2 is closer to position 3 (9 days) than to position 0 (10 days)
        s = Series([0.0, np.nan, np.nan, 20.0], index=idx)
        result = s.nearest_time_fill()
        expected = Series([0.0, 0.0, 20.0, 20.0], index=idx)
        tm.assert_series_equal(result, expected)

    def test_does_not_modify_original(self):
        # The method must not modify the original object when inplace=False
        idx = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-03"])
        s = Series([1.0, np.nan, 3.0], index=idx)
        original_values = s.copy()
        _ = s.nearest_time_fill()
        tm.assert_series_equal(s, original_values)
