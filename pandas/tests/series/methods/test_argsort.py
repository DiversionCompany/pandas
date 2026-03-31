import numpy as np
import pytest

from pandas import (
    Series,
    Timestamp,
    isna,
)
import pandas._testing as tm


class TestSeriesArgsort:
    def test_argsort_axis(self):
        # GH#54257
        ser = Series(range(3))

        msg = "No axis named 2 for object type Series"
        with pytest.raises(ValueError, match=msg):
            ser.argsort(axis=2)

    def test_argsort_numpy(self, datetime_series):
        ser = datetime_series
        res = np.argsort(ser).values
        expected = np.argsort(np.array(ser))
        tm.assert_numpy_array_equal(res, expected)

    def test_argsort_numpy_missing(self):
        data = [0.1, np.nan, 0.2, np.nan, 0.3]
        ser = Series(data)
        result = np.argsort(ser)
        expected = np.argsort(np.array(data))

        tm.assert_numpy_array_equal(result.values, expected)

    def test_argsort(self, datetime_series):
        argsorted = datetime_series.argsort()
        assert issubclass(argsorted.dtype.type, np.integer)

    def test_argsort_dt64(self, unit):
        # GH#2967 (introduced bug in 0.11-dev I think)
        ser = Series(
            [Timestamp(f"201301{i:02d}") for i in range(1, 6)], dtype=f"M8[{unit}]"
        )
        assert ser.dtype == f"datetime64[{unit}]"
        shifted = ser.shift(-1)
        assert shifted.dtype == f"datetime64[{unit}]"
        assert isna(shifted[4])

        result = ser.argsort()
        expected = Series(range(5), dtype=np.intp)
        tm.assert_series_equal(result, expected)

        result = shifted.argsort()
        expected = Series([*list(range(4)), 4], dtype=np.intp)
        tm.assert_series_equal(result, expected)

    def test_argsort_stable(self):
        ser = Series(np.random.default_rng(2).integers(0, 100, size=10000))
        mindexer = ser.argsort(kind="mergesort")
        qindexer = ser.argsort()

        mexpected = np.argsort(ser.values, kind="mergesort")
        qexpected = np.argsort(ser.values, kind="quicksort")

        tm.assert_series_equal(mindexer.astype(np.intp), Series(mexpected))
        tm.assert_series_equal(qindexer.astype(np.intp), Series(qexpected))
        msg = (
            r"ndarray Expected type <class 'numpy\.ndarray'>, "
            r"found <class 'pandas\.Series'> instead"
        )
        with pytest.raises(AssertionError, match=msg):
            tm.assert_numpy_array_equal(qindexer, mindexer)

    def test_argsort_preserve_name(self, datetime_series):
        result = datetime_series.argsort()
        assert result.name == datetime_series.name

    def test_argsort_stable_parameter(self):
        # GH#64255: stable=True should use a stable sort algorithm,
        # equivalent to kind='stable'. Previously, stable was silently ignored.
        rng = np.random.default_rng(42)
        ser = Series(rng.integers(0, 100, size=10000))

        # stable=True should produce the same result as kind='stable'
        result_stable_true = ser.argsort(stable=True)
        result_stable_kind = ser.argsort(kind="stable")
        tm.assert_series_equal(result_stable_true, result_stable_kind)

        # stable=True should produce the same result as kind='mergesort'
        result_mergesort = ser.argsort(kind="mergesort")
        tm.assert_series_equal(result_stable_true, result_mergesort)

        # stable=True should give stable sort (different from quicksort for ties)
        result_quicksort = ser.argsort(kind="quicksort")
        # The indices themselves might differ for ties, so we just verify
        # the values are in sorted order for both
        assert (ser.iloc[result_stable_true.values].diff().dropna() >= 0).all()
        assert (ser.iloc[result_quicksort.values].diff().dropna() >= 0).all()

    def test_argsort_stable_false_keeps_kind(self):
        # GH#64255: stable=False (or None) should not override kind
        rng = np.random.default_rng(42)
        ser = Series(rng.integers(0, 100, size=1000))

        result_stable_none = ser.argsort(kind="mergesort", stable=None)
        result_mergesort = ser.argsort(kind="mergesort")
        tm.assert_series_equal(result_stable_none, result_mergesort)

        result_stable_false = ser.argsort(kind="mergesort", stable=False)
        tm.assert_series_equal(result_stable_false, result_mergesort)

    def test_argsort_stable_true_with_ties(self):
        # GH#64255: stable=True should preserve original order for equal elements
        # Verify that stable sort maintains relative order of equal elements
        ser = Series([3, 1, 2, 1, 3])
        result = ser.argsort(stable=True)
        # Using stable=True: among ties for value=1, index 1 should come before 3
        # Among ties for value=3, index 0 should come before 4
        expected = ser.argsort(kind="stable")
        tm.assert_series_equal(result, expected)
        # The relative order of equal elements should be preserved
        assert list(result.values) == list(expected.values)
