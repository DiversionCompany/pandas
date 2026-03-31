import numpy as np
import pytest

from pandas import (
    Index,
    to_datetime,
    to_timedelta,
)
import pandas._testing as tm


class TestAstype:
    def test_astype_float64_to_uint64(self):
        # GH#45309 used to incorrectly return Index with int64 dtype
        idx = Index([0.0, 5.0, 10.0, 15.0, 20.0], dtype=np.float64)
        result = idx.astype("u8")
        expected = Index([0, 5, 10, 15, 20], dtype=np.uint64)
        tm.assert_index_equal(result, expected, exact=True)

        idx_with_negatives = idx - 10
        with pytest.raises(ValueError, match="losslessly"):
            idx_with_negatives.astype(np.uint64)

    def test_astype_float64_to_object(self):
        float_index = Index([0.0, 2.5, 5.0, 7.5, 10.0], dtype=np.float64)
        result = float_index.astype(object)
        assert result.equals(float_index)
        assert float_index.equals(result)
        assert isinstance(result, Index) and result.dtype == object

    def test_astype_float64_mixed_to_object(self):
        # mixed int-float
        idx = Index([1.5, 2, 3, 4, 5], dtype=np.float64)
        idx.name = "foo"
        result = idx.astype(object)
        assert result.equals(idx)
        assert idx.equals(result)
        assert isinstance(result, Index) and result.dtype == object

    @pytest.mark.parametrize("dtype", ["int16", "int32", "int64"])
    def test_astype_float64_to_int_dtype(self, dtype):
        # GH#12881
        # a float astype int
        idx = Index([0, 1, 2], dtype=np.float64)
        result = idx.astype(dtype)
        expected = Index([0, 1, 2], dtype=dtype)
        tm.assert_index_equal(result, expected, exact=True)

        idx = Index([0, 1.1, 2], dtype=np.float64)
        result = idx.astype(dtype)
        expected = Index([0, 1, 2], dtype=dtype)
        tm.assert_index_equal(result, expected, exact=True)

    @pytest.mark.parametrize("dtype", ["float32", "float64"])
    def test_astype_float64_to_float_dtype(self, dtype):
        # GH#12881
        # a float astype int
        idx = Index([0, 1, 2], dtype=np.float64)
        result = idx.astype(dtype)
        assert isinstance(result, Index) and result.dtype == dtype

    @pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
    def test_astype_float_to_datetimelike(self, dtype):
        # GH#49660 pre-2.0 Index.astype from floating to M8/m8/Period raised,
        #  inconsistent with Series.astype
        idx = Index([0, 1.1, 2], dtype=np.float64)

        result = idx.astype(dtype)
        if dtype[0] == "M":
            expected = to_datetime(idx.values)
        else:
            expected = to_timedelta(idx.values)
        tm.assert_index_equal(result, expected)

        # check that we match Series behavior
        result = idx.to_series().set_axis(range(3)).astype(dtype)
        expected = expected.to_series().set_axis(range(3))
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("dtype", [int, "int16", "int32", "int64"])
    @pytest.mark.parametrize("non_finite", [np.inf, np.nan])
    def test_cannot_cast_inf_to_int(self, non_finite, dtype):
        # GH#13149
        idx = Index([1, 2, non_finite], dtype=np.float64)

        msg = r"Cannot convert non-finite values \(NA or inf\) to integer"
        with pytest.raises(ValueError, match=msg):
            idx.astype(dtype)

    def test_astype_from_object(self):
        index = Index([1.0, np.nan, 0.2], dtype="object")
        result = index.astype(float)
        expected = Index([1.0, np.nan, 0.2], dtype=np.float64)
        assert result.dtype == expected.dtype
        tm.assert_index_equal(result, expected)

    def test_astype_numpy_string_dtype(self):
        # GH#50127: Index.astype(<numpy string dtype>) should not raise
        # NotImplementedError. Result dtype should be object (pandas does not
        # store numpy str/U dtypes internally).
        idx = Index([1, 2, 3])
        for str_dtype in [np.str_, np.dtype("U"), np.dtype("U10"), "U10"]:
            result = idx.astype(str_dtype)
            assert result.dtype == object, (
                f"astype({str_dtype!r}) should give object dtype, got {result.dtype}"
            )
            tm.assert_index_equal(result, Index(["1", "2", "3"], dtype=object))

        # Also verify on float Index
        idx_float = Index([1.0, 2.0, 3.0])
        result = idx_float.astype(np.str_)
        assert result.dtype == object

    def test_astype_numpy_bytes_dtype(self):
        # GH#50127: Index.astype("S3") (numpy bytes dtype) should not raise
        # NotImplementedError. This is the exact regression from the issue:
        # pd.Index(['a', 'b']).astype("S3") worked in pandas 1.5 but raised
        # NotImplementedError in pandas 2.x.
        idx = Index(["a", "b"])
        result = idx.astype("S3")
        # Result dtype is object; values are bytes
        assert result.dtype == object
        assert result[0] == b"a"
        assert result[1] == b"b"
