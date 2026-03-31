import numpy as np
import pytest

import pandas as pd
from pandas import (
    NA,
    Series,
)
import pandas._testing as tm


def test_mask():
    # compare with tested results in test_where
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0

    rs = s.where(~cond, np.nan)
    tm.assert_series_equal(rs, s.mask(cond))

    rs = s.where(~cond)
    rs2 = s.mask(cond)
    tm.assert_series_equal(rs, rs2)

    rs = s.where(~cond, -s)
    rs2 = s.mask(cond, -s)
    tm.assert_series_equal(rs, rs2)

    cond = Series([True, False, False, True, False], index=s.index)
    s2 = -(s.abs())
    rs = s2.where(~cond[:3])
    rs2 = s2.mask(cond[:3])
    tm.assert_series_equal(rs, rs2)

    rs = s2.where(~cond[:3], -s2)
    rs2 = s2.mask(cond[:3], -s2)
    tm.assert_series_equal(rs, rs2)

    msg = "Array conditional must be same shape as self"
    with pytest.raises(ValueError, match=msg):
        s.mask(1)
    with pytest.raises(ValueError, match=msg):
        s.mask(cond[:3].values, -s)


def test_mask_casts():
    # dtype changes
    ser = Series([1, 2, 3, 4])
    result = ser.mask(ser > 2, np.nan)
    expected = Series([1, 2, np.nan, np.nan])
    tm.assert_series_equal(result, expected)


def test_mask_casts2():
    # see gh-21891
    ser = Series([1, 2])
    res = ser.mask([True, False])

    exp = Series([np.nan, 2])
    tm.assert_series_equal(res, exp)


def test_mask_inplace():
    s = Series(np.random.default_rng(2).standard_normal(5))
    cond = s > 0

    rs = s.copy()
    rs.mask(cond, inplace=True)
    tm.assert_series_equal(rs.dropna(), s[~cond])
    tm.assert_series_equal(rs, s.mask(cond))

    rs = s.copy()
    rs.mask(cond, -s, inplace=True)
    tm.assert_series_equal(rs, s.mask(cond, -s))


def test_mask_na_condition_does_not_raise():
    # GH#35429: mask/where with a condition containing NA should not raise
    # TypeError/ValueError. NA in the condition is treated as False
    # (i.e. the value is kept for mask, or set to other for where).
    ser = Series([1, 2, 3, 4])

    # Object-dtype condition with pd.NA
    cond_with_pd_na = Series([True, pd.NA, False, True])
    result = ser.mask(cond_with_pd_na)
    # NA in condition treated as False -> position 1 keeps its value
    expected = Series([np.nan, 2.0, 3.0, np.nan])
    tm.assert_series_equal(result, expected)

    # Object-dtype condition with np.nan
    cond_with_np_nan = Series([True, np.nan, False, True])
    result2 = ser.mask(cond_with_np_nan)
    tm.assert_series_equal(result2, expected)

    # Also verify where() doesn't raise
    # where(cond): True->keep, False/NA->replace with NA
    result3 = ser.where(cond_with_pd_na)
    # Position 0: True -> keep (1); 1: NA -> False -> NA; 2: False -> NA; 3: True -> keep (4)
    expected3 = Series([1.0, np.nan, np.nan, 4.0])
    tm.assert_series_equal(result3, expected3)


def test_mask_na_condition_dataframe_convert_dtypes():
    # GH#35429: The exact issue from the report - DataFrame.mask and where
    # with pd.NA in a boolean Series (via convert_dtypes) should not raise
    import pandas as pd

    a = pd.DataFrame([{"col": "a"}] * 3).convert_dtypes()
    tfna = pd.Series([True, False, pd.NA]).convert_dtypes()

    # mask: True->replace, False->keep, NA->keep (NA treated as False)
    result_mask = a.mask(tfna, "b")
    # Position 0: True -> replace with "b"; 1: False -> keep "a"; 2: NA -> keep "a"
    expected_mask = pd.DataFrame({"col": ["b", "a", "a"]}).convert_dtypes()
    tm.assert_frame_equal(result_mask, expected_mask)

    # where: True->keep, False->replace, NA->replace (NA treated as False)
    result_where = a.where(tfna, "b")
    # Position 0: True -> keep "a"; 1: False -> replace with "b"; 2: NA -> replace with "b"
    expected_where = pd.DataFrame({"col": ["a", "b", "b"]}).convert_dtypes()
    tm.assert_frame_equal(result_where, expected_where)
