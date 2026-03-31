import collections

import numpy as np
import pytest

from pandas import Series
import pandas._testing as tm


class TestSeriesToDict:
    @pytest.mark.parametrize(
        "mapping", (dict, collections.defaultdict(list), collections.OrderedDict)
    )
    def test_to_dict(self, mapping, datetime_series):
        # GH#16122
        result = Series(datetime_series.to_dict(into=mapping), name="ts")
        expected = datetime_series.copy()
        expected.index = expected.index._with_freq(None)
        tm.assert_series_equal(result, expected)

        from_method = Series(datetime_series.to_dict(into=collections.Counter))
        from_constructor = Series(collections.Counter(datetime_series.items()))
        tm.assert_series_equal(from_method, from_constructor)

    @pytest.mark.parametrize(
        "input",
        (
            {"a": np.int64(64), "b": 10},
            {"a": np.int64(64), "b": 10, "c": "ABC"},
            {"a": np.uint64(64), "b": 10, "c": "ABC"},
        ),
    )
    def test_to_dict_return_types(self, input):
        # GH25969

        d = Series(input).to_dict()
        assert isinstance(d["a"], int)
        assert isinstance(d["b"], int)

    def test_to_dict_warns_on_duplicate_index(self):
        # GH#25408 - to_dict() with duplicate index causes data loss,
        # should warn the user
        s = Series([1, 2, 3], index=[0, 0, 1])
        with tm.assert_produces_warning(UserWarning, match="index is not unique"):
            result = s.to_dict()
        # Data loss: only last value for each duplicate index kept
        assert result == {0: 2, 1: 3}

    def test_to_dict_no_warn_on_unique_index(self):
        # GH#25408 - no warning when index is unique
        s = Series([1, 2, 3], index=[0, 1, 2])
        with tm.assert_produces_warning(None):
            result = s.to_dict()
        assert result == {0: 1, 1: 2, 2: 3}
