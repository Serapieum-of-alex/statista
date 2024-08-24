import pytest
import numpy as np
import matplotlib.pyplot as plt
from statista.time_series import TimeSeries


@pytest.fixture
def sample_data_1d():
    return np.random.randn(100)


@pytest.fixture
def sample_data_2d():
    return np.random.randn(100, 3)


@pytest.fixture
def ts_1d(sample_data_1d) -> TimeSeries:
    return TimeSeries(sample_data_1d)


@pytest.fixture
def ts_2d(sample_data_2d) -> TimeSeries:
    return TimeSeries(sample_data_2d, columns=["A", "B", "C"])


@pytest.mark.parametrize("ts", ["ts_1d", "ts_2d"])
def test_stats(ts: TimeSeries, request):
    """Test the stats method."""
    ts = request.getfixturevalue(ts)
    stats = ts.stats
    assert stats.index.to_list() == [
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]
