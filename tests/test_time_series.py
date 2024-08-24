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


class TestBoxPlot:

    @pytest.mark.parametrize("ts", ["ts_1d", "ts_2d"])
    def test_plot_box(self, ts: TimeSeries, request):
        """Test the plot_box method."""
        ts = request.getfixturevalue(ts)
        fig, ax = ts.plot_box()
        assert isinstance(
            fig, plt.Figure
        ), "plot_box should return a matplotlib Figure."
        assert isinstance(ax, plt.Axes), "plot_box should return a matplotlib Axes."

        fig, ax = plt.subplots()
        fig2, ax2 = ts.plot_box(fig=fig, ax=ax)
        assert fig2 is fig, "If fig is provided, plot_box should use it."
        assert ax2 is ax, "If ax is provided, plot_box should use it."
        if ts.shape[1] > 1:
            assert len(ax.get_xticklabels()) == 3
