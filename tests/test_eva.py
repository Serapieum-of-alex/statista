""" Tests for the eva module. """

import matplotlib
import pandas as pd

matplotlib.use("Agg")
from pandas import DataFrame
import shutil
from pathlib import Path
import pytest
from statista.eva import ams_analysis


@pytest.mark.slow
def test_eva(
    ams_gauges: DataFrame,
    gauges_statistical_properties: DataFrame,
    gauges_distribution_properties: DataFrame,
):
    path = Path("tests/data/gauges/figures")
    if path.exists():
        try:
            path.rmdir()
        except PermissionError:
            print("PermissionError: files were not deleted")

    method = "lmoments"
    save_to = "tests/data/gauges"
    statistical_properties, distribution_properties = ams_analysis(
        time_series_df=ams_gauges,
        ams=True,
        save_plots=True,
        save_to=save_to,
        filter_out=-9,
        method=method,
        alpha=0.05,
    )
    statistical_properties.drop(columns=["nyr"], inplace=True)
    gauges_statistical_properties.drop(columns=["nyr"], inplace=True)
    pd.testing.assert_frame_equal(
        statistical_properties, gauges_statistical_properties, rtol=1e-4
    )
    pd.testing.assert_frame_equal(
        distribution_properties, gauges_distribution_properties, rtol=1e-4
    )
    # try:
    shutil.rmtree(path)
