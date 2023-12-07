""" Tests for the eva module. """
import numpy as np
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
        significance_level=0.05,
    )
    statistical_properties.drop(columns=["nyr"], inplace=True)
    gauges_statistical_properties.drop(columns=["nyr"], inplace=True)
    assert (
        np.isclose(
            statistical_properties.values,
            gauges_statistical_properties.values,
            atol=0.01,
        )
    ).all()
    assert (
        np.isclose(
            distribution_properties.values,
            gauges_distribution_properties.values,
            atol=0.01,
        )
    ).all()
    # try:
    shutil.rmtree(path)
