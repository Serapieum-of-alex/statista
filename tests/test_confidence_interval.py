"""Confidence Interval Tests"""
from typing import Dict
import numpy as np
from statista.confidence_interval import ConfidenceInterval
from statista.distributions import GEV


def test_boot_strap(
    time_series1: list,
    ci_cdf: np.ndarray,
    ci_param: Dict[str, float],
):
    """
    - The test function for the boot strap method in the ConfidenceInterval class.
    - the test can not compare the generated LB and UB as they are randomly generated.
    """
    ci = ConfidenceInterval.boot_strap(
        time_series1,
        statfunction=GEV.ci_func,
        gevfit=ci_param,
        n_samples=len(time_series1),
        F=ci_cdf,
        method="lmoments",
    )
    lb = ci["lb"]
    ub = ci["ub"]
    assert isinstance(lb, np.ndarray)
    assert isinstance(ub, np.ndarray)
    assert lb.shape == ub.shape == (len(time_series1),)
