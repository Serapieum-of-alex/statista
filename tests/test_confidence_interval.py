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
    CI = ConfidenceInterval.BootStrap(
        time_series1,
        statfunction=GEV.ci_func,
        gevfit=ci_param,
        n_samples=len(time_series1),
        F=ci_cdf,
        method="lmoments",
    )
    LB = CI["LB"]
    UB = CI["UB"]
    assert isinstance(LB, np.ndarray)
    assert isinstance(UB, np.ndarray)
    assert LB.shape == UB.shape == (len(time_series1),)
