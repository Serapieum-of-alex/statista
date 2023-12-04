from typing import List, Dict

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="module")
def time_series1() -> list:
    return pd.read_csv("examples/data/time_series1.txt", header=None)[0].tolist()


@pytest.fixture(scope="module")
def time_series2() -> list:
    return pd.read_csv("examples/data/time_series2.txt", header=None)[0].tolist()


@pytest.fixture(scope="module")
def dist_estimation_parameters() -> List[str]:
    return ["mle", "lmoments"]


@pytest.fixture(scope="module")
def dist_estimation_parameters_ks() -> str:
    return "lmoments"


@pytest.fixture(scope="module")
def confidence_interval_alpha() -> float:
    return 0.1


@pytest.fixture(scope="module")
def parameter_estimation_optimization_threshold() -> int:
    return 800  # 17


@pytest.fixture(scope="module")
def ci_cdf() -> np.ndarray:
    return np.array(
        [
            0.03571429,
            0.07142857,
            0.10714286,
            0.14285714,
            0.17857143,
            0.21428571,
            0.25,
            0.28571429,
            0.32142857,
            0.35714286,
            0.39285714,
            0.42857143,
            0.46428571,
            0.5,
            0.53571429,
            0.57142857,
            0.60714286,
            0.64285714,
            0.67857143,
            0.71428571,
            0.75,
            0.78571429,
            0.82142857,
            0.85714286,
            0.89285714,
            0.92857143,
            0.96428571,
        ]
    )


@pytest.fixture(scope="module")
def ci_param() -> Dict[str, float]:
    return {"loc": 464.825, "scale": 222.120, "shape": 0.01012}
