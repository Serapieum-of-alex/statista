from typing import List, Dict

import numpy as np
import pandas as pd
from pandas import DataFrame
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
def gev_dist_parameters() -> Dict[str, Dict[str, float]]:
    return {
        "lmoments": {
            "loc": 16.392889171307772,
            "scale": 0.7005442761744839,
            "shape": -0.1614793298009645,
        },
        "mle": {
            "loc": 16.303264414285966,
            "scale": 0.5411914328865949,
            "shape": -0.5013795739666272,
        },
    }


@pytest.fixture(scope="module")
def gev_pdf() -> np.array:
    return np.array(
        [
            0.46686268,
            0.50674728,
            0.13568617,
            0.5171857,
            0.46290923,
            0.4572899,
            0.31771916,
            0.03121843,
            0.40982638,
            0.34582871,
            0.47538097,
            0.48229776,
            0.51992017,
            0.25731877,
            0.07774146,
            0.14318118,
            0.47520795,
            0.52563445,
            0.47327913,
            0.53154392,
            0.3007426,
            0.04651425,
            0.39390943,
            0.50145893,
            0.33531555,
            0.10824839,
            0.09175549,
        ]
    )


@pytest.fixture(scope="module")
def exp_dist_parameters() -> Dict[str, Dict[str, float]]:
    return {
        "mle": {"loc": 144.0, "scale": 446.83333333333337},
        "lmoments": {"loc": 285.74807826694627, "scale": 305.0852550663871},
    }


@pytest.fixture(scope="module")
def gum_dist_parameters() -> Dict[str, Dict[str, float]]:
    return {
        "mle": {"loc": 466.1208189815563, "scale": 214.3001449633138},
        "lmoments": {"loc": 463.8040433832974, "scale": 220.0724922663106},
    }


@pytest.fixture(scope="module")
def gum_pdf() -> np.ndarray:
    return np.array(
        [
            0.0002699,
            0.00062362,
            0.00066007,
            0.00080406,
            0.00107551,
            0.00108773,
            0.00113594,
            0.00118869,
            0.0012884,
            0.00136443,
            0.00141997,
            0.00151536,
            0.00151886,
            0.00153245,
            0.00154542,
            0.00154856,
            0.00160752,
            0.00166602,
            0.00166918,
            0.00166958,
            0.00166028,
            0.00164431,
            0.00163473,
            0.00158442,
            0.00158442,
            0.00158017,
            0.00158017,
            0.00156466,
            0.00155064,
            0.00154824,
            0.00152589,
            0.00151815,
            0.00135704,
            0.00132178,
            0.00128594,
            0.00122319,
            0.00116002,
            0.00116002,
            0.00113677,
            0.00109378,
            0.00097405,
            0.00093331,
            0.00079382,
            0.00079099,
            0.00073328,
            0.00064623,
            0.0006293,
            0.00041714,
            0.00039389,
            0.00023869,
            0.00018416,
            0.00016156,
            0.00016156,
            0.00012409,
        ]
    )


@pytest.fixture(scope="module")
def exp_pdf() -> np.ndarray:
    return np.array(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.00326435,
            0.00317986,
            0.00308743,
            0.00291054,
            0.0027709,
            0.00266403,
            0.00246249,
            0.00245443,
            0.00242246,
            0.00239091,
            0.00238308,
            0.00221728,
            0.00193846,
            0.00190071,
            0.00189449,
            0.00167812,
            0.00159761,
            0.00156137,
            0.00142445,
            0.00142445,
            0.00141514,
            0.00141514,
            0.00138304,
            0.00135611,
            0.00135167,
            0.00131238,
            0.00129953,
            0.00108516,
            0.00104673,
            0.00100966,
            0.0009487,
            0.00089142,
            0.00089142,
            0.0008712,
            0.00083486,
            0.00073951,
            0.00070866,
            0.00060748,
            0.00060549,
            0.00056522,
            0.00050561,
            0.00049414,
            0.0003514,
            0.00033564,
            0.00022724,
            0.00018667,
            0.00016918,
            0.00016918,
            0.00013898,
        ]
    )


@pytest.fixture(scope="module")
def normal_dist_parameters() -> Dict[str, Dict[str, float]]:
    return {
        "mle": {"loc": 590.8333333333334, "scale": 269.6701517423475},
        "lmoments": {"loc": 590.8333333333334, "scale": 270.3747675984547},
    }


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


@pytest.fixture(scope="module")
def ams_gauges() -> DataFrame:
    """AMS gauges"""
    ams = pd.read_csv(f"tests/data/ams-gauges.csv")
    ams.index = ams["date"]
    return ams


@pytest.fixture(scope="module")
def gauges_statistical_properties() -> DataFrame:
    """AMS gauges"""
    return pd.read_csv(f"tests/data/statistical_properties.csv", index_col="id")


@pytest.fixture(scope="module")
def gauges_distribution_properties() -> DataFrame:
    """AMS gauges"""
    return pd.read_csv(f"tests/data/distribution_properties.csv", index_col="id")
