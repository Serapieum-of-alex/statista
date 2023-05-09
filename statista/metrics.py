from numbers import Number
from typing import Union

import numpy as np
from sklearn.metrics import r2_score


def RMSE(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]) -> float:
    """Root Mean Squared Error. Metric for the estimation of performance of the hydrological model.

    Parameters
    ----------
    obs: [ndarray]
        Measured values
    sim: [ndarray]
        Simulated values

    Returns
    -------
    error: [float]
        RMSE value
    """
    # convert Qobs & sim into arrays
    obs = np.array(obs)
    sim = np.array(sim)

    rmse = np.sqrt(np.average((np.array(obs) - np.array(sim)) ** 2), dtype=np.float64)

    return rmse


def RMSEHF(
    obs: Union[list, np.ndarray],
    sim: Union[list, np.ndarray],
    WStype: int,
    N: int,
    alpha: Union[int, float],
) -> float:
    """Weighted Root mean square Error for High flow.

    Parameters
    ----------
    obs: [list/array]
        observed flow
    sim: [list/array]
        simulated flow
    WStype:
        Weighting scheme (1,2,3,4)
    N:
        power
    alpha:
        Upper limit for low flow weight

    Returns
    -------
    error values
    """
    # input data validation
    # data type
    assert isinstance(
        WStype, int
    ), f"Weighting scheme should be an integer number between 1 and 4 and you entered {WStype}"
    assert isinstance(alpha, int) or isinstance(
        alpha, float
    ), "alpha should be a number and between 0 & 1"
    assert isinstance(N, Number), "N should be a number and between 0 & 1"
    # Input values
    assert (
        1 <= WStype <= 4
    ), f"Weighting scheme should be an integer number between 1 and 4 you have enters {WStype}"
    assert (
        N >= 0
    ), f"Weighting scheme Power should be positive number you have entered {N}"
    assert (
        0 < alpha < 1
    ), f"alpha should be float number and between 0 & 1 you have entered {alpha}"

    # convert obs & sim into arrays
    obs = np.array(obs)
    sim = np.array(sim)

    Qmax = max(obs)
    h = obs / Qmax  # rational Discharge

    if WStype == 1:
        w = h**N  # rational Discharge power N
    elif (
        WStype == 2
    ):  # -------------------------------------------------------------N is not in the equation
        w = (h / alpha) ** N
        w[h > alpha] = 1
    elif WStype == 3:
        w = np.zeros(np.size(h))  # zero for h < alpha and 1 for h > alpha
        w[h > alpha] = 1
    elif WStype == 4:
        w = np.zeros(np.size(h))  # zero for h < alpha and 1 for h > alpha
        w[h > alpha] = 1
    else:  # sigmoid function
        w = 1 / (1 + np.exp(-10 * h + 5))

    a = (obs - sim) ** 2
    b = a * w
    c = sum(b)
    error = np.sqrt(c / len(obs))

    return error


def RMSELF(
    obs: Union[list, np.ndarray],
    Qsim: Union[list, np.ndarray],
    WStype: int,
    N: int,
    alpha: Union[int, float],
) -> float:
    """Weighted Root mean square Error for low flow.

    inputs:
    ----------
    obs: [list/array]
        observed flow
    sim: [list/array]
        simulated flow
    WStype:
        Weighting scheme (1,2,3,4)
    N:
        power
    alpha:
        Upper limit for low flow weight

    Returns
    -------
    error values
    """
    if not isinstance(WStype, int):
        raise TypeError(
            f"Weighting scheme should be an integer number between 1 and 4 and you entered {WStype}"
        )

    if not (isinstance(alpha, int) or isinstance(alpha, float)):
        raise TypeError("alpha should be a number and between 0 & 1")

    if not isinstance(N, Number):
        raise TypeError("N should be a number and between 0 & 1")

    # Input values
    if not 1 <= WStype <= 4:
        raise ValueError(
            f"Weighting scheme should be an integer number between 1 and 4 you have enters {WStype}"
        )

    if not N >= 0:
        raise ValueError(
            f"Weighting scheme Power should be positive number you have entered {N}"
        )

    if not 0 < alpha < 1:
        raise ValueError(
            f"alpha should be float number and between 0 & 1 you have entered {alpha}"
        )

    # convert obs & sim into arrays
    obs = np.array(obs)
    Qsim = np.array(Qsim)

    Qmax = max(obs)  # rational Discharge power N
    qr = (Qmax - obs) / Qmax

    if WStype == 1:
        w = qr**N
    elif WStype == 2:  # ------------------------------- N is not in the equation
        #        w=1-qr*((0.50 - alpha)**N)
        w = ((1 / (alpha**2)) * (1 - qr) ** 2) - ((2 / alpha) * (1 - qr)) + 1
        w[1 - qr > alpha] = 0
    elif WStype == 3:  # the same like WStype 2
        #        w=1-qr*((0.50 - alpha)**N)
        w = ((1 / (alpha**2)) * (1 - qr) ** 2) - ((2 / alpha) * (1 - qr)) + 1
        w[1 - qr > alpha] = 0
    elif WStype == 4:
        #        w = 1-qr*(0.50 - alpha)
        w = 1 - ((1 - qr) / alpha)
        w[1 - qr > alpha] = 0
    else:  # sigmoid function
        #        w=1/(1+np.exp(10*h-5))
        w = 1 / (1 + np.exp(-10 * qr + 5))

    a = (obs - Qsim) ** 2
    b = a * w
    c = sum(b)
    error = np.sqrt(c / len(obs))

    return error


def KGE(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]):
    """(Gupta et al. 2009) have showed the limitation of using a single error function to measure the efficiency of calculated flow and showed that Nash-Sutcliff efficiency (NSE) or RMSE can be decomposed into three component correlation, variability and bias.

    Parameters
    ----------
    obs: [list/array]
        observed flow
    sim: [list/array]
        simulated flow

    Returns
    -------
    error values
    """
    # convert obs & sim into arrays
    obs = np.array(obs)
    sim = np.array(sim)

    c = np.corrcoef(obs, sim)[0][1]
    alpha = np.std(sim) / np.std(obs)
    beta = np.mean(sim) / np.mean(obs)

    kge = 1 - np.sqrt(((c - 1) ** 2) + ((alpha - 1) ** 2) + ((beta - 1) ** 2))

    return kge


def WB(obs, Qsim):
    """The mean cumulative error measures how much the model succeed to reproduce the stream flow volume correctly. This error allows error compensation from time step to another and it is not an indication on how accurate is the model in the simulated flow. the naive model of Nash-Sutcliffe (simulated flow is as accurate as average observed flow) will result in WB error equals to 100 %. (Oudin et al. 2006)

     inputs:
     ----------
    obs: [list/array]
         observed flow
     sim: [list/array]
         simulated flow

     Returns
     -------
     error values
    """
    Qobssum = np.sum(obs)
    Qsimsum = np.sum(Qsim)
    wb = 100 * (1 - np.abs(1 - (Qsimsum / Qobssum)))

    return wb


def NSE(obs: np.ndarray, sim: np.ndarray):
    """Nash-Sutcliffe efficiency. Metric for the estimation of performance of the hydrological model.

    Parameters
    ----------
    obs: [list/array]
        observed flow
    sim: [list/array]
        simulated flow

    Returns
    -------
    float:
        NSE value
    """
    # convert obs & sim into arrays
    obs = np.array(obs)
    sim = np.array(sim)

    a = sum((obs - sim) ** 2)
    b = sum((obs - np.average(obs)) ** 2)
    e = 1 - (a / b)

    return e


def NSEHF(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]):
    """NSEHF.

    Modified Nash-Sutcliffe efficiency. Metric for the estimation of performance of the
    hydrological model

    reference:
    Hundecha Y. & Bárdossy A. Modeling of the effect of land use
    changes on the runoff generation of a river basin through
    parameter regionalization of a watershed model. J Hydrol
    2004, 292, (1–4), 281–295

    Parameters
    ----------
    obs: [list/array]
        observed flow
    sim: [list/array]
        simulated flow

    Returns
    -------
    float:
        NSE value
    """
    # convert obs & sim into arrays
    obs = np.array(obs)
    sim = np.array(sim)

    a = sum(obs * (obs - sim) ** 2)
    b = sum(obs * (obs - np.average(obs)) ** 2)
    e = 1 - (a / b)

    return e


def NSELF(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]):
    """NSELF.

    Modified Nash-Sutcliffe efficiency. Metric for the estimation of performance of the
    hydrological model

    reference:
    Hundecha Y. & Bárdossy A. Modeling of the effect of land use
    changes on the runoff generation of a river basin through
    parameter regionalization of a watershed model. J Hydrol
    2004, 292, (1–4), 281–295

    Parameters
    ----------
    obs: [list/array]
        observed flow
    sim: [list/array]
        simulated flow

    Returns
    -------
    float:
        NSELF value
    """
    # convert obs & sim into arrays
    obs = np.array(np.log(obs))
    sim = np.array(np.log(sim))

    a = sum(obs * (obs - sim) ** 2)
    b = sum(obs * (obs - np.average(obs)) ** 2)
    e = 1 - (a / b)

    return e


def MBE(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]):
    """MBE (mean bias error)

    MBE = (sim - obs)/n

    Parameters
    ----------
    obs: [list/array]
        observed flow
    sim: [list/array]
        simulated flow

    Returns
    -------
    float:
        mean bias error.
    """

    return (np.array(sim) - np.array(obs)).mean()


def MAE(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]):
    """MAE (mean absolute error)

    MAE = |(obs - sim)|/n

    Parameters
    ----------
    obs: [list/array]
        observed flow
    sim: [list/array]
        simulated flow

    Returns
    -------
    float:
        mean absolute error.
    """

    return np.abs(np.array(obs) - np.array(sim)).mean()


def PearsonCorre(x: Union[list, np.ndarray], y: Union[list, np.ndarray]) -> Number:
    """Pearson correlation coefficient.

        - Pearson correlation coefficient is independent of the magnitude of the numbers.
        - it is sensitive to relative changes only.

    .. math:: R_{ij} = \\frac{ C_{ij} } { \\sqrt{ C_{ii} C_{jj} } }

        - covariance / std1 * std2

    The values of `R` are between -1 and 1, inclusive.

    Parameters
    ----------
    x : array_like
        A 1-D array containing a variable.
    y : array_like,
        A 1-D array containing a variable.

    Returns
    -------
    R : ndarray
        The correlation coefficient of the variables.
    """
    return np.corrcoef(np.array(x), np.array(y))[0][1]


def R2(obs: Union[list, np.ndarray], sim: Union[list, np.ndarray]):
    """R2.

        the coefficient of determination measures how well the predicted
        values match (and not just follow) the observed values.
        It depends on the distance between the points and the 1:1 line
        (and not the best-fit line)
        Closer the data to the 1:1 line, higher the coefficient of determination.
        The coefficient of determination is often denoted by R². However,
        it is not the square of anything. It can range from any negative number to +1
        - R² = +1 indicates that the predictions match the observations perfectly
        - R² = 0 indicates that the predictions are as good as random guesses around
            the mean of the observed values
        - Negative R² indicates that the predictions are worse than random

    Since R² indicates the distance of points from the 1:1 line, it does depend
    on the magnitude of the numbers (unlike r² peason correlation coefficient).

    Parameters
    ----------
    obs: [list/array]
        observed flow
    sim: [list/array]
        simulated flow
    """
    return r2_score(np.array(obs), np.array(sim))
