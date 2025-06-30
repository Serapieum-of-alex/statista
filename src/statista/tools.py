""""Statistical tools"""

from typing import List, Union
import numpy as np


class Tools:
    """Tools.

    Tools different statistical and interpolation tools
    """

    def __init__(self):
        pass

    @staticmethod
    def normalize(x: Union[List[float], np.ndarray]) -> np.ndarray:
        """Normalizer.

        to normalize values between 0 and 1

        Parameters
        ----------
        x: List[float], np.ndarray
            list of values

        Returns
        -------
        normalized numbers: [List]
            list of normalized values
        """
        x = np.array(x)
        data_max = max(x)
        data_min = min(x)
        return (x - data_min) / (data_max - data_min)

    @staticmethod
    def standardize(x: Union[List[float], np.ndarray]) -> np.ndarray:
        """Standardize.

        to standardize (make the average equals 1 and the standard deviation
        equals 0)

        Parameters
        ----------
        x: List[float], np.ndarray
            list of values

        Returns
        -------
        Standardized values: np.ndarray
            list of normalized values
        """
        x = np.array(x)

        mean = np.mean(x)
        std = np.std(x)
        s = (x - mean) / std
        return s

    @staticmethod
    def rescale(old_value, old_min, old_max, new_min, new_max):
        """Rescale.

        Rescale method rescales a value between two boundaries to a new value bewteen two other boundaries.

        Parameters
        ----------
        old_value: [float]
            The old value you want to transform
        old_min: [float]
            min old value
        old_max: [float]
            max old value
        new_min: [float]
            min new value
        new_max: [float]
            max new value

        Returns
        -------
        float:
            transformed new value
        """
        old_range = old_max - old_min
        new_range = new_max - new_min
        new_value = (((old_value - old_min) * new_range) / old_range) + new_min

        return new_value

    @staticmethod
    def log_rescale(x, min_old, max_old, min_new, max_new):
        """Logarithmic Rescale.

        log_rescale transforms the value between two normal values to a logarithmic scale between logarithmic value
        of both boundaries

            np.log(base)(number) = power
            the inverse of logarithmic is base**power = number

        Parameters
        ----------
        x: [float]
            new value needed to be transformed to a logarithmic scale
        min_old: [float]
            min old value in normal scale
        max_old: [float]
            max old value in normal scale
        min_new: [float]
            min new value in normal scale
        max_new: [float]
            max_new max new value

        Returns
        -------
        Y: [int]
            integer number between new max_new and min_new boundaries
        """
        # get the boundaries of the logarithmic scale
        if min_old == 0.0:
            min_old_log = -7
        else:
            min_old_log = np.log(min_old)

        max_old_log = np.log(max_old)

        if x == 0:
            x_log = -7
        else:
            x_log = np.log(x)

        y = int(
            np.round(Tools.rescale(x_log, min_old_log, max_old_log, min_new, max_new))
        )

        return y

    @staticmethod
    def inv_log_rescale(x, min_old, max_old, min_new, max_new, base=np.e):
        """Inverse Logarithmic Rescale.

        inv_log_rescale transforms the value between two normal values to a logarithmic scale between logarithmic
        value of both boundaries.

            np.log(base)(number) = power
            the inverse of logarithmic is base**power = number

        Parameters
        ----------
        base
        x: [float]
            new value needed to be transformed to a logarithmic scale
        min_old: [float]
            min old value in normal scale
        max_old: [float]
            max old value in normal scale
        min_new: [float]
            min new value in normal scale
        max_new: [float]
            max_new max new value

        Returns
        -------
        Y: [int]
            integer number between new max_new and min_new boundaries
        """
        # get the boundaries of the logarithmic scale

        min_old_power = np.power(base, min_old)
        max_old_power = np.power(base, max_old)
        x_power = np.power(base, x)

        y = int(
            np.round(
                Tools.rescale(x_power, min_old_power, max_old_power, min_new, max_new)
            )
        )
        return y

    @staticmethod
    def round(number: float, precision: int) -> float:
        """round

        Parameters
        ----------
        number: float
            number to be rounded.
        precision: int
            precision of the rounding.

        Returns
        -------
        float:
            rounded number
        """
        return round(number / precision) * precision
