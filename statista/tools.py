"""Created on Thu May 17 04:26:42 2018.

@author: Mostafa
"""

import numpy as np


class Tools:
    """Tools.

    Tools different statistical and interpolation tools
    """

    def __init__(self):
        pass

    @staticmethod
    def normalize(x):
        """Normalizer.

        to normalize values between 0 and 1

        Parameters
        ----------
        x : [List]
            list of values

        Returns
        -------
        normalized numbers : [List]
            list of normalized values
        """
        x = np.array(x)
        DataMax = max(x)
        DataMin = min(x)
        N = (x - DataMin) / (DataMax - DataMin)
        # [i - DataMin / (DataMax - DataMin) for i in x]
        return N

    @staticmethod
    def standardize(x):
        """Standardize.

        to standardize (make the average equals 1 and the standard deviation
        equals 0)

        Parameters
        ----------
        x: [List]
            list of values

        Returns
        -------
        Standardized values: [List]
            list of normalized values
        """
        x = np.array(x)

        mean = np.mean(x)
        std = np.std(x)
        s = (x - mean) / std
        # [i - mean / (std) for i in x]
        return s

    @staticmethod
    def rescale(OldValue, OldMin, OldMax, NewMin, NewMax):
        """Rescale.

        Rescale nethod rescales a value between two boundaries to a new value
        bewteen two other boundaries

        Parameters
        ----------
        OldValue: [float]
            value need to transformed
        OldMin: [float]
            min old value
        OldMax: [float]
            max old value
        NewMin: [float]
            min new value
        NewMax: [float]
            max new value

        Returns
        -------
        NewValue: [float]
            transformed new value
        """
        OldRange = OldMax - OldMin
        NewRange = NewMax - NewMin
        NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin

        return NewValue

    @staticmethod
    def logarithmicRescale(x, min_old, max_old, min_new, max_new):
        """LogarithmicRescale.

        this function transform the value between two normal values to a logarithmic scale
        between logarithmic value of both boundaries
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
    def invLogarithmicRescale(x, min_old, max_old, min_new, max_new, base=np.e):
        """LogarithmicRescale.

        this function transform the value between two normal values to a logarithmic scale
        between logarithmic value of both boundaries
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
    def round(number, roundto):
        return round(number / roundto) * roundto
