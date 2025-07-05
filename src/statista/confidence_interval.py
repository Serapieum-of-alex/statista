"""Confidence interval module."""

from collections import OrderedDict
from loguru import logger
from typing import Union
from numpy.random import randint
import numpy as np


class ConfidenceInterval:
    """ConfidenceInterval."""

    def __init__(self):
        pass

    @staticmethod
    def bs_indexes(data, n_samples=10000) -> np.ndarray:
        """bs_indexes.

            - generate random indeces to shuffle the data of the given array.
            - using the indeces, you can access the given data array and obtain randomly generated data from the
            original given data.
            - Given data points data, where axis 0 is considered to delineate points, return a generator for
            sets of bootstrap indexes.

        This can be used as a list of bootstrap indexes (with
        list(bootstrap_indexes(data))) as well.

        Returns
        -------
        np.ndarray
            array with the same length as the input data, containing integer indeces.

        Examples
        --------
        >>> data = [3.1, 2.4, 5.6, 8.4]
        >>> indeces = ConfidenceInterval.bs_indexes(data, n_samples=2)
        >>> print(indeces)
        >>> [1, 4, 4, 3]
        >>> print(indeces)
        >>> [2, 3, 1, 2]
        """
        for _ in range(n_samples):
            yield randint(data.shape[0], size=(data.shape[0],))

    @staticmethod
    def boot_strap(
        data: Union[list, np.ndarray],
        state_function: callable,
        alpha: float = 0.05,
        n_samples: int = 100,
        **kwargs,
    ):  # ->  Dict[str, OrderedDict[str, Tuple[Any, Any]]]
        """boot_strap

        Calculate confidence intervals using parametric bootstrap and the percentile interval method This is used to
        obtain confidence intervals for the estimators and the return values for several return values.

        More info about bootstrapping can be found on:
            - Efron: "An Introduction to the Bootstrap", Chapman & Hall (1993)
            - https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29

        Parameters
        ----------
        data: [list, np.ndarray]
            data to be used to calculate the confidence interval
        state_function: [callable]
            function to be used to calculate the confidence interval
        n_samples: int, Default is 100.
            number of samples to be generated. .
        alpha: numeric, optional, default is 0.05
                alpha or SignificanceLevel is a value of the confidence interval.
        kwargs:
            gevfit: [list]
                list of the three parameters of the GEV distribution [shape, loc, scale]
            F: [list]
                non-exceedance probability/ cdf
            method: [str]
                method used to fit the generated samples from the bootstrap method ["lmoments", "mle", "mm"]. Default is
                "lmoments".
        """
        alphas = np.array([alpha / 2, 1 - alpha / 2])
        tdata = (np.array(data),)

        # We don't need to generate actual samples; that would take more memory.
        # Instead, we can generate just the indexes, and then apply the stat-fun
        # to those indexes.
        boot_indexes = ConfidenceInterval.bs_indexes(tdata[0], n_samples)
        stat = np.array(
            [
                state_function(*(x[indexes] for x in tdata), **kwargs)
                for indexes in boot_indexes
            ]
        )
        stat.sort(axis=0)

        # Percentile Interval Method
        a_vals = alphas
        n_vals = np.round((n_samples - 1) * a_vals).astype("int")

        if np.any(n_vals == 0) or np.any(n_vals == n_samples - 1):
            logger.debug(
                "Some values used extremal samples; results are probably unstable."
            )
            # warnings.warn(
            #     "Some values used extremal samples; results are probably unstable.",
            #     InstabilityWarning,
            # )
        elif np.any(n_vals < 10) or np.any(n_vals >= n_samples - 10):
            logger.debug(
                "Some values used top 10 low/high samples; results may be unstable."
            )
            # warnings.warn(
            #     "Some values used top 10 low/high samples; results may be unstable.",
            #     InstabilityWarning,
            # )

        if n_vals.ndim == 1:
            # All n_vals are the same. Simple broadcasting
            out = stat[n_vals]
        else:
            # n_vals are different for each data point. Not simple broadcasting.
            # Each set of n_vals along axis 0 corresponds to the data at the same
            # point in other axes.
            out = stat[(n_vals, np.indices(n_vals.shape)[1:].squeeze())]

        ub = out[0, 3:]
        lb = out[1, 3:]
        params = OrderedDict()
        params["shape"] = (out[0, 0], out[1, 0])
        params["location"] = (out[0, 1], out[1, 1])
        params["scale"] = (out[0, 2], out[1, 3])

        return {"lb": lb, "ub": ub, "params": params}
