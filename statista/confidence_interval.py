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
        statfunction,
        alpha: float = 0.05,
        n_samples: int = 100,
        **kargs,
    ):  # ->  Dict[str, OrderedDict[str, Tuple[Any, Any]]]
        """boot_strap

        Calculate confidence intervals using parametric bootstrap and the percentil interval method This is used to
        obtain confidence intervals for the estimators and the return values for several return values.

        More info about bootstrapping can be found on:
            - Efron: "An Introduction to the Bootstrap", Chapman & Hall (1993)
            - https://en.wikipedia.org/wiki/Bootstrapping_%28statistics%29

        parameters:
        -----------
        alpha : [numeric]
                alpha or SignificanceLevel is a value of the confidence interval.
        kwargs :
            gevfit : [list]
                list of the three parameters of the GEV distribution [shape, loc, scale]
            F : [list]
                non exceedence probability/ cdf
            method: [str]
                method used to fit the generated samples from the bootstrap method ["lmoments", "mle", "mm"]. Default is
                "lmoments".
        """
        alphas = np.array([alpha / 2, 1 - alpha / 2])
        tdata = (np.array(data),)

        # We don't need to generate actual samples; that would take more memory.
        # Instead, we can generate just the indexes, and then apply the statfun
        # to those indexes.
        bootindexes = ConfidenceInterval.bs_indexes(tdata[0], n_samples)
        stat = np.array(
            [
                statfunction(*(x[indexes] for x in tdata), **kargs)
                for indexes in bootindexes
            ]
        )
        stat.sort(axis=0)

        # Percentile Interval Method
        avals = alphas
        nvals = np.round((n_samples - 1) * avals).astype("int")

        if np.any(nvals == 0) or np.any(nvals == n_samples - 1):
            logger.debug(
                "Some values used extremal samples; results are probably unstable."
            )
            # warnings.warn(
            #     "Some values used extremal samples; results are probably unstable.",
            #     InstabilityWarning,
            # )
        elif np.any(nvals < 10) or np.any(nvals >= n_samples - 10):
            logger.debug(
                "Some values used top 10 low/high samples; results may be unstable."
            )
            # warnings.warn(
            #     "Some values used top 10 low/high samples; results may be unstable.",
            #     InstabilityWarning,
            # )

        if nvals.ndim == 1:
            # All nvals are the same. Simple broadcasting
            out = stat[nvals]
        else:
            # Nvals are different for each data point. Not simple broadcasting.
            # Each set of nvals along axis 0 corresponds to the data at the same
            # point in other axes.
            out = stat[(nvals, np.indices(nvals.shape)[1:].squeeze())]

        ub = out[0, 3:]
        lb = out[1, 3:]
        params = OrderedDict()
        params["shape"] = (out[0, 0], out[1, 0])
        params["location"] = (out[0, 1], out[1, 1])
        params["scale"] = (out[0, 2], out[1, 3])

        return {"lb": lb, "ub": ub, "params": params}
