"""Plotting functions for statista package."""

from typing import Union, Tuple
from numbers import Number
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import numpy as np


class Plot:
    """plot."""

    def __init__(self):
        pass

    @staticmethod
    def pdf(
        qx: np.ndarray,
        pdf_fitted,
        data_sorted: np.ndarray,
        fig_size: Tuple[float, float] = (6, 5),
        xlabel: str = "Actual data",
        ylabel: str = "pdf",
        fontsize: int = 11,
    ) -> Tuple[Figure, Axes]:
        """pdf.

        Parameters
        ----------
        qx
        pdf_fitted
        data_sorted
        fig_size
        xlabel
        ylabel
        fontsize

        Returns
        -------
        Figure:
            matplotlib figure object
        Axes:
            matplotlib plot axis
        """
        fig = plt.figure(figsize=fig_size)
        # gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)
        # Plot the histogram and the fitted distribution, save it for each gauge.
        ax = fig.add_subplot()
        ax.plot(qx, pdf_fitted, "-", color="#27408B", linewidth=2)
        ax.hist(
            data_sorted, density=True, histtype="stepfilled", color="#DC143C"
        )  # , alpha=0.2
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        plt.show()
        return fig, ax

    @staticmethod
    def cdf(
        qx,
        cdf_fitted,
        data_sorted,
        cdf_weibul,
        fig_size=(6, 5),
        xlabel="Actual data",
        ylabel="cdf",
        fontsize=11,
    ) -> Tuple[Figure, Axes]:
        """cdf.

        Parameters
        ----------
        qx
        cdf_fitted
        data_sorted
        cdf_weibul
        fig_size
        xlabel
        ylabel
        fontsize

        Returns
        -------
        Figure:
            matplotlib figure object
        Axis:
            matplotlib plot axis
        """
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot()
        ax.plot(
            qx, cdf_fitted, "-", label="Estimated CDF", color="#27408B", linewidth=2
        )
        ax.scatter(
            data_sorted,
            cdf_weibul,
            label="Empirical CDF",
            color="orangered",
            facecolors="none",
        )
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        plt.legend(fontsize=fontsize, framealpha=1)
        plt.show()
        return fig, ax

    @staticmethod
    def details(
        qx: Union[np.ndarray, list],
        q_act: Union[np.ndarray, list],
        pdf: Union[np.ndarray, list],
        cdf_fitted: Union[np.ndarray, list],
        cdf: Union[np.ndarray, list],
        fig_size: Tuple[float, float] = (10, 5),
        xlabel: str = "Actual data",
        ylabel: str = "cdf",
        fontsize: int = 11,
    ) -> Tuple[Figure, Tuple[Axes, Axes]]:
        """details.

        Parameters
        ----------
        qx: [np.ndarray, list]
            10,000 values generated between the minimum and maximum values of the actual data.
        q_act: [np.ndarray, list]
            Actual data.
        pdf: [np.ndarray, list]
            Probability density function.
        cdf_fitted: [np.ndarray, list]
            Cumulative distribution function of the fitted distribution.
        cdf
        fig_size:  Tuple[float, float], optional, default=(10, 5)
            Size of the first figure.
        xlabel: str, optional, default="Actual data"
            Label for x-axis.
        ylabel: str, optional, default="cdf"
            Label for y-axis.
        fontsize: int, optional, default=11
            Font size.

        Returns
        -------
        Figure:
            matplotlib figure object
        Tuple[Axes, Axes]:
            matplotlib plot axes
        """
        fig = plt.figure(figsize=fig_size)
        gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)
        # Plot the histogram and the fitted distribution, save it for each gauge.
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(qx, pdf, "-", color="#27408B", linewidth=2)
        ax1.hist(q_act, density=True, histtype="stepfilled", color="#DC143C")
        ax1.set_xlabel(xlabel, fontsize=fontsize)
        ax1.set_ylabel("pdf", fontsize=fontsize)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(qx, cdf_fitted, "-", color="#27408B", linewidth=2)

        q_act.sort()
        ax2.scatter(q_act, cdf, color="#DC143C", facecolors="none")
        ax2.set_xlabel(xlabel, fontsize=fontsize)
        ax2.set_ylabel(ylabel, fontsize=15)
        plt.show()
        return fig, (ax1, ax2)

    @staticmethod
    def confidence_level(
        qth: Union[np.ndarray, list],
        q_act: Union[np.ndarray, list],
        q_lower: Union[np.ndarray, list],
        q_upper: Union[np.ndarray, list],
        fig_size: Tuple[float, float] = (6, 6),
        fontsize: int = 11,
        alpha: Number = None,
        marker_size: int = 10,
    ) -> Tuple[Figure, Axes]:
        """details.

        Parameters
        ----------
        qth: [np.ndarray, list]
            Theoretical quantiles (obtained using the inverse_cdf method).
        q_act: [np.ndarray, list]
            Actual data, unsorted.
        q_lower: [np.ndarray, list]
            Lower limit of the confidence interval.
        q_upper: [np.ndarray, list]
            Upper limit of the confidence interval.
        alpha: [float]
            Significance level.
        fig_size: Tuple[float, float], optional, default=(6, 6)
            Size of the second figure.
        fontsize: int, optional, default=11
            Font size.
        marker_size: int, default is 10.
            Size of the markers for the upper and lower bounds.

        Returns
        -------
        Figure:
            matplotlib figure object
        Axes:
            matplotlib plot axes
        """
        q_act.sort()

        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot()
        ax.plot(qth, qth, "-.", color="#3D59AB", linewidth=2, label="Theoretical Data")
        # confidence interval
        ax.plot(
            qth,
            q_lower,
            "*--",
            color="grey",
            markersize=marker_size,
            label=f"Lower limit ({int((1 - alpha) * 100)} % CI)",
        )
        ax.plot(
            qth,
            q_upper,
            "*--",
            color="grey",
            markersize=marker_size,
            label=f"Upper limit ({int((1 - alpha) * 100)} % CI)",
        )
        ax.scatter(
            qth,
            q_act,
            color="#DC143C",
            facecolors="none",
            label="Actual Data",
            zorder=10,
        )  # "d", markersize=12,
        ax.legend(fontsize=fontsize, framealpha=1)
        ax.set_xlabel("Theoretical Values", fontsize=fontsize)
        ax.set_ylabel("Actual Values", fontsize=fontsize)
        plt.show()
        return fig, ax
