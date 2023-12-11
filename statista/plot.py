"""Plotting functions for statista package."""
from typing import Union, Tuple, List, Any
from numbers import Number
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.figure import Figure
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
        figsize: Tuple[float, float] = (6, 5),
        xlabel: str = "Actual data",
        ylabel: str = "pdf",
        fontsize: int = 11,
    ) -> Tuple[Figure, Any]:
        """pdf.

        Parameters
        ----------
        qx
        pdf_fitted
        data_sorted
        figsize
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
        fig = plt.figure(figsize=figsize)
        # gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)
        # Plot the histogram and the fitted distribution, save it for each gauge.
        ax = fig.add_subplot()
        ax.plot(qx, pdf_fitted, "-", color="#27408B", linewidth=2)
        ax.hist(
            data_sorted, density=True, histtype="stepfilled", color="#DC143C"
        )  # , alpha=0.2
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        return fig, ax

    @staticmethod
    def cdf(
        qx,
        cdf_fitted,
        data_sorted,
        cdf_weibul,
        figsize=(6, 5),
        xlabel="Actual data",
        ylabel="cdf",
        fontsize=11,
    ) -> Tuple[Figure, Any]:
        """cdf.

        Parameters
        ----------
        qx
        cdf_fitted
        data_sorted
        cdf_weibul
        figsize
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
        fig = plt.figure(figsize=figsize)
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
        return fig, ax

    @staticmethod
    def details(
        qx: Union[np.ndarray, list],
        qth: Union[np.ndarray, list],
        q_act: Union[np.ndarray, list],
        pdf: Union[np.ndarray, list],
        cdf_fitted: Union[np.ndarray, list],
        cdf: Union[np.ndarray, list],
        q_lower: Union[np.ndarray, list],
        q_upper: Union[np.ndarray, list],
        alpha: Number,
        fig1_size: Tuple[float, float] = (10, 5),
        fig2_size: Tuple[float, float] = (6, 6),
        xlabel: str = "Actual data",
        ylabel: str = "cdf",
        fontsize: int = 11,
    ) -> Tuple[List[Figure], List[Any]]:
        """details.

        Parameters
        ----------
        qx
        qth
        q_act
        pdf
        cdf_fitted
        cdf
        q_lower
        q_upper
        alpha
        fig1_size
        fig2_size
        xlabel
        ylabel
        fontsize

        Returns
        -------
        """
        fig1 = plt.figure(figsize=fig1_size)
        gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig1)
        # Plot the histogram and the fitted distribution, save it for each gauge.
        ax1 = fig1.add_subplot(gs[0, 0])
        ax1.plot(qx, pdf, "-", color="#27408B", linewidth=2)
        ax1.hist(q_act, density=True, histtype="stepfilled", color="#DC143C")
        ax1.set_xlabel(xlabel, fontsize=fontsize)
        ax1.set_ylabel("pdf", fontsize=fontsize)

        ax2 = fig1.add_subplot(gs[0, 1])
        ax2.plot(qx, cdf_fitted, "-", color="#27408B", linewidth=2)

        q_act.sort()
        ax2.scatter(q_act, cdf, color="#DC143C", facecolors="none")
        ax2.set_xlabel(xlabel, fontsize=fontsize)
        ax2.set_ylabel(ylabel, fontsize=15)

        fig2 = plt.figure(figsize=fig2_size)
        plt.plot(qth, qth, "-.", color="#3D59AB", linewidth=2, label="Theoretical Data")
        # confidence interval
        plt.plot(
            qth,
            q_lower,
            "*--",
            color="grey",
            markersize=10,
            label=f"Lower limit ({int((1 - alpha) * 100)} % CI)",
        )
        plt.plot(
            qth,
            q_upper,
            "*--",
            color="grey",
            markersize=10,
            label=f"Upper limit ({int((1 - alpha) * 100)} % CI)",
        )
        plt.scatter(
            qth, q_act, color="#DC143C", facecolors="none", label="Actual Data"
        )  # "d", markersize=12,
        plt.legend(fontsize=fontsize, framealpha=1)
        plt.xlabel("Theoretical Values", fontsize=fontsize)
        plt.ylabel("Actual Values", fontsize=fontsize)

        return [fig1, fig2], [ax1, ax2]
