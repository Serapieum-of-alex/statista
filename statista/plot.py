from typing import Union, Tuple, List, Any
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
        Qx: np.ndarray,
        pdf_fitted,
        data_sorted: np.ndarray,
        figsize: tuple = (6, 5),
        xlabel: str = "Actual data",
        ylabel: str = "pdf",
        fontsize: int = 15,
    ):
        """pdf.

        Parameters
        ----------
        Qx
        pdf_fitted
        data_sorted
        figsize
        xlabel
        ylabel
        fontsize

        Returns
        -------
        """
        fig = plt.figure(figsize=figsize)
        # gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig)
        # Plot the histogram and the fitted distribution, save it for each gauge.
        ax = fig.add_subplot()
        ax.plot(Qx, pdf_fitted, "r-")
        ax.hist(data_sorted, density=True)
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        return fig, ax

    @staticmethod
    def cdf(
        Qx,
        cdf_fitted,
        data_sorted,
        cdf_Weibul,
        figsize=(6, 5),
        xlabel="Actual data",
        ylabel="cdf",
        fontsize=15,
    ):
        """cdf.

        Parameters
        ----------
        Qx
        cdf_fitted
        data_sorted
        cdf_Weibul
        figsize
        xlabel
        ylabel
        fontsize

        Returns
        -------
        """
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
        ax.plot(Qx, cdf_fitted, "r-", label="Fitted distribution")
        ax.plot(data_sorted, cdf_Weibul, ".-", label="Weibul plotting position")
        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        plt.legend(fontsize=fontsize, framealpha=1)
        return fig, ax

    @staticmethod
    def details(
        Qx: Union[np.ndarray, list],
        Qth: Union[np.ndarray, list],
        Qact: Union[np.ndarray, list],
        pdf: Union[np.ndarray, list],
        cdf_fitted: Union[np.ndarray, list],
        F: Union[np.ndarray, list],
        Qlower: Union[np.ndarray, list],
        Qupper: Union[np.ndarray, list],
        alpha: float,
        fig1size: tuple = (10, 5),
        fig2size: tuple = (6, 6),
        xlabel: str = "Actual data",
        ylabel: str = "cdf",
        fontsize: int = 15,
    ) -> Tuple[List[Figure], List[Any]]:
        """details.

        Parameters
        ----------
        Qx
        Qth
        Qact
        pdf
        cdf_fitted
        F
        Qlower
        Qupper
        alpha
        fig1size
        fig2size
        xlabel
        ylabel
        fontsize

        Returns
        -------
        """
        fig1 = plt.figure(figsize=fig1size)
        gs = gridspec.GridSpec(nrows=1, ncols=2, figure=fig1)
        # Plot the histogram and the fitted distribution, save it for each gauge.
        ax1 = fig1.add_subplot(gs[0, 0])
        ax1.plot(Qx, pdf, "r-")
        ax1.hist(Qact, density=True)
        ax1.set_xlabel(xlabel, fontsize=fontsize)
        ax1.set_ylabel("pdf", fontsize=fontsize)

        ax2 = fig1.add_subplot(gs[0, 1])
        ax2.plot(Qx, cdf_fitted, "r-")
        Qact.sort()
        ax2.plot(Qact, F, ".-")
        ax2.set_xlabel(xlabel, fontsize=fontsize)
        ax2.set_ylabel(ylabel, fontsize=15)

        fig2 = plt.figure(figsize=fig2size)
        plt.plot(Qth, Qact, "d", color="#606060", markersize=12, label="Actual Data")
        plt.plot(Qth, Qth, "^-.", color="#3D59AB", label="Theoretical Data")

        plt.plot(
            Qth,
            Qlower,
            "*--",
            color="#DC143C",
            markersize=12,
            label="Lower limit (" + str(int((1 - alpha) * 100)) + " % CI)",
        )
        plt.plot(
            Qth,
            Qupper,
            "*--",
            color="#DC143C",
            markersize=12,
            label="Upper limit (" + str(int((1 - alpha) * 100)) + " % CI)",
        )
        plt.legend(fontsize=fontsize, framealpha=1)
        plt.xlabel("Theoretical Values", fontsize=fontsize)
        plt.ylabel("Actual Values", fontsize=fontsize)

        return [fig1, fig2], [ax1, ax2]
