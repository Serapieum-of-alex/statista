"""Extreme value analysis."""
from typing import Union, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame

from statista.distributions import GEV, Gumbel, PlottingPosition


def ams_analysis(
    time_series_df: DataFrame,
    ams: bool = False,
    ams_start: str = "A-OCT",
    save_plots: bool = False,
    save_to: str = None,
    filter_out: Union[float, int] = None,
    distribution: str = "GEV",
    method: str = "lmoments",
    estimate_parameters: bool = False,
    quartile: float = 0,
    significance_level: float = 0.1,
) -> Tuple[DataFrame, DataFrame]:
    """StatisticalProperties.

    ams analysis method reads resamples all the the time series in the given dataframe to annual maximum, then fits
    the time series to a given distribution and parameter estimation method.

    Parameters
    ----------
    time_series_df : [DataFrame]
        DataFrame containing multiple time series to do the statistical analysis on.
    ams: [bool]
        True if the the given time series is annual mean series. Default is False.
    ams_start: [str]
        The beginning of the year which is used to resample the time series to get the annual maximum series.
        Default is"A-OCT".
    save_plots : [Bool]
        True if you want to save the plots.
    save_to : [str]
        The rdir where you want to  save the statistical properties.
    filter_out: [Bool]
        For observed or hydraulic model data it has gaps of times where the
        model did not run or gaps in the observed data if these gap days
        are filled with a specific value and you want to ignore it here
        give filter_out = Value you want
    distribution: [str]
        Default is "GEV".
    method: [str]
        available methods are 'mle', 'mm', 'lmoments', optimization. Default is "lmoments"
    estimate_parameters: [bool]
        Default is False.
    quartile: [float]
        the quartile is only used when estinating the distribution parameters based on optimization and a threshould
        value, the threshould value will be calculated as a the quartile coresponding to the value of this parameter.
    significance_level:
        Default is [0.1].

    Returns
    -------
    Statistical Properties.csv:
        file containing some statistical properties like mean, std, min, 5%, 25%,
        median, 75%, 95%, max, t_beg, t_end, nyr, q1.5, q2, q5, q10, q25, q50,
        q100, q200, q500.
    """
    gauges = time_series_df.columns.tolist()
    # List of the table output, including some general data and the return periods.
    col_csv = [
        "mean",
        "std",
        "min",
        "5%",
        "25%",
        "median",
        "75%",
        "95%",
        "max",
        "t_beg",
        "t_end",
        "nyr",
    ]
    rp_name = [
        "q1.5",
        "q2",
        "q5",
        "q10",
        "q25",
        "q50",
        "q100",
        "q200",
        "q500",
        "q1000",
    ]
    col_csv = col_csv + rp_name

    # In a table where duplicates are removed (np.unique), find the number of
    # gauges contained in the .csv file.
    # Declare a dataframe for the output file, with as index the gaugne numbers
    # and as columns all the output names.
    statistical_properties = pd.DataFrame(np.nan, index=gauges, columns=col_csv)
    statistical_properties.index.name = "id"
    if distribution == "GEV":
        distribution_properties = pd.DataFrame(
            np.nan,
            index=gauges,
            columns=["c", "loc", "scale", "D-static", "P-Value"],
        )
    else:
        distribution_properties = pd.DataFrame(
            np.nan,
            index=gauges,
            columns=["loc", "scale", "D-static", "P-Value"],
        )
    distribution_properties.index.name = "id"
    # required return periods
    T = [1.5, 2, 5, 10, 25, 50, 50, 100, 200, 500, 1000]
    T = np.array(T)
    # these values are the Non Exceedance probability (F) of the chosen
    # return periods F = 1 - (1/T)
    # Non Exceedance propabilities
    # F = [1/3, 0.5, 0.8, 0.9, 0.96, 0.98, 0.99, 0.995, 0.998]
    F = 1 - (1 / T)
    save_to = Path(save_to)
    # Iteration over all the gauge numbers.
    if save_plots:
        rpath = save_to.joinpath("figures")
        if not rpath.exists():
            # os.mkdir(rpath)
            rpath.mkdir(parents=True, exist_ok=True)

    for i in gauges:
        QTS = time_series_df.loc[:, i]
        # The time series is resampled to the annual maxima, and turned into a numpy array.
        # The hydrological year is 1-Nov/31-Oct (from Petrow and Merz, 2009, JoH).
        if not ams:
            ams_df = QTS.resample(ams_start).max().values
        else:
            ams_df = QTS.values

        if filter_out is not None:
            ams_df = ams_df[ams_df != filter_out]

        if estimate_parameters:
            # TODO: still to be tested and prepared for GEV
            # estimate the parameters through an optimization
            # alpha = (np.sqrt(6) / np.pi) * ams_df.std()
            # beta = ams_df.mean() - 0.5772 * alpha
            # param_dist = [beta, alpha]
            threshold = np.quantile(ams_df, quartile)
            if distribution == "GEV":
                dist = GEV(ams_df)
                param_dist = dist.estimateParameter(
                    method="optimization",
                    ObjFunc=Gumbel.ObjectiveFn,
                    threshold=threshold,
                )
            else:
                dist = Gumbel(ams_df)
                param_dist = dist.estimateParameter(
                    method="optimization",
                    ObjFunc=Gumbel.ObjectiveFn,
                    threshold=threshold,
                )
        else:
            # estimate the parameters through maximum liklehood method
            try:
                if distribution == "GEV":
                    dist = GEV(ams_df)
                    # defult parameter estimation method is maximum liklihood method
                    param_dist = dist.estimateParameter(method=method)
                else:
                    # A gumbel distribution is fitted to the annual maxima
                    dist = Gumbel(ams_df)
                    # defult parameter estimation method is maximum liklihood method
                    param_dist = dist.estimateParameter(method=method)
            except Exception as e:
                logger.warning(
                    f"The gauge {i} parameters could not be estimated because of {e}"
                )
                continue

        (
            distribution_properties.loc[i, "D-static"],
            distribution_properties.loc[i, "P-Value"],
        ) = dist.ks()
        if distribution == "GEV":
            distribution_properties.loc[i, "c"] = param_dist[0]
            distribution_properties.loc[i, "loc"] = param_dist[1]
            distribution_properties.loc[i, "scale"] = param_dist[2]
        else:
            distribution_properties.loc[i, "loc"] = param_dist[0]
            distribution_properties.loc[i, "scale"] = param_dist[1]

        # Return periods from the fitted distribution are stored.
        # get the Discharge coresponding to the return periods
        if distribution == "GEV":
            Qrp = dist.theporeticalEstimate(
                param_dist[0], param_dist[1], param_dist[2], F
            )
        else:
            Qrp = dist.theporeticalEstimate(param_dist[0], param_dist[1], F)

        # to get the Non Exceedance probability for a specific Value
        # sort the ams_df
        ams_df.sort()
        # calculate the F (Exceedence probability based on weibul)
        cdf_Weibul = PlottingPosition.weibul(ams_df)
        # Gumbel.probapilityPlot method calculates the theoretical values
        # based on the Gumbel distribution
        # parameters, theoretical cdf (or weibul), and calculate the confidence interval
        if save_plots:
            if distribution == "GEV":
                fig, ax = dist.probapilityPlot(
                    param_dist[0],
                    param_dist[1],
                    param_dist[2],
                    cdf_Weibul,
                    alpha=significance_level,
                    method=method,
                )
            else:
                fig, ax = dist.probapilityPlot(
                    param_dist[0],
                    param_dist[1],
                    cdf_Weibul,
                    alpha=significance_level,
                )

            fig[0].savefig(f"{save_to}/figures/{i}.png", format="png")
            plt.close()

            fig[1].savefig(f"{save_to}/figures/f-{i}.png", format="png")
            plt.close()

        statistical_properties.loc[i, "mean"] = QTS.mean()
        statistical_properties.loc[i, "std"] = QTS.std()
        statistical_properties.loc[i, "min"] = QTS.min()
        statistical_properties.loc[i, "5%"] = QTS.quantile(0.05)
        statistical_properties.loc[i, "25%"] = QTS.quantile(0.25)
        statistical_properties.loc[i, "median"] = QTS.quantile(0.50)
        statistical_properties.loc[i, "75%"] = QTS.quantile(0.75)
        statistical_properties.loc[i, "95%"] = QTS.quantile(0.95)
        statistical_properties.loc[i, "max"] = QTS.max()
        statistical_properties.loc[i, "t_beg"] = QTS.index.min()
        statistical_properties.loc[i, "t_end"] = QTS.index.max()
        if not ams:
            statistical_properties.loc[i, "nyr"] = (
                statistical_properties.loc[i, "t_end"]
                - statistical_properties.loc[i, "t_beg"]
            ).days / 365.25

        for irp, irp_name in zip(Qrp, rp_name):
            statistical_properties.loc[i, irp_name] = irp

        # Print for prompt and check progress.
        logger.info(f"Gauge {i} done.")
    return statistical_properties, distribution_properties
