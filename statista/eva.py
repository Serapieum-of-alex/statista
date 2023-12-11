"""Extreme value analysis."""
from typing import Union, Tuple
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from pandas import DataFrame

from statista.distributions import PlottingPosition, Distributions


def ams_analysis(
    time_series_df: DataFrame,
    ams: bool = False,
    ams_start: str = "A-OCT",
    save_plots: bool = False,
    save_to: str = None,
    filter_out: Union[float, int] = None,
    distribution: str = "GEV",
    method: str = "lmoments",
    obj_func: callable = None,
    quartile: float = 0,
    significance_level: float = 0.1,
) -> Tuple[DataFrame, DataFrame]:
    """ams_analysis.

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
        available methods are 'mle', 'mm', 'lmoments', 'optimization'. Default is "lmoments"
    obj_func: [callable]
        objective function to be used in the optimization method, default is None. for Gumbel distribution there is the
        Gumbel.objective_fn and similarly for the GEV distribution there is the GEV.objective_fn.
    quartile: [float]
        the quartile is only used when estinating the distribution parameters based on optimization and a threshould
        value, the threshould value will be calculated as a the quartile coresponding to the value of this parameter.
    significance_level:
        Default is [0.1].

    Returns
    -------
    DataFrame:
        Statistical properties like mean, std, min, 5%, 25%,
        median, 75%, 95%, max, t_beg, t_end, nyr, q1.5, q2, q5, q10, q25, q50,
        q100, q200, q500.

        id,mean,std,min,5%,25%,median,75%,95%,max,t_beg,t_end,nyr,q1.5,q2,q5,q10,q25,q50,q100,q200,q500,q1000
        Frankfurt,694.4,552.8,-9.0,-9.0,220.8,671.0,1090.0,1760.0,1990.0,1951.0,2004.0,,683.3,855.3,1261.6,1517.8,1827.5,2047.6,2047.6,2258.3,2460.8,2717.0
        Mainz,4153.3,1192.8,1150.0,2286.5,3415.0,4190.0,4987.5,5914.0,6920.0,1951.0,2004.0,,3627.9,4164.8,5203.5,5716.9,6217.2,6504.8,6504.8,6734.9,6919.9,7110.8
        Kaub,4327.1,1254.7,1190.0,2394.5,3635.0,4350.0,5147.5,6383.5,7160.0,1951.0,2004.0,,3761.3,4321.1,5425.0,5983.7,6539.7,6865.8,6865.8,7131.4,7348.7,7577.3
        Andernach,6333.4,2035.1,1470.0,3178.0,5175.0,6425.0,7412.5,9717.0,10400.0,1951.0,2004.0,,5450.1,6369.7,8129.5,8987.6,9813.9,10283.1,10283.1,10654.9,10950.9,11252.8
        Cologne,6489.3,2056.1,1580.0,3354.5,5277.5,6585.0,7560.0,9728.9,10700.0,1951.0,2004.0,,5583.6,6507.7,8297.0,9182.4,10046.1,10542.9,10542.9,10940.9,11261.1,11591.7
        Rees,6701.4,2094.5,1810.0,3556.5,5450.0,6575.0,7901.8,10005.0,11300.0,1951.0,2004.0,,5759.2,6693.5,8533.3,9463.1,10386.9,10928.2,10928.2,11368.4,11728.2,12106.0
        date,1977.5,15.7,1951.0,1953.7,1964.2,1977.5,1990.8,2001.3,2004.0,1951.0,2004.0,,1970.3,1977.4,1991.6,1998.7,2005.8,2010.0,2010.0,2013.4,2016.1,2019.1
    DataFrame:
        Distribution properties like the shape, location, and scale parameters of the fitted distribution, plus the
        D-static and P-Value of the KS test.

        id,c,loc,scale,D-static,P-Value
        Frankfurt,0.1,718.7,376.2,0.1,1.0
        Mainz,0.3,3743.8,1214.6,0.1,1.0
        Kaub,0.3,3881.6,1262.4,0.1,1.0
        Andernach,0.3,5649.1,2084.4,0.1,1.0
        Cologne,0.3,5783.0,2090.2,0.1,1.0
        Rees,0.3,5960.0,2107.2,0.1,1.0
        date,0.3,1971.8,16.2,0.1,1.0
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
    col_csv += rp_name

    # In a table where duplicates are removed (np.unique), find the number of
    # gauges contained in the .csv file.
    # Declare a dataframe for the output file, with as index the gaugne numbers
    # and as columns all the output names.
    statistical_properties = pd.DataFrame(np.nan, index=gauges, columns=col_csv)
    statistical_properties.index.name = "id"

    if distribution == "GEV":
        cols = ["c", "loc", "scale", "D-static", "P-Value"]
    else:
        cols = ["loc", "scale", "D-static", "P-Value"]

    distribution_properties = pd.DataFrame(
        np.nan,
        index=gauges,
        columns=cols,
    )
    distribution_properties.index.name = "id"
    # required return periods
    return_period = [1.5, 2, 5, 10, 25, 50, 50, 100, 200, 500, 1000]
    return_period = np.array(return_period)
    # these values are the Non Exceedance probability (F) of the chosen
    # return periods non_exceed_prop = 1 - (1/return_period)
    # Non Exceedance propabilities
    # non_exceed_prop = [1/3, 0.5, 0.8, 0.9, 0.96, 0.98, 0.99, 0.995, 0.998]
    non_exceed_prop = 1 - (1 / return_period)
    save_to = Path(save_to)
    # Iteration over all the gauge numbers.
    if save_plots:
        rpath = save_to.joinpath("figures")
        if not rpath.exists():
            rpath.mkdir(parents=True, exist_ok=True)

    for i in gauges:
        q_ts = time_series_df.loc[:, i]
        # The time series is resampled to the annual maxima, and turned into a numpy array.
        # The hydrological year is 1-Nov/31-Oct (from Petrow and Merz, 2009, JoH).
        if not ams:
            ams_df = q_ts.resample(ams_start).max().values
        else:
            ams_df = q_ts.values

        if filter_out is not None:
            ams_df = ams_df[ams_df != filter_out]

        dist = Distributions(distribution, data=ams_df)
        # estimate the parameters through the given method
        try:
            threshold = np.quantile(ams_df, quartile)
            param_dist = dist.fit_model(
                method=method,
                obj_func=obj_func,
                threshold=threshold,
            )
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
            distribution_properties.loc[i, "c"] = param_dist["shape"]
            distribution_properties.loc[i, "loc"] = param_dist["loc"]
            distribution_properties.loc[i, "scale"] = param_dist["scale"]
        else:
            distribution_properties.loc[i, "loc"] = param_dist["loc"]
            distribution_properties.loc[i, "scale"] = param_dist["scale"]

        # Return periods from the fitted distribution are stored.
        # get the Discharge coresponding to the return periods
        q_rp = dist.theoretical_estimate(param_dist, non_exceed_prop)

        # to get the Non Exceedance probability for a specific Value
        # sort the ams_df
        ams_df.sort()
        # calculate the F (Exceedence probability based on weibul)
        cdf_weibul = PlottingPosition.weibul(ams_df)
        # Gumbel.probapilityPlot method calculates the theoretical values
        # based on the Gumbel distribution
        # parameters, theoretical cdf (or weibul), and calculate the confidence interval
        if save_plots:
            fig, _ = dist.probability_plot(
                param_dist,
                cdf_weibul,
                alpha=significance_level,
                method=method,
            )

            fig[0].savefig(f"{save_to}/figures/{i}.png", format="png")
            plt.close()

            fig[1].savefig(f"{save_to}/figures/f-{i}.png", format="png")
            plt.close()

        statistical_properties.loc[i, "mean"] = q_ts.mean()
        statistical_properties.loc[i, "std"] = q_ts.std()
        statistical_properties.loc[i, "min"] = q_ts.min()
        statistical_properties.loc[i, "5%"] = q_ts.quantile(0.05)
        statistical_properties.loc[i, "25%"] = q_ts.quantile(0.25)
        statistical_properties.loc[i, "median"] = q_ts.quantile(0.50)
        statistical_properties.loc[i, "75%"] = q_ts.quantile(0.75)
        statistical_properties.loc[i, "95%"] = q_ts.quantile(0.95)
        statistical_properties.loc[i, "max"] = q_ts.max()
        statistical_properties.loc[i, "t_beg"] = q_ts.index.min()
        statistical_properties.loc[i, "t_end"] = q_ts.index.max()
        if not ams:
            statistical_properties.loc[i, "nyr"] = (
                statistical_properties.loc[i, "t_end"]
                - statistical_properties.loc[i, "t_beg"]
            ).days / 365.25

        for irp, irp_name in zip(q_rp, rp_name):
            statistical_properties.loc[i, irp_name] = irp

        # Print for prompt and check progress.
        logger.info(f"Gauge {i} done.")
    return statistical_properties, distribution_properties
