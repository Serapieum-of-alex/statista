"""Statistical distributions."""
from numbers import Number
from typing import Any, List, Tuple, Union, Dict, Callable
from abc import ABC, abstractmethod
import numpy as np
import scipy.optimize as so
from matplotlib.figure import Figure

from numpy import ndarray
from scipy.stats import chisquare, genextreme, gumbel_r, ks_2samp, norm, expon

from statista.parameters import Lmoments
from statista.tools import Tools as st
from statista.plot import Plot
from statista.confidence_interval import ConfidenceInterval

ninf = 1e-5

__all__ = [
    "PlottingPosition",
    "Gumbel",
    "GEV",
    "Exponential",
    "Normal",
    "Distributions",
]


class PlottingPosition:
    """PlottingPosition."""

    def __init__(self):
        pass

    @staticmethod
    def return_period(prob_non_exceed: Union[list, np.ndarray]) -> np.ndarray:
        """returnPeriod.

        Parameters
        ----------
        prob_non_exceed: [list/array]
            non exceedence probability.

        Returns
        -------
        array:
           return period.
        """
        prob_non_exceed = np.array(prob_non_exceed)
        t = 1 / (1 - prob_non_exceed)
        return t

    @staticmethod
    def weibul(data: Union[list, np.ndarray], return_period: int = False) -> np.ndarray:
        """Weibul.

        Weibul method to calculate the cumulative distribution function cdf or
        return period.

        Parameters
        ----------
        data : [list/array]
            list/array of the data.
        return_period : [bool]
            False to calculate the cumulative distribution function cdf or
            True to calculate the return period. Default=False

        Returns
        -------
        cdf/T: [list]
            list of cumulative distribution function or return period.
        """
        data = np.array(data)
        data.sort()
        n = len(data)
        cdf = np.array(range(1, n + 1)) / (n + 1)
        if not return_period:
            return cdf
        else:
            t = PlottingPosition.return_period(cdf)
            return t


class AbstractDistribution(ABC):
    """
    AbstractDistribution.
    """

    parameters: Dict[str, Union[float, Any]]
    cdf_Weibul: ndarray

    def __init__(
        self,
        data: Union[list, np.ndarray] = None,
        parameters: Dict[str, str] = None,
    ):
        """Gumbel.

        Parameters
        ----------
        data : [list]
            data time series.
        parameters: Dict[str, str]
            {"loc": val, "scale": val, "shape": value}
            - loc: [numeric]
                location parameter
            - scale: [numeric]
                scale parameter
        """
        if isinstance(data, list) or isinstance(data, np.ndarray):
            self.data = np.array(data)
            self.data_sorted = np.sort(data)
            self.cdf_Weibul = PlottingPosition.weibul(data)
            self.KStable = 1.22 / np.sqrt(len(self.data))

        self.parameters = parameters

        self.Dstatic = None
        self.KS_Pvalue = None
        self.chistatic = None
        self.chi_Pvalue = None

        pass

    @staticmethod
    @abstractmethod
    def _pdf_eq(
        data: Union[list, np.ndarray], parameters: Dict[str, Union[float, Any]]
    ) -> np.ndarray:
        pass

    @abstractmethod
    def pdf(
        self,
        parameters: Dict[str, Union[float, Any]],
        plot_figure: bool = False,
        figsize: tuple = (6, 5),
        xlabel: str = "Actual data",
        ylabel: str = "pdf",
        fontsize: Union[float, int] = 15,
        actual_data: Union[bool, np.ndarray] = True,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """pdf.

        Returns the value of Gumbel's pdf with parameters loc and scale at x .

        Parameters:
        -----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.

        kwargs:
            figsize: tuple = (6, 5),
            xlabel: str = "Actual data",
            ylabel: str = "pdf",
            fontsize: Union[float, int] = 15,
            actual_data: np.ndarray = None,

        Returns
        -------
        pdf : [array]
            probability density function pdf.
        """

        if actual_data is None:
            ts = self.data
        else:
            ts = actual_data

        pdf = self._pdf_eq(ts, parameters)

        if plot_figure:
            qx = np.linspace(
                float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
            )
            pdf_fitted = self.pdf(parameters, actual_data=qx)

            fig, ax = Plot.pdf(
                qx,
                pdf_fitted,
                self.data_sorted,
                figsize=figsize,
                xlabel=xlabel,
                ylabel=ylabel,
                fontsize=fontsize,
            )
            return pdf, fig, ax
        else:
            return pdf

    @staticmethod
    @abstractmethod
    def _cdf_eq(
        data: Union[list, np.ndarray], parameters: Dict[str, Union[float, Any]]
    ) -> np.ndarray:
        pass

    @abstractmethod
    def cdf(
        self,
        parameters: Dict[str, Union[float, Any]],
        plot_figure: bool = False,
        figsize: tuple = (6, 5),
        xlabel: str = "data",
        ylabel: str = "cdf",
        fontsize: int = 15,
        actual_data: Union[bool, np.ndarray] = True,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """cdf.

        cdf calculates the value of Gumbel's cdf with parameters loc and scale at x.

        parameter:
        ----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        """
        if isinstance(actual_data, bool):
            ts = self.data
        else:
            ts = actual_data

        cdf = self._cdf_eq(ts, parameters)

        if plot_figure:
            qx = np.linspace(
                float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
            )
            cdf_fitted = self.cdf(parameters, actual_data=qx)

            cdf_weibul = PlottingPosition.weibul(self.data_sorted)

            fig, ax = Plot.cdf(
                qx,
                cdf_fitted,
                self.data_sorted,
                cdf_weibul,
                figsize=figsize,
                xlabel=xlabel,
                ylabel=ylabel,
                fontsize=fontsize,
            )

            return cdf, fig, ax
        else:
            return cdf

    @abstractmethod
    def fit_model(
        self,
        method: str = "mle",
        obj_func: Callable = None,
        threshold: Union[None, float, int] = None,
        test: bool = True,
    ) -> Union[Dict[str, str], Any]:
        """estimateParameter.

        EstimateParameter estimate the distribution parameter based on MLM
        (Maximum liklihood method), if an objective function is entered as an input

        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for values below (x < C), now the likeliest parameters
        are those at the max value of mutiplication between two functions max(L1*L2).

        In this case the L1 is still the product of multiplication of probability
        density function's values at xi, but the L2 is the probability that threshold
        value C will be exceeded (1-F(C)).

        Parameters
        ----------
        obj_func : [function]
            function to be used to get the distribution parameters.
        threshold : [numeric]
            Value you want to consider only the greater values.
        method : [string]
            'mle', 'mm', 'lmoments', optimization
        test: [bool]
            Default is True.

        Returns
        -------
        Dict[str, str]:
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        """
        method = method.lower()
        if method not in ["mle", "mm", "lmoments", "optimization"]:
            raise ValueError(
                f"{method} value should be 'mle', 'mm', 'lmoments' or 'optimization'"
            )
        return method

    @staticmethod
    @abstractmethod
    def theoretical_estimate(
        parameters: Dict[str, Union[float, Any]], cdf: np.ndarray
    ) -> np.ndarray:
        """theporeticalEstimate.

        TheporeticalEstimate method calculates the theoretical values based on the Gumbel distribution

        Parameters:
        -----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        cdf: [list]
            cummulative distribution function/ Non Exceedence probability.

        Return:
        -------
        theoreticalvalue : [numeric]
            Value based on the theoretical distribution
        """
        pass

    @abstractmethod
    def ks(self) -> tuple:
        """Kolmogorov-Smirnov (KS) test.

        The smaller the D static the more likely that the two samples are drawn from the same distribution
        IF Pvalue < signeficance level ------ reject

        returns:
        --------
        Dstatic: [numeric]
            The smaller the D static the more likely that the two samples are drawn from the same distribution
        Pvalue : [numeric]
            IF Pvalue < signeficance level ------ reject the null hypotethis
        """
        if self.parameters is None:
            raise ValueError(
                "Value of parameters is unknown please use "
                "'EstimateParameter' to obtain estimate the distribution parameters"
            )
        qth = self.theoretical_estimate(self.parameters, self.cdf_Weibul)

        test = ks_2samp(self.data, qth)
        self.Dstatic = test.statistic
        self.KS_Pvalue = test.pvalue

        print("-----KS Test--------")
        print(f"Statistic = {test.statistic}")
        if self.Dstatic < self.KStable:
            print("Accept Hypothesis")
        else:
            print("reject Hypothesis")
        print(f"P value = {test.pvalue}")
        return test.statistic, test.pvalue

    @abstractmethod
    def chisquare(self) -> Union[tuple, None]:
        """
        chisquare test
        """
        if self.parameters is None:
            raise ValueError(
                "Value of loc/scale parameter is unknown please use "
                "'EstimateParameter' to obtain them"
            )

        qth = self.theoretical_estimate(self.parameters, self.cdf_Weibul)
        try:
            test = chisquare(st.standardize(qth), st.standardize(self.data))
            self.chistatic = test.statistic
            self.chi_Pvalue = test.pvalue
            print("-----chisquare Test-----")
            print("Statistic = " + str(test.statistic))
            print("P value = " + str(test.pvalue))
            return test.statistic, test.pvalue
        except Exception as e:
            print(e)
            return

    def confidence_interval(
        self,
        parameters: Dict[str, Union[float, Any]],
        prob_non_exceed: np.ndarray,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """confidence_interval.

        Parameters:
        -----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        prob_non_exceed : [list]
            Non Exceedence probability
        alpha : [numeric]
            alpha or SignificanceLevel is a value of the confidence interval.

        Return:
        -------
        parameters: Dict[str, str]
            {"loc": val, "scale": val, "shape": value}
            - loc: [numeric]
                location parameter
            - scale: [numeric]
                scale parameter
        q_upper : [list]
            upper bound coresponding to the confidence interval.
        q_lower : [list]
            lower bound coresponding to the confidence interval.
        """
        pass

    def probability_plot(
        self,
        parameters: Dict[str, Union[float, Any]],
        prob_non_exceed: np.ndarray,
        alpha: float = 0.1,
        fig1size: tuple = (10, 5),
        fig2size: tuple = (6, 6),
        xlabel: str = "Actual data",
        ylabel: str = "cdf",
        fontsize: int = 15,
    ) -> Tuple[List[Figure], list]:
        """probapilityPlot.

        ProbapilityPlot method calculates the theoretical values based on the Gumbel distribution
        parameters, theoretical cdf (or weibul), and calculate the confidence interval.

        Parameters
        ----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        prob_non_exceed : [np.ndarray]
            theoretical cdf calculated using weibul or using the distribution cdf function.
        alpha : [float]
            value between 0 and 1.
        fig1size: [tuple]
            Default is (10, 5)
        fig2size: [tuple]
            Default is (6, 6)
        xlabel: [str]
            Default is "Actual data"
        ylabel: [str]
            Default is "cdf"
        fontsize: [float]
            Default is 15.

        Returns
        -------
        Qth : [list]
            theoretical generated values based on the theoretical cdf calculated from
            weibul or the distribution parameters.
        q_upper : [list]
            upper bound coresponding to the confidence interval.
        q_lower : [list]
            lower bound coresponding to the confidence interval.
        """
        pass


class Gumbel(AbstractDistribution):
    """Gumbel distribution."""

    cdf_Weibul: ndarray
    parameters: dict[str, Union[float, Any]]
    data: ndarray

    def __init__(
        self,
        data: Union[list, np.ndarray] = None,
        parameters: Dict[str, str] = None,
    ):
        """Gumbel.

        Parameters
        ----------
        data : [list]
            data time series.
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        """
        super().__init__(data, parameters)
        pass

    @staticmethod
    def _pdf_eq(
        data: Union[list, np.ndarray], parameters: Dict[str, Union[float, Any]]
    ) -> np.ndarray:
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale <= 0:
            raise ValueError("Scale parameter is negative")
        # z = (ts - loc) / scale
        # pdf = (1.0 / scale) * (np.exp(-(z + (np.exp(-z)))))
        pdf = gumbel_r.pdf(data, loc=loc, scale=scale)
        return pdf

    def pdf(
        self,
        parameters: Dict[str, Union[float, Any]],
        plot_figure: bool = False,
        actual_data: np.ndarray = None,
        *args,
        **kwargs,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """pdf.

        Returns the value of Gumbel's pdf with parameters loc and scale at x .

        Parameters:
        -----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        actual_data : [bool/array]
            true if you want to calculate the pdf for the actual time series, array
            if you want to calculate the pdf for a theoretical time series
        plot_figure: [bool]
            Default is False.
        kwargs:
            figsize: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "pdf".
            fontsize: [int]
                Default is 15.

        Returns
        -------
        pdf : [array]
            probability density function pdf.
        """
        result = super().pdf(
            parameters,
            actual_data=actual_data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )
        return result

    @staticmethod
    def _cdf_eq(
        data: Union[list, np.ndarray], parameters: Dict[str, Union[float, Any]]
    ) -> np.ndarray:
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale <= 0:
            raise ValueError("Scale parameter is negative")
        # z = (ts - loc) / scale
        # cdf = np.exp(-np.exp(-z))
        cdf = gumbel_r.cdf(data, loc=loc, scale=scale)
        return cdf

    def cdf(
        self,
        parameters: Dict[str, Union[float, Any]],
        plot_figure: bool = False,
        actual_data: Union[bool, np.ndarray] = True,
        *args,
        **kwargs,
    ) -> Union[
        Tuple[np.ndarray, Figure, Any], np.ndarray
    ]:  # pylint: disable=arguments-differ
        """cdf.

        cdf calculates the value of Gumbel's cdf with parameters loc and scale at x.

        parameter:
        ----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        actual_data : [bool/array]
            true if you want to calculate the pdf for the actual time series, array
            if you want to calculate the pdf for a theoretical time series
        plot_figure: [bool]
            Default is False.
        kwargs:
            figsize: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "cdf".
            fontsize: [int]
                Default is 15.
        """
        result = super().cdf(
            parameters,
            actual_data=actual_data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )
        return result

    def get_rp(self, loc, scale, data):
        """getRP.

            getRP calculates the return period for a list/array of values or a single value.

        Parameters
        ----------
        data:[list/array/float]
            value you want the coresponding return value for
        loc: [float]
            location parameter
        scale: [float]
            scale parameter

        Returns
        -------
        float:
            return period
        """
        # if isinstance(data, list) or isinstance(data, np.ndarray):
        cdf = self.cdf(loc, scale, actual_data=data)
        # else:
        #     cdf = gumbel_r.cdf(data, loc, scale)

        rp = 1 / (1 - cdf)

        return rp

    @staticmethod
    def objective_fn(p, x):
        """ObjectiveFn.

        Link :
        https://stackoverflow.com/questions/23217484/how-to-find-parameters-of-gumbels-distribution-using-scipy-optimize

        Parameters
        ----------
        p:
        x:
        """
        threshold = p[0]
        loc = p[1]
        scale = p[2]

        x1 = x[x < threshold]
        nx2 = len(x[x >= threshold])
        # pdf with a scaled pdf
        # L1 is pdf based
        parameters = {"loc": loc, "scale": scale}
        pdf = Gumbel._pdf_eq(x1, parameters)
        cdf = Gumbel._cdf_eq(threshold, parameters)
        l1 = (-np.log((pdf / scale))).sum()
        # L2 is cdf based
        l2 = (-np.log(1 - cdf)) * nx2
        # print x1, nx2, L1, L2
        return l1 + l2

    def fit_model(
        self,
        method: str = "mle",
        obj_func: Callable = None,
        threshold: Union[None, float, int] = None,
        test: bool = True,
    ) -> Dict[str, float]:
        """fit_model.

        EstimateParameter estimate the distribution parameter based on MLM
        (Maximum liklihood method), if an objective function is entered as an input

        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for values below (x < C), now the likeliest parameters
        are those at the max value of mutiplication between two functions max(L1*L2).

        In this case the L1 is still the product of multiplication of probability
        density function's values at xi, but the L2 is the probability that threshold
        value C will be exceeded (1-F(C)).

        Parameters
        ----------
        obj_func : [function]
            function to be used to get the distribution parameters.
        threshold : [numeric]
            Value you want to consider only the greater values.
        method : [string]
            'mle', 'mm', 'lmoments', optimization
        test: [bool]
            Default is True.

        Returns
        -------
        Dict[str, str]:
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        """
        # obj_func = lambda p, x: (-np.log(Gumbel.pdf(x, p[0], p[1]))).sum()
        # #first we make a simple Gumbel fit
        # Par1 = so.fmin(obj_func, [0.5,0.5], args=(np.array(data),))
        method = super().fit_model(method=method)

        if method == "mle" or method == "mm":
            param = list(gumbel_r.fit(self.data, method=method))
        elif method == "lmoments":
            lm = Lmoments(self.data)
            lmu = lm.Lmom()
            param = Lmoments.gumbel(lmu)
        elif method == "optimization":
            if obj_func is None or threshold is None:
                raise TypeError("threshold should be numeric value")

            param = gumbel_r.fit(self.data, method="mle")
            # then we use the result as starting value for your truncated Gumbel fit
            param = so.fmin(
                obj_func,
                [threshold, param[0], param[1]],
                args=(self.data,),
                maxiter=500,
                maxfun=500,
            )
            param = [param[1], param[2]]
        else:
            raise ValueError(f"The given: {method} does not exist")

        param = {"loc": param[0], "scale": param[1]}
        self.parameters = param

        if test:
            self.ks()
            self.chisquare()

        return param

    @staticmethod
    def theoretical_estimate(
        parameters: Dict[str, Union[float, Any]], cdf: np.ndarray
    ) -> np.ndarray:
        """theporeticalEstimate.

        TheporeticalEstimate method calculates the theoretical values based on the Gumbel distribution

        Parameters:
        -----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        cdf: [list]
            cummulative distribution function/ Non Exceedence probability.

        Return:
        -------
        theoreticalvalue : [numeric]
            Value based on the theoretical distribution
        """
        loc = parameters.get("loc")
        scale = parameters.get("scale")

        if scale <= 0:
            raise ValueError("Scale parameter is negative")

        if any(cdf) <= 0 or any(cdf) > 1:
            raise ValueError("cdf Value Invalid")

        cdf = np.array(cdf)
        # Qth = loc - scale * (np.log(-np.log(cdf)))

        # the main equation form scipy
        qth = gumbel_r.ppf(cdf, loc=loc, scale=scale)
        return qth

    def ks(self) -> tuple:
        """Kolmogorov-Smirnov (KS) test.

        The smaller the D static the more likely that the two samples are drawn from the same distribution
        IF Pvalue < signeficance level ------ reject

        returns:
        --------
        Dstatic: [numeric]
            The smaller the D static the more likely that the two samples are drawn from the same distribution
        Pvalue : [numeric]
            IF Pvalue < signeficance level ------ reject the null hypotethis
        """
        return super().ks()

    def chisquare(self) -> tuple:
        """chisquare test"""
        return super().chisquare()

    def confidence_interval(
        self,
        parameters: Dict[str, Union[float, Any]],
        prob_non_exceed: np.ndarray,
        alpha: float = 0.1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """confidence_interval.

        Parameters:
        -----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        prob_non_exceed : [list]
            Non Exceedence probability
        alpha : [numeric]
            alpha or SignificanceLevel is a value of the confidence interval.

        Return:
        -------
        parameters: Dict[str, str]
            {"loc": val, "scale": val, "shape": value}
            - loc: [numeric]
                location parameter
            - scale: [numeric]
                scale parameter
        q_upper : [list]
            upper bound coresponding to the confidence interval.
        q_lower : [list]
            lower bound coresponding to the confidence interval.
        """
        scale = parameters.get("scale")

        if scale <= 0:
            raise ValueError("Scale parameter is negative")

        qth = self.theoretical_estimate(parameters, prob_non_exceed)
        y = [-np.log(-np.log(j)) for j in prob_non_exceed]
        std_error = [
            (scale / np.sqrt(len(self.data)))
            * np.sqrt(1.1087 + 0.5140 * j + 0.6079 * j**2)
            for j in y
        ]
        v = norm.ppf(1 - alpha / 2)
        q_upper = np.array([qth[j] + v * std_error[j] for j in range(len(self.data))])
        q_lower = np.array([qth[j] - v * std_error[j] for j in range(len(self.data))])
        return q_upper, q_lower

    def probability_plot(
        self,
        parameters: Dict[str, Union[float, Any]],
        cdf: Union[np.ndarray, list],
        alpha: float = 0.1,
        fig1_size: Tuple[float, float] = (10, 5),
        fig2_size: Tuple[float, float] = (6, 6),
        xlabel: str = "Actual data",
        ylabel: str = "cdf",
        fontsize: int = 15,
    ) -> tuple[list[Figure], list[Any]]:  # pylint: disable=arguments-differ
        """probapilityPlot.

        ProbapilityPlot method calculates the theoretical values based on the Gumbel distribution
        parameters, theoretical cdf (or weibul), and calculate the confidence interval.

        Parameters
        ----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        cdf : [np.ndarray]
            theoretical cdf calculated using weibul or using the distribution cdf function.
        alpha : [float]
            value between 0 and 1.
        fig1_size: [tuple]
            Default is (10, 5)
        fig2_size: [tuple]
            Default is (6, 6)
        xlabel: [str]
            Default is "Actual data"
        ylabel: [str]
            Default is "cdf"
        fontsize: [float]
            Default is 15.

        Returns
        -------
        Qth : [list]
            theoretical generated values based on the theoretical cdf calculated from
            weibul or the distribution parameters.
        q_upper : [list]
            upper bound coresponding to the confidence interval.
        q_lower : [list]
            lower bound coresponding to the confidence interval.
        """
        scale = parameters.get("scale")

        if scale <= 0:
            raise ValueError("Scale parameter is negative")

        q_th = self.theoretical_estimate(parameters, cdf)
        q_upper, q_lower = self.confidence_interval(parameters, cdf, alpha)

        q_x = np.linspace(
            float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
        )
        pdf_fitted = self.pdf(parameters, actual_data=q_x)
        cdf_fitted = self.cdf(parameters, actual_data=q_x)

        fig, ax = Plot.details(
            q_x,
            q_th,
            self.data,
            pdf_fitted,
            cdf_fitted,
            cdf,
            q_lower,
            q_upper,
            alpha,
            fig1_size=fig1_size,
            fig2_size=fig2_size,
            xlabel=xlabel,
            ylabel=ylabel,
            fontsize=fontsize,
        )

        return fig, ax


class GEV(AbstractDistribution):
    """GEV (Genalized Extreme value statistics)"""

    parameters: dict[str, Union[float, Any]]
    data: ndarray

    def __init__(
        self,
        data: Union[list, np.ndarray] = None,
        parameters: Dict[str, str] = None,
    ):
        """GEV.

        Parameters
        ----------
        data : [list]
            data time series.

        parameters: Dict[str, str]
            {"loc": val, "scale": val, "shape": value}
            - loc: [numeric]
                location parameter of the GEV distribution.
            - scale: [numeric]
                scale parameter of the GEV distribution.
            - shape: [numeric]
                shape parameter of the GEV distribution.
        """
        super().__init__(data, parameters)
        pass

    @staticmethod
    def _pdf_eq(
        data: Union[list, np.ndarray], parameters: Dict[str, Union[float, Any]]
    ) -> np.ndarray:
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        shape = parameters.get("shape")
        # pdf = []
        # for ts_i in ts:
        #     z = (ts_i - loc) / scale
        #
        #     # Gumbel
        #     if shape == 0:
        #         val = np.exp(-(z + np.exp(-z)))
        #         pdf.append((1 / scale) * val)
        #         continue
        #
        #     # GEV
        #     y = 1 - shape * z
        #     if y > ninf:
        #         # np.log(y) = ln(y)
        #         # ln is the inverse of e
        #         lnY = (-1 / shape) * np.log(y)
        #         val = np.exp(-(1 - shape) * lnY - np.exp(-lnY))
        #         pdf.append((1 / scale) * val)
        #         continue
        #
        #     if shape < 0:
        #         pdf.append(0)
        #         continue
        #     pdf.append(0)
        #
        # if len(pdf) == 1:
        #     pdf = pdf[0]

        # pdf = np.array(pdf)
        pdf = genextreme.pdf(data, loc=loc, scale=scale, c=shape)
        return pdf

    def pdf(
        self,
        parameters: Dict[str, Union[float, Any]],
        plot_figure: bool = False,
        actual_data: np.ndarray = None,
        *args,
        **kwargs,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """pdf.

        Returns the value of GEV's pdf with parameters loc and scale at x .

        Parameters
        ----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val, "shape": value}
            - loc: [numeric]
                location parameter of the GEV distribution.
            - scale: [numeric]
                scale parameter of the GEV distribution.
            - shape: [numeric]
                shape parameter of the GEV distribution.
        actual_data : [bool/array]
            true if you want to calculate the pdf for the actual time series, array
            if you want to calculate the pdf for a theoretical time series
        plot_figure: [bool]
            Default is False.
        kwargs:
            figsize: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "pdf".
            fontsize: [int]
                Default is 15

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        result = super().pdf(
            parameters,
            actual_data=actual_data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )

        return result

    @staticmethod
    def _cdf_eq(
        data: Union[list, np.ndarray], parameters: Dict[str, Union[float, Any]]
    ) -> np.ndarray:
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        shape = parameters.get("shape")
        # equation https://www.rdocumentation.org/packages/evd/versions/2.3-6/topics/fextreme
        # z = (ts - loc) / scale
        # if shape == 0:
        #     # GEV is Gumbel distribution
        #     cdf = np.exp(-np.exp(-z))
        # else:
        #     y = 1 - shape * z
        #     cdf = list()
        #     for y_i in y:
        #         if y_i > ninf:
        #             logY = -np.log(y_i) / shape
        #             cdf.append(np.exp(-np.exp(-logY)))
        #         elif shape < 0:
        #             cdf.append(0)
        #         else:
        #             cdf.append(1)
        #
        # cdf = np.array(cdf)
        cdf = genextreme.cdf(data, c=shape, loc=loc, scale=scale)
        return cdf

    def cdf(
        self,
        parameters: Dict[str, Union[float, Any]],
        plot_figure: bool = False,
        actual_data: Union[bool, np.ndarray] = True,
        *args,
        **kwargs,
    ) -> Union[
        Tuple[np.ndarray, Figure, Any], np.ndarray
    ]:  # pylint: disable=arguments-differ
        """cdf.

        cdf calculates the value of Gumbel's cdf with parameters loc and scale at x.

        parameter:
        ----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        actual_data : [bool/array]
            true if you want to calculate the pdf for the actual time series, array
            if you want to calculate the pdf for a theoretical time series
        plot_figure: [bool]
            Default is False.
        kwargs:
            figsize: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "cdf".
            fontsize: [int]
                Default is 15.
        """
        result = super().cdf(
            parameters,
            actual_data=actual_data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )
        return result

    def get_rp(self, parameters: Dict[str, Union[float, Any]], data: np.ndarray):
        """get_rp.

            getRP calculates the return period for a list/array of values or a single value.

        Parameters
        ----------
        data:[list/array/float]
            value you want the coresponding return value for
        parameters: Dict[str, str]
            {"loc": val, "scale": val, "shape": value}
            - shape: [float]
                shape parameter
            - loc: [float]
                location parameter
            - scale: [float]
                scale parameter

        Returns
        -------
        float:
            return period
        """
        cdf = self.cdf(parameters, actual_data=data)

        rp = 1 / (1 - cdf)

        return rp

    def fit_model(
        self,
        method: str = "mle",
        obj_func=None,
        threshold: Union[int, float, None] = None,
        test: bool = True,
    ) -> Dict[str, float]:
        """estimateParameter.

        EstimateParameter estimate the distribution parameter based on MLM
        (Maximum liklihood method), if an objective function is entered as an input

        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for values below (x < C), now the likeliest parameters
        are those at the max value of mutiplication between two functions max(L1*L2).

        In this case the L1 is still the product of multiplication of probability
        density function's values at xi, but the L2 is the probability that threshold
        value C will be exceeded (1-F(C)).

        Parameters
        ----------
        obj_func : [function]
            function to be used to get the distribution parameters.
        threshold : [numeric]
            Value you want to consider only the greater values.
        method : [string]
            'mle', 'mm', 'lmoments', optimization
        test: bool
            Default is True

        Returns
        -------
        Param : [list]
            shape, loc, scale parameter of the gumbel distribution in that order.
        """
        # obj_func = lambda p, x: (-np.log(Gumbel.pdf(x, p[0], p[1]))).sum()
        # #first we make a simple Gumbel fit
        # Par1 = so.fmin(obj_func, [0.5,0.5], args=(np.array(data),))

        method = super().fit_model(method=method)
        if method == "mle" or method == "mm":
            param = list(genextreme.fit(self.data, method=method))
        elif method == "lmoments":
            lm = Lmoments(self.data)
            lmu = lm.Lmom()
            param = Lmoments.gev(lmu)
        elif method == "optimization":
            if obj_func is None or threshold is None:
                raise TypeError("obj_func and threshold should be numeric value")

            param = genextreme.fit(self.data, method="mle")
            # then we use the result as starting value for your truncated Gumbel fit
            param = so.fmin(
                obj_func,
                [threshold, param[0], param[1], param[2]],
                args=(self.data,),
                maxiter=500,
                maxfun=500,
            )
            param = [param[1], param[2], param[3]]
        else:
            raise ValueError(f"The given: {method} does not exist")

        param = {"loc": param[1], "scale": param[2], "shape": param[0]}
        self.parameters = param

        if test:
            self.ks()
            try:
                self.chisquare()
            except ValueError:
                print("chisquare test failed")

        return param

    @staticmethod
    def theoretical_estimate(
        parameters: Dict[str, Union[float, Any]],
        cdf: np.ndarray,
    ) -> np.ndarray:
        """TheporeticalEstimate.

        TheporeticalEstimate method calculates the theoretical values based on a given  non exceedence probability

        Parameters:
        -----------
        param : [list]
            location ans scale parameters of the gumbel distribution.
        cdf: [list]
            cummulative distribution function/ Non Exceedence probability.

        Return:
        -------
        theoreticalvalue : [numeric]
            Value based on the theoretical distribution
        """
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        shape = parameters.get("shape")

        if scale <= 0:
            raise ValueError("Parameters Invalid")

        if shape is None:
            raise ValueError("Shape parameter should not be None")

        if any(cdf) < 0 or any(cdf) > 1:
            raise ValueError("cdf Value Invalid")

        # q_th = list()
        # for i in range(len(cdf)):
        #     if cdf[i] <= 0 or cdf[i] >= 1:
        #         if cdf[i] == 0 and shape < 0:
        #             q_th.append(loc + scale / shape)
        #         elif cdf[i] == 1 and shape > 0:
        #             q_th.append(loc + scale / shape)
        #         else:
        #             raise ValueError(str(cdf[i]) + " value of cdf is Invalid")
        #     # cdf = np.array(cdf)
        #     Y = -np.log(-np.log(cdf[i]))
        #     if shape != 0:
        #         Y = (1 - np.exp(-1 * shape * Y)) / shape
        #
        #     q_th.append(loc + scale * Y)
        # q_th = np.array(q_th)
        # the main equation from scipy
        q_th = genextreme.ppf(cdf, shape, loc=loc, scale=scale)
        return q_th

    def ks(self):
        """Kolmogorov-Smirnov (KS) test.

        The smaller the D static the more likely that the two samples are drawn from the same distribution
        IF Pvalue < signeficance level ------ reject

        returns:
        --------
            Dstatic: [numeric]
                The smaller the D static the more likely that the two samples are drawn from the same distribution
            Pvalue : [numeric]
                IF Pvalue < signeficance level ------ reject the null hypotethis
        """
        return super().ks()

    def chisquare(self) -> tuple:
        """chisquare test"""
        return super().chisquare()

    def confidence_interval(
        self,
        parameters: Dict[str, Union[float, Any]],
        prob_non_exceed: np.ndarray,
        alpha: float = 0.1,
        statfunction=np.average,
        n_samples: int = 100,
        method: str = "lmoments",
        **kargs,
    ):  # pylint: disable=arguments-differ
        """confidence_interval.

        Parameters:
        -----------
        loc : [numeric]
            location parameter of the gumbel distribution.
        scale : [numeric]
            scale parameter of the gumbel distribution.
        prob_non_exceed : [list]
            Non Exceedence probability
        alpha : [numeric]
            alpha or SignificanceLevel is a value of the confidence interval.
        statfunction: [callable]
            Default is np.average.
        n_samples: [int]
            number of samples generated by the bootstrap method Default is 100.
        method: [str]
            method used to fit the generated samples from the bootstrap method ["lmoments", "mle", "mm"]. Default is
            "lmoments".

        Return:
        -------
        q_upper : [list]
            upper bound coresponding to the confidence interval.
        q_lower : [list]
            lower bound coresponding to the confidence interval.
        """
        scale = parameters.get("scale")
        if scale <= 0:
            raise ValueError("Scale parameter is negative")

        ci = ConfidenceInterval.boot_strap(
            self.data,
            statfunction=statfunction,
            gevfit=parameters,
            F=prob_non_exceed,
            alpha=alpha,
            n_samples=n_samples,
            method=method,
            **kargs,
        )
        q_lower = ci["lb"]
        q_upper = ci["ub"]

        return q_upper, q_lower

    def probability_plot(
        self,
        parameters: Dict[str, Union[float, Any]],
        cdf: Union[np.ndarray, list],
        alpha: Number = 0.1,
        func: Callable = None,
        method: str = "lmoments",
        n_samples=100,
        fig1_size=(10, 5),
        fig2_size=(6, 6),
        xlabel="Actual data",
        ylabel="cdf",
        fontsize=15,
    ):
        """probapilityPlot.

        ProbapilityPlot method calculates the theoretical values based on the Gumbel distribution
        parameters, theoretical cdf (or weibul), and calculate the confidence interval.

        Parameters
        ----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val, shape: val}
            - loc : [numeric]
                Location parameter of the GEV distribution.
            - scale : [numeric]
                Scale parameter of the GEV distribution.
            - shape: [float, int]
                Shape parameter for the GEV distribution.
        cdf : [list]
            Theoretical cdf calculated using weibul or using the distribution cdf function.
        method: [str]
            Method used to fit the generated samples from the bootstrap method ["lmoments", "mle", "mm"]. Default is
            "lmoments".
        alpha : [float]
            Value between 0 and 1.
        fontsize : [numeric]
            Font size of the axis labels and legend
        ylabel : [string]
            y label string
        xlabel : [string]
            X label string
        fig1_size : [tuple]
            size of the pdf and cdf figure
        fig2_size : [tuple]
            size of the confidence interval figure
        n_samples : [integer]
            number of points in the condidence interval calculation
        alpha : [numeric]
            alpha or SignificanceLevel is a value of the confidence interval.
        func : [function]
            function to be used in the confidence interval calculation.
        """
        scale = parameters.get("scale")

        if scale <= 0:
            raise ValueError("Scale parameter is negative")

        q_th = self.theoretical_estimate(parameters, cdf)
        if func is None:
            func = GEV.ci_func

        ci = ConfidenceInterval.boot_strap(
            self.data,
            statfunction=func,
            gevfit=parameters,
            n_samples=n_samples,
            F=cdf,
            method=method,
        )
        q_lower = ci["lb"]
        q_upper = ci["ub"]

        q_x = np.linspace(
            float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
        )
        pdf_fitted = self.pdf(parameters, actual_data=q_x)
        cdf_fitted = self.cdf(parameters, actual_data=q_x)

        fig, ax = Plot.details(
            q_x,
            q_th,
            self.data,
            pdf_fitted,
            cdf_fitted,
            cdf,
            q_lower,
            q_upper,
            alpha,
            fig1_size=fig1_size,
            fig2_size=fig2_size,
            xlabel=xlabel,
            ylabel=ylabel,
            fontsize=fontsize,
        )

        return fig, ax

        # The function to bootstrap

    @staticmethod
    def ci_func(data: Union[list, np.ndarray], **kwargs):
        """GEV distribution function.

        Parameters
        ----------
        data: [list, np.ndarray]
            time series
        kwargs:
            - gevfit: [list]
                GEV parameter [shape, location, scale]
            - F: [list]
                Non Exceedence probability
            - method: [str]
                method used to fit the generated samples from the bootstrap method ["lmoments", "mle", "mm"]. Default is
                "lmoments".
        """
        gevfit = kwargs["gevfit"]
        prob_non_exceed = kwargs["F"]
        method = kwargs["method"]
        # generate theoretical estimates based on a random cdf, and the dist parameters
        sample = GEV.theoretical_estimate(gevfit, np.random.rand(len(data)))

        # get parameters based on the new generated sample
        dist = GEV(sample)
        new_param = dist.fit_model(method=method, test=False)

        # return period
        # T = np.arange(0.1, 999.1, 0.1) + 1
        # +1 in order not to make 1- 1/0.1 = -9
        # T = np.linspace(0.1, 999, len(data)) + 1
        # coresponding theoretical estimate to T
        # prob_non_exceed = 1 - 1 / T
        q_th = GEV.theoretical_estimate(new_param, prob_non_exceed)

        res = list(new_param.values())
        res.extend(q_th)
        return tuple(res)


# class Frechet:
#
#     """
#     f(x: threshold, scale) = (1/scale) e **(- (x-threshold)/scale)
#
#     """
#
#     def __init__(
#         self,
#         data: Union[list, np.ndarray] = None,
#         loc: Union[int, float] = None,
#         scale: Union[int, float] = None,
#     ):
#         """Gumbel.
#
#         Parameters
#         ----------
#         data : [list]
#             data time series.
#         loc: [numeric]
#             location parameter
#         scale: [numeric]
#             scale parameter
#         """
#         if isinstance(data, list) or isinstance(data, np.ndarray):
#             self.data = np.array(data)
#             self.data_sorted = np.sort(data)
#             self.cdf_Weibul = PlottingPosition.weibul(data)
#             self.KStable = 1.22 / np.sqrt(len(self.data))
#
#         self.loc = loc
#         self.scale = scale
#         self.Dstatic = None
#         self.KS_Pvalue = None
#         self.chistatic = None
#         self.chi_Pvalue = None
#
#     def pdf(
#         self,
#         loc: Union[float, int],
#         scale: Union[float, int],
#         plot_figure: bool = False,
#         figsize: tuple = (6, 5),
#         xlabel: str = "Actual data",
#         ylabel: str = "pdf",
#         fontsize: Union[float, int] = 15,
#         actual_data: Union[bool, np.ndarray] = True,
#     ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
#         """pdf.
#
#         Returns the value of Gumbel's pdf with parameters loc and scale at x .
#
#         Parameters:
#         -----------
#         loc : [numeric]
#             location parameter of the gumbel distribution.
#         scale : [numeric]
#             scale parameter of the gumbel distribution.
#
#         Returns
#         -------
#         pdf : [array]
#             probability density function pdf.
#         """
#         if scale <= 0:
#             raise ValueError("Scale parameter is negative")
#
#         if isinstance(actual_data, bool):
#             ts = self.data
#         else:
#             ts = actual_data
#
#         # pdf = []
#         #
#         # for i in ts:
#         #     Y = (i - loc) / scale
#         #     if Y <= 0:
#         #         pdf.append(0)
#         #     else:
#         #         pdf.append(np.exp(-Y) / scale)
#         #
#         # if len(pdf) == 1:
#         #     pdf = pdf[0]
#
#         pdf = expon.pdf(ts, loc=loc, scale=scale)
#         if plot_figure:
#             q_x = np.linspace(
#                 float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
#             )
#             pdf_fitted = self.pdf(loc, scale, actual_data=q_x)
#
#             fig, ax = Plot.pdf(
#                 q_x,
#                 pdf_fitted,
#                 self.data_sorted,
#                 figsize=figsize,
#                 xlabel=xlabel,
#                 ylabel=ylabel,
#                 fontsize=fontsize,
#             )
#             return pdf, fig, ax
#         else:
#             return pdf
#
#     def cdf(
#         self,
#         loc: Union[float, int],
#         scale: Union[float, int],
#         plot_figure: bool = False,
#         figsize: tuple = (6, 5),
#         xlabel: str = "data",
#         ylabel: str = "cdf",
#         fontsize: int = 15,
#         actual_data: Union[bool, np.ndarray] = True,
#     ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
#         """cdf.
#
#         cdf calculates the value of Gumbel's cdf with parameters loc and scale at x.
#
#         parameter:
#         ----------
#         loc : [numeric]
#             location parameter of the gumbel distribution.
#         scale : [numeric]
#             scale parameter of the gumbel distribution.
#         """
#         if scale <= 0:
#             raise ValueError("Scale parameter is negative")
#         if loc <= 0:
#             raise ValueError("Threshold parameter should be greater than zero")
#
#         if isinstance(actual_data, bool):
#             ts = self.data
#         else:
#             ts = actual_data
#
#         # Y = (ts - loc) / scale
#         # cdf = 1 - np.exp(-Y)
#         #
#         # for i in range(0, len(cdf)):
#         #     if cdf[i] < 0:
#         #         cdf[i] = 0
#         cdf = expon.cdf(ts, loc=loc, scale=scale)
#
#         if plot_figure:
#             q_x = np.linspace(
#                 float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
#             )
#             cdf_fitted = self.cdf(loc, scale, actual_data=q_x)
#
#             cdf_Weibul = PlottingPosition.weibul(self.data_sorted)
#
#             fig, ax = Plot.cdf(
#                 q_x,
#                 cdf_fitted,
#                 self.data_sorted,
#                 cdf_Weibul,
#                 figsize=figsize,
#                 xlabel=xlabel,
#                 ylabel=ylabel,
#                 fontsize=fontsize,
#             )
#
#             return cdf, fig, ax
#         else:
#             return cdf
#
#     def estimateParameter(
#         self,
#         method: str = "mle",
#         obj_func=None,
#         threshold: Union[int, float, None] = None,
#         test: bool = True,
#     ) -> tuple:
#         """estimateParameter.
#
#         EstimateParameter estimate the distribution parameter based on MLM
#         (Maximum liklihood method), if an objective function is entered as an input
#
#         There are two likelihood functions (L1 and L2), one for values above some
#         threshold (x>=C) and one for values below (x < C), now the likeliest parameters
#         are those at the max value of mutiplication between two functions max(L1*L2).
#
#         In this case the L1 is still the product of multiplication of probability
#         density function's values at xi, but the L2 is the probability that threshold
#         value C will be exceeded (1-F(C)).
#
#         Parameters
#         ----------
#         obj_func : [function]
#             function to be used to get the distribution parameters.
#         threshold : [numeric]
#             Value you want to consider only the greater values.
#         method : [string]
#             'mle', 'mm', 'lmoments', optimization
#         test: bool
#             Default is True
#
#         Returns
#         -------
#         Param : [list]
#             shape, loc, scale parameter of the gumbel distribution in that order.
#         """
#         # obj_func = lambda p, x: (-np.log(Gumbel.pdf(x, p[0], p[1]))).sum()
#         # #first we make a simple Gumbel fit
#         # Par1 = so.fmin(obj_func, [0.5,0.5], args=(np.array(data),))
#         method = method.lower()
#         if method not in ["mle", "mm", "lmoments", "optimization"]:
#             raise ValueError(
#                 method + "value should be 'mle', 'mm', 'lmoments' or 'optimization'"
#             )
#
#         if method == "mle" or method == "mm":
#             Param = list(expon.fit(self.data, method=method))
#         elif method == "lmoments":
#             LM = Lmoments(self.data)
#             LMU = LM.Lmom()
#             Param = Lmoments.gev(LMU)
#         elif method == "optimization":
#             if obj_func is None or threshold is None:
#                 raise TypeError("obj_func and threshold should be numeric value")
#
#             Param = expon.fit(self.data, method="mle")
#             # then we use the result as starting value for your truncated Gumbel fit
#             Param = so.fmin(
#                 obj_func,
#                 [threshold, Param[0], Param[1]],
#                 args=(self.data,),
#                 maxiter=500,
#                 maxfun=500,
#             )
#             Param = [Param[1], Param[2]]
#
#         self.loc = Param[0]
#         self.scale = Param[1]
#
#         if test:
#             self.ks()
#             try:
#                 self.chisquare()
#             except ValueError:
#                 print("chisquare test failed")
#
#         return Param
#
#     @staticmethod
#     def theporeticalEstimate(
#         loc: Union[float, int],
#         scale: Union[float, int],
#         prob_non_exceed: np.ndarray,
#     ) -> np.ndarray:
#         """TheporeticalEstimate.
#
#         TheporeticalEstimate method calculates the theoretical values based on a given  non exceedence probability
#
#         Parameters:
#         -----------
#         param : [list]
#             location ans scale parameters of the gumbel distribution.
#         prob_non_exceed : [list]
#             cummulative distribution function/ Non Exceedence probability.
#
#         Return:
#         -------
#         theoreticalvalue : [numeric]
#             Value based on the theoretical distribution
#         """
#         if scale <= 0:
#             raise ValueError("Parameters Invalid")
#
#         if any(prob_non_exceed) < 0 or any(prob_non_exceed) > 1:
#             raise ValueError("cdf Value Invalid")
#
#         # the main equation from scipy
#         q_th = expon.ppf(prob_non_exceed, loc=loc, scale=scale)
#         return q_th


class Exponential(AbstractDistribution):
    """
    f(x: threshold, scale) = (1/scale) e **(- (x-threshold)/scale)
    """

    def __init__(
        self,
        data: Union[list, np.ndarray] = None,
        parameters: Dict[str, str] = None,
    ):
        """Exponential Distribution.

        Parameters
        ----------
        data : [list]
            data time series.
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the exponential distribution.
            - scale: [numeric]
                scale parameter of the exponential distribution.
        """
        super().__init__(data, parameters)

    @staticmethod
    def _pdf_eq(
        data: Union[list, np.ndarray], parameters: Dict[str, Union[float, Any]]
    ) -> np.ndarray:
        loc = parameters.get("loc")
        scale = parameters.get("scale")

        if scale <= 0:
            raise ValueError("Scale parameter is negative")

        # pdf = []
        #
        # for i in ts:
        #     Y = (i - loc) / scale
        #     if Y <= 0:
        #         pdf.append(0)
        #     else:
        #         pdf.append(np.exp(-Y) / scale)
        #
        # if len(pdf) == 1:
        #     pdf = pdf[0]

        pdf = expon.pdf(data, loc=loc, scale=scale)
        return pdf

    def pdf(
        self,
        parameters: Dict[str, Union[float, Any]],
        plot_figure: bool = False,
        actual_data: np.ndarray = None,
        *args,
        **kwargs,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """pdf.

        Returns the value of Gumbel's pdf with parameters loc and scale at x .

        Parameters:
        -----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        actual_data : [bool/array]
            true if you want to calculate the pdf for the actual time series, array
            if you want to calculate the pdf for a theoretical time series
        plot_figure: [bool]
            Default is False.
        kwargs:
            figsize: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "pdf".
            fontsize: [int]
                Default is 15

        Returns
        -------
        pdf : [array]
            probability density function pdf.
        """
        result = super().pdf(
            parameters,
            actual_data=actual_data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )

        return result

    @staticmethod
    def _cdf_eq(
        data: Union[list, np.ndarray], parameters: Dict[str, Union[float, Any]]
    ) -> np.ndarray:
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale <= 0:
            raise ValueError("Scale parameter is negative")
        if loc <= 0:
            raise ValueError("Threshold parameter should be greater than zero")
        # Y = (ts - loc) / scale
        # cdf = 1 - np.exp(-Y)
        #
        # for i in range(0, len(cdf)):
        #     if cdf[i] < 0:
        #         cdf[i] = 0
        cdf = expon.cdf(data, loc=loc, scale=scale)
        return cdf

    def cdf(
        self,
        parameters: Dict[str, Union[float, Any]],
        plot_figure: bool = False,
        actual_data: Union[bool, np.ndarray] = True,
        *args,
        **kwargs,
    ) -> Union[
        Tuple[np.ndarray, Figure, Any], np.ndarray
    ]:  # pylint: disable=arguments-differ
        """cdf.

        cdf calculates the value of Gumbel's cdf with parameters loc and scale at x.

        parameter:
        ----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        actual_data : [bool/array]
            true if you want to calculate the pdf for the actual time series, array
            if you want to calculate the pdf for a theoretical time series
        plot_figure: [bool]
            Default is False.
        kwargs:
            figsize: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "cdf".
            fontsize: [int]
                Default is 15.
        """
        result = super().cdf(
            parameters,
            actual_data=actual_data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )
        return result

    def fit_model(
        self,
        method: str = "mle",
        obj_func=None,
        threshold: Union[int, float, None] = None,
        test: bool = True,
    ) -> Dict[str, float]:
        """estimateParameter.

        EstimateParameter estimate the distribution parameter based on MLM
        (Maximum liklihood method), if an objective function is entered as an input

        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for values below (x < C), now the likeliest parameters
        are those at the max value of mutiplication between two functions max(L1*L2).

        In this case the L1 is still the product of multiplication of probability
        density function's values at xi, but the L2 is the probability that threshold
        value C will be exceeded (1-F(C)).

        Parameters
        ----------
        obj_func : [function]
            function to be used to get the distribution parameters.
        threshold : [numeric]
            Value you want to consider only the greater values.
        method : [string]
            'mle', 'mm', 'lmoments', optimization
        test: bool
            Default is True

        Returns
        -------
        param : [list]
            shape, loc, scale parameter of the gumbel distribution in that order.
        """
        # obj_func = lambda p, x: (-np.log(Gumbel.pdf(x, p[0], p[1]))).sum()
        # #first we make a simple Gumbel fit
        # Par1 = so.fmin(obj_func, [0.5,0.5], args=(np.array(data),))
        method = super().fit_model(method=method)

        if method == "mle" or method == "mm":
            param = list(expon.fit(self.data, method=method))
        elif method == "lmoments":
            lm = Lmoments(self.data)
            lmu = lm.Lmom()
            param = Lmoments.exponential(lmu)
        elif method == "optimization":
            if obj_func is None or threshold is None:
                raise TypeError("obj_func and threshold should be numeric value")

            param = expon.fit(self.data, method="mle")
            # then we use the result as starting value for your truncated Gumbel fit
            param = so.fmin(
                obj_func,
                [threshold, param[0], param[1]],
                args=(self.data,),
                maxiter=500,
                maxfun=500,
            )
            param = [param[1], param[2]]
        else:
            raise ValueError(f"The given: {method} does not exist")

        param = {"loc": param[0], "scale": param[1]}
        self.parameters = param

        if test:
            self.ks()
            try:
                self.chisquare()
            except ValueError:
                print("chisquare test failed")

        return param

    @staticmethod
    def theoretical_estimate(
        parameters: Dict[str, Union[float, Any]],
        cdf: np.ndarray,
    ) -> np.ndarray:
        """TheporeticalEstimate.

        TheporeticalEstimate method calculates the theoretical values based on a given  non exceedence probability

        Parameters:
        -----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        cdf: [list]
            cummulative distribution function/ Non Exceedence probability.

        Return:
        -------
        theoreticalvalue : [numeric]
            Value based on the theoretical distribution
        """
        loc = parameters.get("loc")
        scale = parameters.get("scale")

        if scale <= 0:
            raise ValueError("Parameters Invalid")

        if any(cdf) < 0 or any(cdf) > 1:
            raise ValueError("cdf Value Invalid")

        # the main equation from scipy
        q_th = expon.ppf(cdf, loc=loc, scale=scale)
        return q_th

    def ks(self):
        """Kolmogorov-Smirnov (KS) test.

        The smaller the D static the more likely that the two samples are drawn from the same distribution
        IF Pvalue < signeficance level ------ reject

        returns:
        --------
            Dstatic: [numeric]
                The smaller the D static the more likely that the two samples are drawn from the same distribution
            Pvalue : [numeric]
                IF Pvalue < signeficance level ------ reject the null hypotethis
        """
        return super().ks()

    def chisquare(self) -> tuple:
        """chisquare test"""
        return super().chisquare()


class Normal(AbstractDistribution):
    """
    f(x: threshold, scale) = (1/scale) e **(- (x-threshold)/scale)
    """

    def __init__(
        self,
        data: Union[list, np.ndarray] = None,
        parameters: Dict[str, str] = None,
    ):
        """Gumbel.

        Parameters
        ----------
        data : [list]
            data time series.
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the exponential distribution.
            - scale: [numeric]
                scale parameter of the exponential distribution.
        """
        super().__init__(data, parameters)

    @staticmethod
    def _pdf_eq(
        data: Union[list, np.ndarray], parameters: Dict[str, Union[float, Any]]
    ) -> np.ndarray:
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale <= 0:
            raise ValueError("Scale parameter is negative")
        pdf = norm.pdf(data, loc=loc, scale=scale)

        return pdf

    def pdf(
        self,
        parameters: Dict[str, Union[float, Any]],
        plot_figure: bool = False,
        actual_data: np.ndarray = None,
        *args,
        **kwargs,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """pdf.

        Returns the value of Gumbel's pdf with parameters loc and scale at x .

        Parameters:
        -----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val, "shape": value}
            - loc: [numeric]
                location parameter of the GEV distribution.
            - scale: [numeric]
                scale parameter of the GEV distribution.
        actual_data : [bool/array]
            true if you want to calculate the pdf for the actual time series, array
            if you want to calculate the pdf for a theoretical time series
        plot_figure: [bool]
            Default is False.
        kwargs:
            figsize: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "pdf".
            fontsize: [int]
                Default is 15

        Returns
        -------
        pdf : [array]
            probability density function pdf.
        """
        result = super().pdf(
            parameters,
            actual_data=actual_data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )

        return result

    @staticmethod
    def _cdf_eq(
        data: Union[list, np.ndarray], parameters: Dict[str, Union[float, Any]]
    ) -> np.ndarray:
        loc = parameters.get("loc")
        scale = parameters.get("scale")

        if scale <= 0:
            raise ValueError("Scale parameter is negative")
        if loc <= 0:
            raise ValueError("Threshold parameter should be greater than zero")

        cdf = norm.cdf(data, loc=loc, scale=scale)
        return cdf

    def cdf(
        self,
        parameters: Dict[str, Union[float, Any]],
        plot_figure: bool = False,
        actual_data: Union[bool, np.ndarray] = True,
        *args,
        **kwargs,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """cdf.

        cdf calculates the value of Normal distribution cdf with parameters loc and scale at x.

        parameter:
        ----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val, "shape": value}
            - loc: [numeric]
                location parameter of the Normal distribution.
            - scale: [numeric]
                scale parameter of the Normal distribution.
        actual_data : [bool/array]
            true if you want to calculate the pdf for the actual time series, array
            if you want to calculate the pdf for a theoretical time series
        plot_figure: [bool]
            Default is False.
        kwargs:
            figsize: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "cdf".
            fontsize: [int]
                Default is 15.
        """
        result = super().cdf(
            parameters,
            actual_data=actual_data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )
        return result

    def fit_model(
        self,
        method: str = "mle",
        obj_func=None,
        threshold: Union[int, float, None] = None,
        test: bool = True,
    ) -> Dict[str, float]:
        """estimateParameter.

        EstimateParameter estimate the distribution parameter based on MLM
        (Maximum liklihood method), if an objective function is entered as an input

        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for values below (x < C), now the likeliest parameters
        are those at the max value of mutiplication between two functions max(L1*L2).

        In this case the L1 is still the product of multiplication of probability
        density function's values at xi, but the L2 is the probability that threshold
        value C will be exceeded (1-F(C)).

        Parameters
        ----------
        obj_func : [function]
            function to be used to get the distribution parameters.
        threshold : [numeric]
            Value you want to consider only the greater values.
        method : [string]
            'mle', 'mm', 'lmoments', optimization
        test: bool
            Default is True

        Returns
        -------
        param : [list]
            shape, loc, scale parameter of the gumbel distribution in that order.
        """
        # obj_func = lambda p, x: (-np.log(Gumbel.pdf(x, p[0], p[1]))).sum()
        # #first we make a simple Gumbel fit
        # Par1 = so.fmin(obj_func, [0.5,0.5], args=(np.array(data),))
        method = super().fit_model(method=method)

        if method == "mle" or method == "mm":
            param = list(norm.fit(self.data, method=method))
        elif method == "lmoments":
            lm = Lmoments(self.data)
            lmu = lm.Lmom()
            param = Lmoments.normal(lmu)
        elif method == "optimization":
            if obj_func is None or threshold is None:
                raise TypeError("obj_func and threshold should be numeric value")

            param = norm.fit(self.data, method="mle")
            # then we use the result as starting value for your truncated Gumbel fit
            param = so.fmin(
                obj_func,
                [threshold, param[0], param[1]],
                args=(self.data,),
                maxiter=500,
                maxfun=500,
            )
            param = [param[1], param[2]]
        else:
            raise ValueError(f"The given: {method} does not exist")

        param = {"loc": param[0], "scale": param[1]}
        self.parameters = param

        if test:
            self.ks()
            try:
                self.chisquare()
            except ValueError:
                print("chisquare test failed")

        return param

    @staticmethod
    def theoretical_estimate(
        parameters: Dict[str, Union[float, Any]],
        cdf: np.ndarray,
    ) -> np.ndarray:
        """TheporeticalEstimate.

        TheporeticalEstimate method calculates the theoretical values based on a given  non exceedence probability

        Parameters:
        -----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}
            - loc: [numeric]
                location parameter of the Normal distribution.
            - scale: [numeric]
                scale parameter of the Normal distribution.
        cdf: [list]
            cummulative distribution function/ Non Exceedence probability.

        Return:
        -------
        numeric:
            Value based on the theoretical distribution
        """
        loc = parameters.get("loc")
        scale = parameters.get("scale")

        if scale <= 0:
            raise ValueError("Parameters Invalid")

        if any(cdf) < 0 or any(cdf) > 1:
            raise ValueError("cdf Value Invalid")

        # the main equation from scipy
        q_th = norm.ppf(cdf, loc=loc, scale=scale)
        return q_th

    def ks(self):
        """Kolmogorov-Smirnov (KS) test.

        The smaller the D static the more likely that the two samples are drawn from the same distribution
        IF Pvalue < signeficance level ------ reject

        returns:
        --------
            Dstatic: [numeric]
                The smaller the D static the more likely that the two samples are drawn from the same distribution
            Pvalue : [numeric]
                IF Pvalue < signeficance level ------ reject the null hypotethis
        """
        return super().ks()

    def chisquare(self) -> tuple:
        """chisquare test"""
        return super().chisquare()


class Distributions:
    """Distributions."""

    available_distributions = {
        "GEV": GEV,
        "Gumbel": Gumbel,
        "Exponential": Exponential,
        "Normal": Normal,
    }

    def __init__(
        self,
        distribution: str,
        data: Union[list, np.ndarray] = None,
        parameters: Dict[str, Number] = None,
    ):
        if distribution not in self.available_distributions.keys():
            raise ValueError(f"{distribution} not supported")

        self.distribution = self.available_distributions[distribution](data, parameters)

    def __getattr__(self, name: str):
        """Delegate method calls to the sub-class"""
        # Retrieve the attribute or method from the animal object
        try:
            # Retrieve the attribute or method from the sub-classes
            attribute = getattr(self.distribution, name)

            # If the attribute is a method, return a callable function
            if callable(attribute):

                def method(*args, **kwargs):
                    """A callable function that simply calls the attribute if it is a method"""
                    return attribute(*args, **kwargs)

                return method

            # If it's a regular attribute, return its value
            return attribute

        except AttributeError:
            raise AttributeError(
                f"'{type(self).__name__}' object has no attribute '{name}'"
            )
