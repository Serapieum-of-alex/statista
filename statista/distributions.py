"""Statistical distributions."""

from numbers import Number
from typing import Any, List, Tuple, Union, Dict, Callable
from abc import ABC, abstractmethod
import numpy as np
from statistics import mode
import scipy.optimize as so
from matplotlib.figure import Figure
from matplotlib.axes import Axes

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
            non-exceedance probability.

        Returns
        -------
        array:
           return period.

        Examples
        --------
        - First generate some random numbers between 0 and 1 as a non-exceedance probability. then use this non-exceedance
            to calculate the return period.

            >>> data = np.random.random(15)
            >>> rp = PlottingPosition.return_period(data)
            >>> print(rp) # doctest: +SKIP
            [ 1.33088992  4.75342173  2.46855419  1.42836548  2.75320582  2.2268505
              8.06500888 10.56043917 18.28884687  1.10298241  1.2113997   1.40988022
              1.02795867  1.01326322  1.05572108]
        """
        if any(prob_non_exceed > 1):
            raise ValueError("Non-exceedance probability should be less than 1")
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

        Examples
        --------
        >>> data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        >>> cdf = PlottingPosition.weibul(data)
        >>> print(cdf)
        [0.09090909 0.18181818 0.27272727 0.36363636 0.45454545 0.54545455
         0.63636364 0.72727273 0.81818182 0.90909091]
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

    def __init__(
        self,
        data: Union[list, np.ndarray] = None,
        parameters: Dict[str, float] = None,
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
        if data is None and parameters is None:
            raise ValueError("Either data or parameters must be provided")

        if isinstance(data, list) or isinstance(data, np.ndarray):
            self._data = np.array(data)
        elif data is None:
            self._data = data
        else:
            raise TypeError("The `data` argument should be list or numpy array")

        if isinstance(parameters, dict) or parameters is None:
            self._parameters = parameters
        else:
            raise TypeError("The `parameters` argument should be dictionary")

    def __str__(self) -> str:
        message = ""
        if self.data is not None:
            message += f"""
                    Dataset of {len(self.data)} value
                    min: {np.min(self.data)}
                    max: {np.max(self.data)}
                    mean: {np.mean(self.data)}
                    median: {np.median(self.data)}
                    mode: {mode(self.data)}
                    std: {np.std(self.data)}
                    Distribution : {self.__class__.__name__}
                    parameters: {self.parameters}
                    """
        if self.parameters is not None:
            message += f"""
                Distribution : {self.__class__.__name__}
                parameters: {self.parameters}
                """
        return message

    @property
    def parameters(self) -> Dict[str, float]:
        """Distribution parameters"""
        return self._parameters

    @parameters.setter
    def parameters(self, value: Dict[str, float]):
        self._parameters = value

    @property
    def data(self) -> ndarray:
        """data."""
        return self._data

    @property
    def data_sorted(self) -> ndarray:
        """data_sorted."""
        return np.sort(self.data)

    @property
    def kstable(self) -> float:
        """KStable."""
        return 1.22 / np.sqrt(len(self.data))

    @property
    def cdf_weibul(self) -> ndarray:
        """cdf_Weibul."""
        return PlottingPosition.weibul(self.data)

    @staticmethod
    @abstractmethod
    def _pdf_eq(
        data: Union[list, np.ndarray], parameters: Dict[str, Union[float, Any]]
    ) -> np.ndarray:
        pass

    @abstractmethod
    def pdf(
        self,
        parameters: Dict[str, Union[float, Any]] = None,
        plot_figure: bool = False,
        fig_size: tuple = (6, 5),
        xlabel: str = "Actual data",
        ylabel: str = "pdf",
        fontsize: Union[float, int] = 15,
        data: Union[List[float], np.ndarray] = None,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Figure, Axes]]:
        """pdf.

        Returns the value of Gumbel's pdf with parameters loc and scale at x.

        Parameters
        ----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        data : np.ndarray, default is None.
            array if you want to calculate the pdf for different data than the time series given to the constructor
            method.
        plot_figure: [bool], Default is False.
            True to plot the figure.
        fig_size: [tuple]
                Default is (6, 5).
        xlabel: [str]
            Default is "Actual data".
        ylabel: [str]
            Default is "cdf".
        fontsize: [int]
            Default is 15.

        Returns
        -------
        pdf: [array]
            probability density function pdf.
        fig: matplotlib.figure.Figure, if `plot_figure` is True.
            Figure object.
        ax: matplotlib.axes.Axes, if `plot_figure` is True.
            Axes object.
        """

        if data is None:
            ts = self.data
            data_sorted = self.data_sorted
        else:
            ts = data
            data_sorted = np.sort(data)

        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        pdf = self._pdf_eq(ts, parameters)

        if plot_figure:
            qx = np.linspace(float(data_sorted[0]), 1.5 * float(data_sorted[-1]), 10000)
            pdf_fitted = self.pdf(parameters=parameters, data=qx)

            fig, ax = Plot.pdf(
                qx,
                pdf_fitted,
                data_sorted,
                fig_size=fig_size,
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
        parameters: Dict[str, Union[float, Any]] = None,
        plot_figure: bool = False,
        fig_size: tuple = (6, 5),
        xlabel: str = "data",
        ylabel: str = "cdf",
        fontsize: int = 15,
        data: Union[List[float], np.ndarray] = None,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Figure, Axes]]:
        """Cumulative distribution function.

        Parameters
        ----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        data : np.ndarray, default is None.
            array if you want to calculate the cdf for different data than the time series given to the constructor
            method.
        plot_figure: [bool], Default is False.
            True to plot the figure.
        fig_size: [tuple]
                Default is (6, 5).
        xlabel: [str]
            Default is "Actual data".
        ylabel: [str]
            Default is "cdf".
        fontsize: [int]
            Default is 15.
        """
        if data is None:
            ts = self.data
            data_sorted = self.data_sorted
        else:
            ts = data
            data_sorted = np.sort(data)

        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        cdf = self._cdf_eq(ts, parameters)

        if plot_figure:
            qx = np.linspace(float(data_sorted[0]), 1.5 * float(data_sorted[-1]), 10000)
            cdf_fitted = self.cdf(parameters=parameters, data=qx)

            cdf_weibul = PlottingPosition.weibul(data_sorted)

            fig, ax = Plot.cdf(
                qx,
                cdf_fitted,
                data_sorted,
                cdf_weibul,
                fig_size=fig_size,
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
        """fit_model.

        fit_model estimates the distribution parameter based on MLM
        (Maximum likelihood method), if an objective function is entered as an input

        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for the values below (x < C), now the likeliest parameters
        are those at the max value of multiplication between two functions max(L1*L2).

        In this case, the L1 is still the product of multiplication of probability
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
            loc: [numeric]
                location parameter of the gumbel distribution.
            scale: [numeric]
                scale parameter of the gumbel distribution.
        """
        method = method.lower()
        if method not in ["mle", "mm", "lmoments", "optimization"]:
            raise ValueError(
                f"{method} value should be 'mle', 'mm', 'lmoments' or 'optimization'"
            )
        return method

    @abstractmethod
    def inverse_cdf(
        self,
        cdf: Union[np.ndarray, List[float]],
        parameters: Dict[str, Union[float, Any]],
    ) -> np.ndarray:
        """theoretical Estimate.

        Theoretical Estimate method calculates the theoretical values based on the Gumbel distribution

        Parameters
        ----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        cdf: [list]
            cumulative distribution function/ Non-Exceedance probability.

        Returns
        -------
        theoretical value: [numeric]
            Value based on the theoretical distribution
        """
        pass

    @abstractmethod
    def ks(self) -> tuple:
        """Kolmogorov-Smirnov (KS) test.

        The smaller the D static, the more likely that the two samples are drawn from the same distribution
        IF Pvalue < significance level ------ reject

        returns
        -------
        Dstatic: [numeric]
            The smaller the D static the more likely that the two samples are drawn from the same distribution
        Pvalue : [numeric]
            IF Pvalue < significance level ------ reject the null hypothesis.
        """
        if self.parameters is None:
            raise ValueError(
                "The Value of parameters is unknown. Please use 'fit_model' to estimate the distribution parameters"
            )
        qth = self.inverse_cdf(self.cdf_weibul, self.parameters)

        test = ks_2samp(self.data, qth)

        print("-----KS Test--------")
        print(f"Statistic = {test.statistic}")
        if test.statistic < self.kstable:
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
                "The Value of parameters is unknown. Please use 'fit_model' to estimate the distribution parameters"
            )

        qth = self.inverse_cdf(self.cdf_weibul, self.parameters)
        try:
            test = chisquare(st.standardize(qth), st.standardize(self.data))
            print("-----chisquare Test-----")
            print("Statistic = " + str(test.statistic))
            print("P value = " + str(test.pvalue))
            return test.statistic, test.pvalue
        except Exception as e:
            print(e)

    def confidence_interval(
        self,
        alpha: float = 0.1,
        plot_figure: bool = False,
        prob_non_exceed: np.ndarray = None,
        parameters: Dict[str, Union[float, Any]] = None,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Figure, Axes]
    ]:
        """confidence_interval.

        Parameters
        ----------
        alpha: numeric, default is 0.1
            alpha or Significance level is a value of the confidence interval.
        plot_figure: bool, optional, default is False.
            to plot the confidence interval.
        parameters: Dict[str, str], optional, default is None.
            if not provided, the parameters provided in the class initialization will be used.
            {"loc": val, "scale": val}

            - loc: numeric
                location parameter of the gumbel distribution.
            - scale: numeric
                scale parameter of the gumbel distribution.
        prob_non_exceed: list, default is None.
            Non-Exceedance probability, if not given, the plotting position will be calculated using the weibul method.
        kwargs:
            fig_size: Tuple[float, float], optional, default=(6, 6)
                Size of the second figure.
            fontsize: int, optional, default=11
                Font size.

        Returns
        -------
        q_upper: [list]
            upper-bound coresponding to the confidence interval.
        q_lower: [list]
            lower bound coresponding to the confidence interval.
        fig: matplotlib.figure.Figure
            Figure object.
        ax: matplotlib.axes.Axes
            Axes object.
        """
        pass

    def plot(
        self,
        fig_size: tuple = (10, 5),
        xlabel: str = "Actual data",
        ylabel: str = "cdf",
        fontsize: int = 15,
        cdf: np.ndarray = None,
        parameters: Dict[str, Union[float, Any]] = None,
    ) -> Tuple[List[Figure], list]:
        """Probability Plot.

        Probability Plot method calculates the theoretical values based on the Gumbel distribution
        parameters, theoretical cdf (or weibul), and calculates the confidence interval.

        Parameters
        ----------
        fig_size: tuple, Default is (10, 5).
            Size of the figure.
        xlabel: [str]
            Default is "Actual data"
        ylabel: [str]
            Default is "cdf"
        fontsize: [float]
            Default is 15.
        parameters: Dict[str, str]
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        cdf: [np.ndarray]
            theoretical cdf calculated using weibul or using the distribution cdf function.

        Returns
        -------
        Figure:
            matplotlib figure object
        Tuple[Axes, Axes]:
            matplotlib plot axes
        """
        pass


class Gumbel(AbstractDistribution):
    """Gumbel distribution (Maximum - Right Skewed).

    The Gumbel distribution is used to model the distribution of the maximum (or the minimum) of a number of samples of
    various distributions.

    - The probability density function (PDF) of the Gumbel distribution (Type I) is:

        .. math::
            f(x; \\zeta, \\delta) = \\frac{1}{\\delta} \\exp\\left(-\\frac{x - \\zeta}{\\delta} \\right)
            \\exp\\left(-\\exp\\left(-\\frac{x - \\zeta}{\\delta} \\right) \\right)
          :label: gumbel-pdf

        where :math:`\\zeta` (zeta) is the location parameter, and :math:`\\delta` (delta) is the scale parameter.

    - The location parameter :math:`\\zeta` shifts the distribution along the x-axis. It essentially determines the mode
        (peak) of the distribution and its location. Changing the location parameter moves the distribution left or
        right without altering its shape. The location parameter ranges from negative infinity to positive infinity.
    - The scale parameter :math:`\\delta` controls the spread or dispersion of the distribution. A larger scale parameter
        results in a wider distribution, while a smaller scale parameter results in a narrower distribution. It must
        always be positive.

    - The probability density function above is defined in the “un-standardized” form.

    The Gumbel distribution is a special case of the Generalized Extreme Value (GEV) distribution for a particular
    choice of the shape parameter, :math:`\\xi = 0` (xi).

    - The cumulative distribution functions.

        .. math::
            F(x; \\zeta, \\delta) = \\exp\\left(-\\exp\\left(-\\frac{x - \\zeta}{\\delta} \\right) \\right)
          :label: gumbel-cdf

    """

    def __init__(
        self,
        data: Union[list, np.ndarray] = None,
        parameters: Dict[str, float] = None,
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

        Examples
        --------
        - First load a sample data.

            >>> data = np.loadtxt("examples/data/gumbel.txt")

        - I nstantiate the Gumbel class only with the data.

            >>> gumbel_dist = Gumbel(data)
            >>> print(gumbel_dist) # doctest: +SKIP
            <statista.distributions.Gumbel object at 0x000001CDDE9563F0>

        - You can also instantiate the Gumbel class with the data and the parameters if you already have them.

            >>> parameters = {"loc": 0, "scale": 1}
            >>> gumbel_dist = Gumbel(data, parameters)
            >>> print(gumbel_dist) # doctest: +SKIP
            <statista.distributions.Gumbel object at 0x000001CDDEB32C00>
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
        plot_figure: bool = False,
        parameters: Dict[str, Union[float, Any]] = None,
        data: Union[List[float], np.ndarray] = None,
        *args,
        **kwargs,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Figure, Any]]:
        """pdf.

        Returns the value of Gumbel's pdf with parameters loc and scale at x.

        Parameters
        ----------
        parameters: Dict[str, str], optional, default is None.
            if not provided, the parameters provided in the class initialization will be used.
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        data : np.ndarray, default is None.
            array if you want to calculate the pdf for different data than the time series given to the constructor
            method.
        plot_figure: [bool]
            Default is False.
        kwargs:
            fig_size: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "pdf".
            fontsize: [int]
                Default is 15.

        Returns
        -------
        pdf: [np.ndarray]
            probability density function pdf.
        fig: matplotlib.figure.Figure, if `plot_figure` is True.
            Figure object.
        ax: matplotlib.axes.Axes, if `plot_figure` is True.
            Axes object.

        Examples
        --------
        >>> data = np.loadtxt("examples/data/gumbel.txt")
        >>> parameters = {'loc': 0, 'scale': 1}
        >>> gumbel_dist = Gumbel(data, parameters)
        >>> gumbel_dist.pdf(plot_figure=True)

        .. image:: /_images/gumbel-random-pdf.png
            :align: center
        """
        result = super().pdf(
            parameters=parameters,
            data=data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )
        return result

    def random(
        self,
        size: int,
        parameters: Dict[str, Union[float, Any]] = None,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """Generate Random Variable.

        Parameters
        ----------
        size: int
            size of the random generated sample.
        parameters: Dict[str, str]
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.

        Returns
        -------
        data: [np.ndarray]
            random generated data.

        Examples
        --------
        - To generate a random sample that follow the gumbel distribution with the parameters loc=0 and scale=1.

            >>> parameters = {'loc': 0, 'scale': 1}
            >>> gumbel_dist = Gumbel(parameters=parameters)
            >>> random_data = gumbel_dist.random(1000)

        - then we can use the `pdf` method to plot the pdf of the random data.

            >>> gumbel_dist.pdf(data=random_data, plot_figure=True, xlabel="Random data")

            .. image:: /_images/gumbel-random-pdf.png
                :align: center

            >>> gumbel_dist.cdf(data=random_data, plot_figure=True, xlabel="Random data")

            .. image:: /_images/gumbel-random-cdf.png
                :align: center
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale <= 0:
            raise ValueError("Scale parameter is negative")

        random_data = gumbel_r.rvs(loc=loc, scale=scale, size=size)
        return random_data

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
        plot_figure: bool = False,
        parameters: Dict[str, Union[float, Any]] = None,
        data: Union[List[float], np.ndarray] = None,
        *args,
        **kwargs,
    ) -> Union[
        np.ndarray, Tuple[np.ndarray, Figure, Axes]
    ]:  # pylint: disable=arguments-differ
        """Cumulative distribution function.

        parameter
        ---------
        parameters: Dict[str, str], optional, default is None.
            if not provided, the parameters provided in the class initialization will be used.
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        data : np.ndarray, default is None.
            array if you want to calculate the cdf for different data than the time series given to the constructor
            method.
        plot_figure: [bool], Default is False.
            True to plot the figure.
        kwargs:
            fig_size: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "cdf".
            fontsize: [int]
                Default is 15.

        Returns
        -------
        cdf: [array]
            cumulative distribution function cdf.
        fig: matplotlib.figure.Figure, if `plot_figure` is True.
            Figure object.
        ax: matplotlib.axes.Axes, if `plot_figure` is True.
            Axes object.

        Examples
        --------
        >>> data = np.loadtxt("examples/data/gumbel.txt")
        >>> parameters = {'loc': 0, 'scale': 1}
        >>> gumbel_dist = Gumbel(data, parameters)
        >>> gumbel_dist.cdf(plot_figure=True)  # doctest: +SKIP

        .. image:: /_images/gumbel-random-cdf.png
            :align: center
        """
        result = super().cdf(
            parameters=parameters,
            data=data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )
        return result

    def return_period(
        self,
        data: Union[bool, List[float]] = None,
        parameters: Dict[str, Union[float, Any]] = None,
    ):
        """Calculate return period.

            return_period calculates the return period for a list/array of values or a single value.

        Parameters
        ----------
        data:[list/array/float]
            value you want the corresponding return value for
        parameters: Dict[str, str]
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.

        Returns
        -------
        float:
            return period
        """
        if data is None:
            ts = self.data
        else:
            ts = data

        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        cdf: np.ndarray = self.cdf(parameters, data=ts)

        rp = 1 / (1 - cdf)

        return rp

    @staticmethod
    def truncated_distribution(opt_parameters: list[float], data: list[float]):
        """function to estimate the parameters of a truncated Gumbel distribution.

        Link :
        https://stackoverflow.com/questions/23217484/how-to-find-parameters-of-gumbels-distribution-using-scipy-optimize

        function calculates the negative log-likelihood of a Gumbel distribution that is truncated(i.e., the data only
        includes values above a certain threshold)

        the function calculates the negative log-likelihood, effectively fitting the truncated Gumbel distribution
        to the data.

        This approach is useful when the dataset is incomplete or when data is only available above a certain threshold,
        a common scenario in environmental sciences, finance, and other fields dealing with extremes.

        Parameters
        ----------
        opt_parameters:
        data: list
            data
        """
        threshold = opt_parameters[0]
        loc = opt_parameters[1]
        scale = opt_parameters[2]

        non_truncated_data = data[data < threshold]
        nx2 = len(data[data >= threshold])
        # pdf with a scaled pdf
        # L1 is pdf based
        parameters = {"loc": loc, "scale": scale}
        pdf = Gumbel._pdf_eq(non_truncated_data, parameters)
        #  the CDF at the threshold is used because the data is assumed to be truncated, meaning that observations below
        #  this threshold are not included in the dataset. When dealing with truncated data, it's essential to adjust
        #  the likelihood calculation to account for the fact that only values above the threshold are observed. The
        #  CDF at the threshold effectively normalizes the distribution, ensuring that the probabilities sum to 1 over
        #  the range of the observed data.
        cdf_at_threshold = 1 - Gumbel._cdf_eq(threshold, parameters)
        # calculates the negative log-likelihood of a Gumbel distribution
        # Adjust the likelihood for the truncation
        # likelihood = pdf / (1 - adjusted_cdf)

        l1 = (-np.log((pdf / scale))).sum()
        # L2 is cdf based
        l2 = (-np.log(cdf_at_threshold)) * nx2
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

        fit_model estimates the distribution parameter based on MLM
        (Maximum likelihood method), if an objective function is entered as an input

        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for the values below (x < C), now the likeliest parameters
        are those at the max value of multiplication between two functions max(L1*L2).

        In this case, the L1 is still the product of multiplication of probability
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

        Examples
        --------
        - Instantiate the Gumbel class only with the data.

            >>> data = np.loadtxt("examples/data/gumbel.txt")
            >>> gumbel_dist = Gumbel(data)

        - Then use the `fit_model` method to estimate the distribution parameters. the method takes the method as
            parameter, the default is 'mle'. the `test` parameter is used to perform the Kolmogorov-Smirnov and chisquare
            test.

            >>> parameters = gumbel_dist.fit_model(method="mle", test=True)
            -----KS Test--------
            Statistic = 0.019
            Accept Hypothesis
            P value = 0.9937026761524456
            >>> print(parameters)
            {'loc': 0.010101355750222706, 'scale': 1.0313042643102108}

        - You can also use the `lmoments` method to estimate the distribution parameters.

            >>> parameters = gumbel_dist.fit_model(method="lmoments", test=True)
            -----KS Test--------
            Statistic = 0.019
            Accept Hypothesis
            P value = 0.9937026761524456
            >>> print(parameters)
            {'loc': 0.006700226367219564, 'scale': 1.0531061622114444}

        - You can also use the `fit_model` method to estimate the distribution parameters using the 'optimization'
            method. the optimization method requires the `obj_func` and `threshold` parameter. the method
            will take the `threshold` number and try to fit the data values that are greater than the threshold.
            >>> threshold = np.quantile(data, 0.80)
            >>> print(threshold)
            1.5717000000000005
            >>> parameters = gumbel_dist.fit_model(method="optimization", obj_func=Gumbel.truncated_distribution, threshold=threshold)
            Optimization terminated successfully.
                     Current function value: 0.000000
                     Iterations: 39
                     Function evaluations: 116
            -----KS Test--------
            Statistic = 0.107
            reject Hypothesis
            P value = 2.0977827855404345e-05

            - As you see, the P value is less than the significance level, so we reject the null hypothesis,
            but we are trying to fit the distribution to part of the data, not the whole data.
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
            # self.chisquare()

        return param

    def inverse_cdf(
        self,
        cdf: Union[np.ndarray, List[float]] = None,
        parameters: Dict[str, float] = None,
    ) -> np.ndarray:
        """inverse CDF.

        inverse_cdf method calculates the theoretical values based on a given cumulative distribution function.

        Parameters
        ----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        cdf: [list]
            cumulative distribution function/ Non Exceedance probability.

        Returns
        -------
        theoretical value: [numeric]
            Value based on the theoretical distribution

        Examples
        --------
        - Instantiate the Gumbel class only with the data.

            >>> data = np.loadtxt("examples/data/gumbel.txt")
            >>> parameters = {'loc': 0, 'scale': 1}
            >>> gumbel_dist = Gumbel(data, parameters)

        - We will generate a random numbers between 0 and 1 and pass it to the inverse_cdf method as a probabilities
            to get the data that coresponds to these probabilities based on the distribution.

            >>> cdf = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            >>> data_values = gumbel_dist.inverse_cdf(cdf)
            >>> print(data_values)
            [-0.83403245 -0.475885    0.08742157  0.67172699  1.49993999  2.25036733]
        """
        if parameters is None:
            parameters = self.parameters

        if any(cdf) <= 0 or any(cdf) > 1:
            raise ValueError("cdf Value Invalid")

        cdf = np.array(cdf)
        qth = self._inv_cdf(cdf, parameters)

        return qth

    @staticmethod
    def _inv_cdf(cdf: Union[np.ndarray, List[float]], parameters: Dict[str, float]):
        # the main equation from scipy
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale <= 0:
            raise ValueError("Scale parameter is negative")
        # the main equation from scipy
        # Qth = loc - scale * (np.log(-np.log(cdf)))
        qth = gumbel_r.ppf(cdf, loc=loc, scale=scale)

        return qth

    def ks(self) -> tuple:
        """Kolmogorov-Smirnov (KS) test.

        The smaller the D static, the more likely that the two samples are drawn from the same distribution
        IF P value < significance level ------ reject

        Returns
        -------
        Dstatic: [numeric]
            The smaller the D static the more likely that the two samples are drawn from the same distribution.
            - The KS test statistic measures the maximum distance between the empirical cumulative distribution function
                (ECDF) of the sample (like Weibul plotting position) and the cumulative distribution function (CDF) of
                the reference distribution.
            - A smaller KS statistic indicates a smaller difference between the sample distribution and the reference
                distribution.
        P value: [numeric]
            A high p-value (close to 1) suggests that there is a high probability that the sample comes from the
            specified distribution. IF P value < significance level ------ reject the null hypothesis
        """
        return super().ks()

    def chisquare(self) -> tuple:
        """chisquare test"""
        return super().chisquare()

    def confidence_interval(
        self,
        alpha: float = 0.1,
        prob_non_exceed: np.ndarray = None,
        parameters: Dict[str, Union[float, Any]] = None,
        plot_figure: bool = False,
        **kwargs,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Figure, Axes]
    ]:
        """confidence_interval.

        Parameters
        ----------
        alpha: numeric, default is 0.1
            alpha or Significance level is a value of the confidence interval.
        plot_figure: bool, optional, default is False.
            to plot the confidence interval.
        parameters: Dict[str, str], optional, default is None.
            if not provided, the parameters provided in the class initialization will be used.
            {"loc": val, "scale": val}

            - loc: numeric
                location parameter of the gumbel distribution.
            - scale: numeric
                scale parameter of the gumbel distribution.
        prob_non_exceed: list, default is None.
            Non-Exceedance probability, if not given, the plotting position will be calculated using the weibul method.
        kwargs:
            fig_size: Tuple[float, float], optional, default=(6, 6)
                Size of the second figure.
            fontsize: int, optional, default=11
                Font size.

        Returns
        -------
        q_upper : [list]
            upper bound corresponding to the confidence interval.
        q_lower : [list]
            lower bound corresponding to the confidence interval.
        fig: matplotlib.figure.Figure
            Figure object.
        ax: matplotlib.axes.Axes
            Axes object.

        Examples
        --------
        - Instantiate the Gumbel class with the data and the parameters.

            >>> import matplotlib.pyplot as plt
            >>> data = np.loadtxt("examples/data/time_series2.txt")
            >>> parameters = {"loc": 463.8040, "scale": 220.0724}
            >>> gumbel_dist = Gumbel(data, parameters)

        - to calculate the confidence interval, we need to provide the confidence level (`alpha`).

            >>> upper, lower = gumbel_dist.confidence_interval(alpha=0.1)

        - You can also plot confidence intervals

            >>> upper, lower, fig, ax = gumbel_dist.confidence_interval(alpha=0.1, plot_figure=True, marker_size=10)

        .. image:: /_images/gumbel-confidence-interval.png
            :align: center
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        scale = parameters.get("scale")
        if scale <= 0:
            raise ValueError("Scale parameter is negative")

        if prob_non_exceed is None:
            prob_non_exceed = PlottingPosition.weibul(self.data)
        else:
            # if the prob_non_exceed is given, check if the length is the same as the data
            if len(prob_non_exceed) != len(self.data):
                raise ValueError(
                    "Length of prob_non_exceed does not match the length of data, use the `PlottingPosition.weibul(data)` "
                    "to the get the non-exceedance probability"
                )

        qth = self._inv_cdf(prob_non_exceed, parameters)
        y = [-np.log(-np.log(j)) for j in prob_non_exceed]
        std_error = [
            (scale / np.sqrt(len(self.data)))
            * np.sqrt(1.1087 + 0.5140 * j + 0.6079 * j**2)
            for j in y
        ]
        v = norm.ppf(1 - alpha / 2)
        q_upper = np.array([qth[j] + v * std_error[j] for j in range(len(self.data))])
        q_lower = np.array([qth[j] - v * std_error[j] for j in range(len(self.data))])

        if plot_figure:
            fig, ax = Plot.confidence_level(
                qth, self.data, q_lower, q_upper, alpha=alpha, **kwargs
            )
            return q_upper, q_lower, fig, ax
        else:
            return q_upper, q_lower

    def plot(
        self,
        fig_size: Tuple[float, float] = (10, 5),
        xlabel: str = "Actual data",
        ylabel: str = "cdf",
        fontsize: int = 15,
        cdf: Union[np.ndarray, list] = None,
        parameters: Dict[str, Union[float, Any]] = None,
    ) -> Tuple[Figure, Tuple[Axes, Axes]]:  # pylint: disable=arguments-differ
        """Probability plot.

        Probability Plot method calculates the theoretical values based on the Gumbel distribution
        parameters, theoretical cdf (or weibul), and calculates the confidence interval.

        Parameters
        ----------
        fig_size: tuple, Default is (10, 5).
            Size of the figure.
        cdf: [np.ndarray]
            theoretical cdf calculated using weibul or using the distribution cdf function.
        fig_size: [tuple]
            Default is (10, 5)
        xlabel: [str]
            Default is "Actual data"
        ylabel: [str]
            Default is "cdf"
        fontsize: [float]
            Default is 15.
        parameters: Dict[str, str]
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.

        Returns
        -------
        Figure:
            matplotlib figure object
        Tuple[Axes, Axes]:
            matplotlib plot axes

        Examples
        --------
        - Instantiate the Gumbel class with the data and the parameters.

            >>> import matplotlib.pyplot as plt
            >>> data = np.loadtxt("examples/data/time_series2.txt")
            >>> parameters = {"loc": 463.8040, "scale": 220.0724}
            >>> gumbel_dist = Gumbel(data, parameters)

        - to calculate the confidence interval, we need to provide the confidence level (`alpha`).

            >>> fig, ax = gumbel_dist.plot()
            >>> print(fig)
            Figure(1000x500)
            >>> print(ax)
            (<Axes: xlabel='Actual data', ylabel='pdf'>, <Axes: xlabel='Actual data', ylabel='cdf'>)

        .. image:: /_images/gumbel-plot.png
            :align: center
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        scale = parameters.get("scale")

        if scale <= 0:
            raise ValueError("Scale parameter is negative")

        if cdf is None:
            cdf = PlottingPosition.weibul(self.data)
        else:
            # if the cdf is given, check if the length is the same as the data
            if len(cdf) != len(self.data):
                raise ValueError(
                    "Length of cdf does not match the length of data, use the `PlottingPosition.weibul(data)` "
                    "to the get the non-exceedance probability"
                )

        q_x = np.linspace(
            float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
        )
        pdf_fitted: np.ndarray = self.pdf(parameters=parameters, data=q_x)
        cdf_fitted: np.ndarray = self.cdf(parameters=parameters, data=q_x)

        fig, ax = Plot.details(
            q_x,
            self.data,
            pdf_fitted,
            cdf_fitted,
            cdf,
            fig_size=fig_size,
            xlabel=xlabel,
            ylabel=ylabel,
            fontsize=fontsize,
        )

        return fig, ax


class GEV(AbstractDistribution):
    """GEV (Generalized Extreme value statistics)

    - The Generalized Extreme Value (GEV) distribution is used to model the largest or smallest value among a large
        set of independent, identically distributed random values.
    - The GEV distribution encompasses three types of distributions: Gumbel, Fréchet, and Weibull, which are
        distinguished by a shape parameter (:math:`\\xi` (xi)).

    - The probability density function (PDF) of the Generalized-extreme-value distribution is:

        .. math::
            f(x; \\zeta, \\delta, \\xi)=\\frac{1}{\\delta}\\mathrm{*}{\\mathrm{Q(x)}}^{\\xi+1}\\mathrm{
            *} e^{\\mathrm{-Q(x)}}

        .. math::
            Q(x; \\zeta, \\delta, \\xi)=
            \\begin{cases}
                \\left(1+ \\xi \\left(\\frac{x-\\zeta}{\\delta} \\right) \\right)^\\frac{-1}{\\xi} &
                \\quad\\land\\xi\\neq 0 \\\\
                e^{- \\left(\\frac{x-\\zeta}{\\delta} \\right)} & \\quad \\land \\xi=0
            \\end{cases}
          :label: gev-pdf

        Where the :math:`\\delta` (delta) is the scale parameter, :math:`\\zeta` (zeta) is the location parameter,
        and :math:`\\xi` (xi) is the shape parameter.

    - The location parameter :math:`\\zeta` shifts the distribution along the x-axis. It essentially determines the mode
        (peak) of the distribution and its location. Changing the location parameter moves the distribution left or
        right without altering its shape. The location parameter ranges from negative infinity to positive infinity.
    - The scale parameter :math:`\\delta` controls the spread or dispersion of the distribution. A larger scale parameter
        results in a wider distribution, while a smaller scale parameter results in a narrower distribution. It must
        always be positive.
    - The shape parameter :math:`\\xi` (xi) determines the shape of the distribution. The shape parameter can be positive,
        negative, or zero. The shape parameter is used to classify the GEV distribution into three types: :math:`\\xi = 0`
        Gumbel (Type I), :math:`\\xi > 0` Fréchet (Type II), and :math:`\\xi < 0` Weibull (Type III). The shape
        parameter determines the tail behavior of the distribution.

        In hydrology, the distribution is reparametrized with :math:`k=-\\xi` (xi) (El Adlouni et al., 2008)
        The cumulative distribution functions.

    - The cumulative distribution functions.

        .. math::
            F(x; \\zeta, \\delta, \\xi)=
            \\begin{cases}
                \\exp\\left(- \\left(1+ \\xi \\left(\\frac{x-\\zeta}{\\delta} \\right) \\right)^\\frac{-1}{\\xi} \\right) &
                \\quad\\land\\xi\\neq 0 and 1 + \\xi \\left( \\frac{x-\\zeta}{\\delta}\\right) \\\\
                \\exp\\left(- \\exp\\left(- \\frac{x-\\zeta}{\\delta} \\right) \\right) & \\quad \\land \\xi=0
            \\end{cases}
          :label: gev-cdf

    """

    def __init__(
        self,
        data: Union[list, np.ndarray] = None,
        parameters: Dict[str, float] = None,
    ):
        """GEV.

        Parameters
        ----------
        data: [list]
            data time series.
        parameters: Dict[str, str]
            {"loc": val, "scale": val, "shape": value}

            - loc: [numeric]
                location parameter of the GEV distribution.
            - scale: [numeric]
                scale parameter of the GEV distribution.
            - shape: [numeric]
                shape parameter of the GEV distribution.

        Examples
        --------
        - First load the sample data.

            >>> data = np.loadtxt("examples/data/gev.txt")

        - I nstantiate the Gumbel class only with the data.

            >>> gev_dist = GEV(data)
            >>> print(gev_dist) # doctest: +SKIP
            <statista.distributions.Gumbel object at 0x000001CDDE9563F0>

        - You can also instantiate the Gumbel class with the data and the parameters if you already have them.

            >>> parameters = {"loc": 0, "scale": 1, "shape": 0.1}
            >>> gev_dist = GEV(data, parameters)
            >>> print(gev_dist) # doctest: +SKIP
            <statista.distributions.Gumbel object at 0x000001CDDEB32C00>
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
        plot_figure: bool = False,
        parameters: Dict[str, float] = None,
        data: Union[List[float], np.ndarray] = None,
        *args,
        **kwargs,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """pdf.

        Returns the value of GEV's pdf with parameters loc and scale at x.

        Parameters
        ----------
        parameters: Dict[str, float], optional, default is None.
            if not provided, the parameters provided in the class initialization will be used.
            {"loc": val, "scale": val, "shape": value}

            - loc: [numeric]
                location parameter of the GEV distribution.
            - scale: [numeric]
                scale parameter of the GEV distribution.
            - shape: [numeric]
                shape parameter of the GEV distribution.
        data : np.ndarray, default is None.
            array if you want to calculate the pdf for different data than the time series given to the constructor
            method.
        plot_figure: [bool]
            Default is False.
        kwargs:
            fig_size: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "pdf".
            fontsize: [int]
                Default is 15

        Returns
        -------
        pdf: [np.ndarray]
            probability density function pdf.
        fig: matplotlib.figure.Figure, if `plot_figure` is True.
            Figure object.
        ax: matplotlib.axes.Axes, if `plot_figure` is True.
            Axes object.

        Examples
        --------
        >>> data = np.loadtxt("examples/data/gev.txt")
        >>> parameters = {"loc": 0, "scale": 1, "shape": 0.1}
        >>> gev_dist = GEV(data, parameters)
        >>> gev_dist.pdf(plot_figure=True)

        .. image:: /_images/gev-random-pdf.png
            :align: center
        """
        result = super().pdf(
            parameters=parameters,
            data=data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )

        return result

    def random(
        self,
        size: int,
        parameters: Dict[str, Union[float, Any]] = None,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """Generate Random Variable.

        Parameters
        ----------
        size: int
            size of the random generated sample.
        parameters: Dict[str, str]
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.

        Returns
        -------
        data: [np.ndarray]
            random generated data.

        Examples
        --------
        - To generate a random sample that follow the gumbel distribution with the parameters loc=0 and scale=1.

            >>> parameters = {'loc': 0, 'scale': 1, "shape": 0.1}
            >>> gev_dist = GEV(parameters=parameters)
            >>> random_data = gev_dist.random(100)

        - then we can use the `pdf` method to plot the pdf of the random data.

            >>> gev_dist.pdf(data=random_data, plot_figure=True, xlabel="Random data")

            .. image:: /_images/gev-random-pdf.png
                :align: center

            >>> gev_dist.cdf(data=random_data, plot_figure=True, xlabel="Random data")

            .. image:: /_images/gev-random-cdf.png
                :align: center
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        loc = parameters.get("loc")
        scale = parameters.get("scale")
        shape = parameters.get("shape")

        if scale <= 0:
            raise ValueError("Scale parameter is negative")

        random_data = genextreme.rvs(loc=loc, scale=scale, c=shape, size=size)
        return random_data

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
        plot_figure: bool = False,
        parameters: Dict[str, Union[float, Any]] = None,
        data: Union[List[float], np.ndarray] = None,
        *args,
        **kwargs,
    ) -> Union[
        Tuple[np.ndarray, Figure, Axes], np.ndarray
    ]:  # pylint: disable=arguments-differ
        """cdf.

        cdf calculates the value of Gumbel's cdf with parameters loc and scale at x.

        Parameters
        ----------
        parameters: Dict[str, str], optional, default is None.
            if not provided, the parameters provided in the class initialization will be used.
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        data : np.ndarray, default is None.
            array if you want to calculate the cdf for different data than the time series given to the constructor
            method.
        plot_figure: [bool]
            Default is False.
        kwargs:
            fig_size: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "cdf".
            fontsize: [int]
                Default is 15.

        Returns
        -------
        cdf: [array]
            cumulative distribution function cdf.
        fig: matplotlib.figure.Figure, if `plot_figure` is True.
            Figure object.
        ax: matplotlib.axes.Axes, if `plot_figure` is True.
            Axes object.

        Examples
        --------
        >>> data = np.loadtxt("examples/data/gev.txt")
        >>> parameters = {"loc": 0, "scale": 1, "shape": 0.1}
        >>> gev_dist = GEV(data, parameters)
        >>> gev_dist.cdf(plot_figure=True)

        .. image:: /_images/gev-random-cdf.png
            :align: center
        """
        result = super().cdf(
            parameters=parameters,
            data=data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )
        return result

    def return_period(self, parameters: Dict[str, Union[float, Any]], data: np.ndarray):
        """return_period.

            calculate return period calculates the return period for a list/array of values or a single value.

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
        cdf = self.cdf(parameters, data=data)

        rp = 1 / (1 - cdf)

        return rp

    def fit_model(
        self,
        method: str = "mle",
        obj_func=None,
        threshold: Union[int, float, None] = None,
        test: bool = True,
    ) -> Dict[str, float]:
        """Fit model.

        fit_model estimates the distribution parameter based on MLM
        (Maximum likelihood method), if an objective function is entered as an input

        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for the values below (x < C), now the likeliest parameters
        are those at the max value of multiplication between two functions max(L1*L2).

        In this case, the L1 is still the product of multiplication of probability
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
        Dict[str, str]:
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the GEV distribution.
            - scale: [numeric]
                scale parameter of the GEV distribution.
            - shape: [numeric]
                shape parameter of the GEV distribution.

        Examples
        --------
        - Instantiate the Gumbel class only with the data.

            >>> data = np.loadtxt("examples/data/gev.txt")
            >>> gev_dist = GEV(data)

        - Then use the `fit_model` method to estimate the distribution parameters. the method takes the method as
            parameter, the default is 'mle'. the `test` parameter is used to perform the Kolmogorov-Smirnov and chisquare
            test.

            >>> parameters = gev_dist.fit_model(method="mle", test=True)
            -----KS Test--------
            Statistic = 0.06
            Accept Hypothesis
            P value = 0.9942356257694902
            >>> print(parameters)
            {'loc': -0.05962776672431072, 'scale': 0.9114319092295455, 'shape': 0.03492066094614391}

        - You can also use the `lmoments` method to estimate the distribution parameters.

            >>> parameters = gev_dist.fit_model(method="lmoments", test=True)
            -----KS Test--------
            Statistic = 0.05
            Accept Hypothesis
            P value = 0.9996892272702655
            >>> print(parameters)
            {'loc': -0.07182150513604696, 'scale': 0.9153288314267931, 'shape': 0.018944589308927475}

        - You can also use the `fit_model` method to estimate the distribution parameters using the 'optimization'
            method. the optimization method requires the `obj_func` and `threshold` parameter. the method
            will take the `threshold` number and try to fit the data values that are greater than the threshold.
            >>> threshold = np.quantile(data, 0.80)
            >>> print(threshold)
            1.39252
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
            # try:
            #     self.chisquare()
            # except ValueError:
            #     print("chisquare test failed")

        return param

    def inverse_cdf(
        self,
        cdf: Union[np.ndarray, List[float]] = None,
        parameters: Dict[str, Union[float, Any]] = None,
    ) -> np.ndarray:
        """Theoretical Estimate.

        Theoretical Estimate method calculates the theoretical values based on a given non-exceedance probability

        Parameters
        ----------
        parameters: [list]
            location and scale parameters of the gumbel distribution.
        cdf: [list]
            cumulative distribution function/ Non-Exceedance probability.

        Returns
        -------
        theoretical value: [numeric]
            Value based on the theoretical distribution

        Examples
        --------
        - Instantiate the Gumbel class only with the data.

            >>> data = np.loadtxt("examples/data/gev.txt")
            >>> parameters = {'loc': 0, 'scale': 1, "shape": 0.1}
            >>> gev_dist = GEV(data, parameters)

        - We will generate a random numbers between 0 and 1 and pass it to the inverse_cdf method as a probabilities
            to get the data that coresponds to these probabilities based on the distribution.

            >>> cdf = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            >>> data_values = gev_dist.inverse_cdf(cdf)
            >>> print(data_values)
            [-0.86980039 -0.4873901   0.08704056  0.64966292  1.39286858  2.01513112]
        """
        if parameters is None:
            parameters = self.parameters

        if any(cdf) < 0 or any(cdf) > 1:
            raise ValueError("cdf Value Invalid")

        q_th = self._inv_cdf(cdf, parameters)
        return q_th

    @staticmethod
    def _inv_cdf(cdf: Union[np.ndarray, List[float]], parameters: Dict[str, float]):
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        shape = parameters.get("shape")

        if scale <= 0:
            raise ValueError("Parameters Invalid")

        if shape is None:
            raise ValueError("Shape parameter should not be None")
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

        The smaller the D static, the more likely that the two samples are drawn from the same distribution
        IF Pvalue < significance level ------ reject

        Returns
        -------
        Dstatic: [numeric]
            The smaller the D static the more likely that the two samples are drawn from the same distribution
        Pvalue : [numeric]
            IF Pvalue < significance level ------ reject the null hypothesis
        """
        return super().ks()

    def chisquare(self) -> tuple:
        """chisquare test"""
        return super().chisquare()

    def confidence_interval(
        self,
        alpha: float = 0.1,
        plot_figure: bool = False,
        prob_non_exceed: np.ndarray = None,
        parameters: Dict[str, Union[float, Any]] = None,
        state_function: callable = None,
        n_samples: int = 100,
        method: str = "lmoments",
        **kwargs,
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, Figure, Axes]
    ]:  # pylint: disable=arguments-differ
        """confidence_interval.

        Parameters
        ----------
        parameters: Dict[str, str], optional, default is None.
            if not provided, the parameters provided in the class initialization will be used.
            {"loc": val, "scale": val, "shape": value}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        prob_non_exceed : [list]
            Non-Exceedance probability
        alpha : [numeric]
            alpha or SignificanceLevel is a value of the confidence interval.
        state_function: callable, Default is GEV.ci_func
            function to calculate the confidence interval.
        n_samples: [int]
            number of samples generated by the bootstrap method Default is 100.
        method: [str]
            method used to fit the generated samples from the bootstrap method ["lmoments", "mle", "mm"]. Default is
            "lmoments".
        plot_figure: bool, optional, default is False.
            to plot the confidence interval.

        Returns
        -------
        q_upper: [list]
            upper-bound coresponding to the confidence interval.
        q_lower: [list]
            lower-bound coresponding to the confidence interval.
        fig: matplotlib.figure.Figure
            Figure object.
        ax: matplotlib.axes.Axes
            Axes object.

        Examples
        --------
        - Instantiate the GEV class with the data and the parameters.

            >>> import matplotlib.pyplot as plt
            >>> data = np.loadtxt("examples/data/time_series1.txt")
            >>> parameters = {"loc": 16.3928, "scale": 0.70054, "shape": -0.1614793,}
            >>> gev_dist = GEV(data, parameters)

        - to calculate the confidence interval, we need to provide the confidence level (`alpha`).

            >>> upper, lower = gev_dist.confidence_interval(alpha=0.1)

        - You can also plot confidence intervals

            >>> upper, lower, fig, ax = gev_dist.confidence_interval(alpha=0.1, plot_figure=True, marker_size=10)

        .. image:: /_images/gev-confidence-interval.png
            :align: center
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        scale = parameters.get("scale")
        if scale <= 0:
            raise ValueError("Scale parameter is negative")

        if prob_non_exceed is None:
            prob_non_exceed = PlottingPosition.weibul(self.data)
        else:
            # if the prob_non_exceed is given, check if the length is the same as the data
            if len(prob_non_exceed) != len(self.data):
                raise ValueError(
                    "Length of prob_non_exceed does not match the length of data, use the `PlottingPosition.weibul(data)` "
                    "to the get the non-exceedance probability"
                )
        if state_function is None:
            state_function = GEV.ci_func

        ci = ConfidenceInterval.boot_strap(
            self.data,
            state_function=state_function,
            gevfit=parameters,
            F=prob_non_exceed,
            alpha=alpha,
            n_samples=n_samples,
            method=method,
            **kwargs,
        )
        q_lower = ci["lb"]
        q_upper = ci["ub"]

        if plot_figure:
            qth = self._inv_cdf(prob_non_exceed, parameters)
            fig, ax = Plot.confidence_level(
                qth, self.data, q_lower, q_upper, alpha=alpha, **kwargs
            )
            return q_upper, q_lower, fig, ax
        else:
            return q_upper, q_lower

    def plot(
        self,
        fig_size=(10, 5),
        xlabel="Actual data",
        ylabel="cdf",
        fontsize=15,
        cdf: Union[np.ndarray, list] = None,
        parameters: Dict[str, Union[float, Any]] = None,
    ) -> Tuple[Figure, Tuple[Axes, Axes]]:
        """Probability Plot.

        Probability Plot method calculates the theoretical values based on the Gumbel distribution
        parameters, theoretical cdf (or weibul), and calculates the confidence interval.

        Parameters
        ----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val, shape: val}

            - loc: [numeric]
                Location parameter of the GEV distribution.
            - scale: [numeric]
                Scale parameter of the GEV distribution.
            - shape: [float, int]
                Shape parameter for the GEV distribution.
        cdf: [list]
            Theoretical cdf calculated using weibul or using the distribution cdf function.
        fontsize: [numeric]
            Font size of the axis labels and legend
        ylabel: [string]
            y label string
        xlabel: [string]
            X label string
        fig_size: [tuple]
            size of the pdf and cdf figure

        Returns
        -------
        Figure:
            matplotlib figure object
        Tuple[Axes, Axes]:
            matplotlib plot axes

        Examples
        --------
        - Instantiate the Gumbel class with the data and the parameters.

            >>> import numpy as np
            >>> data = np.loadtxt("examples/data/time_series1.txt")
            >>> parameters = {"loc": 16.3928, "scale": 0.70054, "shape": -0.1614793,}
            >>> gev_dist = GEV(data, parameters)

        - to calculate the confidence interval, we need to provide the confidence level (`alpha`).

            >>> fig, ax = gumbel_dist.plot()
            >>> print(fig)
            Figure(1000x500)
            >>> print(ax)
            (<Axes: xlabel='Actual data', ylabel='pdf'>, <Axes: xlabel='Actual data', ylabel='cdf'>)

        .. image:: /_images/gev-plot.png
            :align: center
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters
        scale = parameters.get("scale")

        if scale <= 0:
            raise ValueError("Scale parameter is negative")

        if cdf is None:
            cdf = PlottingPosition.weibul(self.data)
        else:
            # if the prob_non_exceed is given, check if the length is the same as the data
            if len(cdf) != len(self.data):
                raise ValueError(
                    "Length of prob_non_exceed does not match the length of data, use the `PlottingPosition.weibul(data)` "
                    "to the get the non-exceedance probability"
                )

        q_x = np.linspace(
            float(self.data_sorted[0]), 1.5 * float(self.data_sorted[-1]), 10000
        )
        pdf_fitted = self.pdf(parameters=parameters, data=q_x)
        cdf_fitted = self.cdf(parameters=parameters, data=q_x)

        fig, ax = Plot.details(
            q_x,
            self.data,
            pdf_fitted,
            cdf_fitted,
            cdf,
            fig_size=fig_size,
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
            gevfit: [list]
                GEV parameter [shape, location, scale]
            F: [list]
                Non-Exceedance probability
            method: [str]
                method used to fit the generated samples from the bootstrap method ["lmoments", "mle", "mm"]. Default is
                "lmoments".
        """
        gevfit = kwargs["gevfit"]
        prob_non_exceed = kwargs["F"]
        method = kwargs["method"]
        # generate theoretical estimates based on a random cdf, and the dist parameters
        sample = GEV._inv_cdf(np.random.rand(len(data)), gevfit)

        # get parameters based on the new generated sample
        dist = GEV(sample)
        new_param = dist.fit_model(method=method, test=False)

        # return period
        # T = np.arange(0.1, 999.1, 0.1) + 1
        # +1 in order not to make 1- 1/0.1 = -9
        # T = np.linspace(0.1, 999, len(data)) + 1
        # coresponding theoretical estimate to T
        # prob_non_exceed = 1 - 1 / T
        q_th = GEV._inv_cdf(prob_non_exceed, new_param)

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
#         fig_size: tuple = (6, 5),
#         xlabel: str = "Actual data",
#         ylabel: str = "pdf",
#         fontsize: Union[float, int] = 15,
#         data: Union[bool, np.ndarray] = True,
#     ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
#         """pdf.
#
#         Returns the value of Gumbel's pdf with parameters loc and scale at x .
#
#         Parameters
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
#         if isinstance(data, bool):
#             ts = self.data
#         else:
#             ts = data
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
#             pdf_fitted = self.pdf(loc, scale, data=q_x)
#
#             fig, ax = Plot.pdf(
#                 q_x,
#                 pdf_fitted,
#                 self.data_sorted,
#                 fig_size=fig_size,
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
#         fig_size: tuple = (6, 5),
#         xlabel: str = "data",
#         ylabel: str = "cdf",
#         fontsize: int = 15,
#         data: Union[bool, np.ndarray] = True,
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
#         if isinstance(data, bool):
#             ts = self.data
#         else:
#             ts = data
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
#             cdf_fitted = self.cdf(loc, scale, data=q_x)
#
#             cdf_Weibul = PlottingPosition.weibul(self.data_sorted)
#
#             fig, ax = Plot.cdf(
#                 q_x,
#                 cdf_fitted,
#                 self.data_sorted,
#                 cdf_Weibul,
#                 fig_size=fig_size,
#                 xlabel=xlabel,
#                 ylabel=ylabel,
#                 fontsize=fontsize,
#             )
#
#             return cdf, fig, ax
#         else:
#             return cdf
#
#     def fit_model(
#         self,
#         method: str = "mle",
#         obj_func=None,
#         threshold: Union[int, float, None] = None,
#         test: bool = True,
#     ) -> tuple:
#         """fit_model.
#
#         fit_model estimates the distribution parameter based on MLM
#         (Maximum likelihood method), if an objective function is entered as an input
#
#         There are two likelihood functions (L1 and L2), one for values above some
#         threshold (x>=C) and one for values below (x < C), now the likeliest parameters
#         are those at the max value of multiplication between two functions max(L1*L2).
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
#     def inverse_cdf(
#         loc: Union[float, int],
#         scale: Union[float, int],
#         prob_non_exceed: np.ndarray,
#     ) -> np.ndarray:
#         """inverse_cdf.
#
#         inverse_cdf method calculates the theoretical values based on a given non-exceedance probability
#
#         Parameters
#         -----------
#         param : [list]
#             location ans scale parameters of the gumbel distribution.
#         prob_non_exceed : [list]
#             cummulative distribution function/ Non Exceedence probability.
#
#         Returns
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
    """Exponential distribution.

    - The exponential distribution assumes that small values occur more frequently than large values.

    - The probability density function (PDF) of the Exponential distribution is:

        .. math::
            f(x; \\delta, \\beta) =
            \\begin{cases}
                f(x; \\delta, \\beta) = \\frac{1}{\\beta} e^{-\\frac{x - \\delta}{\\beta}} & \\quad x \\geq 0 \\\\
                0 & \\quad x < 0
            \\end{cases}
          :label: exp-equation

    - The probability density function above uses the location parameter :math:`\\delta` and the scale parameter
        :math:`\\beta` to define the distribution in a standardized form.
    - A common parameterization for the exponential distribution is in terms of the rate parameter :math:`\\lambda`,
        such that :math:`\\lambda = 1 / \\beta`.
    - The Location Parameter (:math:`\\delta`): This shifts the starting point of the distribution. The distribution is
        defined for :math:`x \\geq \\delta`.
    - Scale Parameter (:math:`\\beta`): This determines the spread of the distribution. The rate parameter
        :math:`\\lambda` is the inverse of the scale parameter, so :math:`\\lambda = \\frac{1}{\\beta}`.

    - The cumulative distribution functions.

        .. math::
            F(x; \\delta, \\beta) =
            \\begin{cases}
                F(x; \\delta, \\beta) = 1 - e^{-\\frac{x - \\delta}{\\beta}} & \\quad x \\geq 0 \\\\
                0 & \\quad x < 0
            \\end{cases}
          :label: exp-cdf

    """

    def __init__(
        self,
        data: Union[list, np.ndarray] = None,
        parameters: Dict[str, float] = None,
    ):
        """Exponential Distribution.

        Parameters
        ----------
        data: [list]
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
        plot_figure: bool = False,
        parameters: Dict[str, float] = None,
        data: Union[List[float], np.ndarray] = None,
        *args,
        **kwargs,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """pdf.

        Returns the value of Gumbel's pdf with parameters loc and scale at x.

        Parameters
        ----------
        parameters: Dict[str, str], optional, default is None.
            if not provided, the parameters provided in the class initialization will be used.
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        data: np.ndarray, default is None.
            array if you want to calculate the pdf for different data than the time series given to the constructor
            method.
        plot_figure: [bool]
            Default is False.
        kwargs:
            fig_size: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "pdf".
            fontsize: [int]
                Default is 15

        Returns
        -------
        pdf: [array]
            probability density function pdf.
        fig: matplotlib.figure.Figure, if `plot_figure` is True.
            Figure object.
        ax: matplotlib.axes.Axes, if `plot_figure` is True.
            Axes object.

        Examples
        --------
        >>> data = np.loadtxt("examples/data/expo.txt")
        >>> parameters = {'loc': 0, 'scale': 2}
        >>> expo_dist = Exponential(data, parameters)
        >>> expo_dist.pdf(plot_figure=True)

        .. image:: /_images/expo-random-pdf.png
            :align: center
        """
        result = super().pdf(
            parameters=parameters,
            data=data,
            plot_figure=plot_figure,
            *args,
            **kwargs,
        )

        return result

    def random(
        self,
        size: int,
        parameters: Dict[str, Union[float, Any]] = None,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """Generate Random Variable.

        Parameters
        ----------
        size: int
            size of the random generated sample.
        parameters: Dict[str, str]
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.

        Returns
        -------
        data: [np.ndarray]
            random generated data.

        Examples
        --------
        - To generate a random sample that follow the gumbel distribution with the parameters loc=0 and scale=1.

            >>> parameters = {'loc': 0, 'scale': 2}
            >>> expon_dist = Exponential(parameters=parameters)
            >>> random_data = expon_dist.random(1000)

        - then we can use the `pdf` method to plot the pdf of the random data.

            >>> expon_dist.pdf(data=random_data, plot_figure=True, xlabel="Random data")

            .. image:: /_images/expo-random-pdf.png
                :align: center

            >>> expon_dist.cdf(data=random_data, plot_figure=True, xlabel="Random data")

            .. image:: /_images/expo-random-cdf.png
                :align: center
        """
        # if no parameters are provided, take the parameters provided in the class initialization.
        if parameters is None:
            parameters = self.parameters

        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale <= 0:
            raise ValueError("Scale parameter is negative")

        random_data = expon.rvs(loc=loc, scale=scale, size=size)
        return random_data

    @staticmethod
    def _cdf_eq(
        data: Union[list, np.ndarray], parameters: Dict[str, Union[float, Any]]
    ) -> np.ndarray:
        loc = parameters.get("loc")
        scale = parameters.get("scale")
        if scale <= 0:
            raise ValueError("Scale parameter is negative")
        # if loc <= 0:
        #     raise ValueError("Threshold parameter should be greater than zero")
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
        plot_figure: bool = False,
        parameters: Dict[str, Union[float, Any]] = None,
        data: Union[List[float], np.ndarray] = None,
        *args,
        **kwargs,
    ) -> Union[
        Tuple[np.ndarray, Figure, Any], np.ndarray
    ]:  # pylint: disable=arguments-differ
        """cdf.

        cdf calculates the value of Gumbel's cdf with parameters loc and scale at x.

        parameter:
        ----------
        parameters: Dict[str, str], optional, default is None.
            if not provided, the parameters provided in the class initialization will be used.
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        data: np.ndarray, default is None.
            array if you want to calculate the cdf for different data than the time series given to the constructor
            method.
        plot_figure: [bool]
            Default is False.
        kwargs:
            fig_size: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "cdf".
            fontsize: [int]
                Default is 15.

        Returns
        -------
        cdf: [array]
            probability density function cdf.
        fig: matplotlib.figure.Figure, if `plot_figure` is True.
            Figure object.
        ax: matplotlib.axes.Axes, if `plot_figure` is True.
            Axes object.

        Examples
        --------
        >>> data = np.loadtxt("examples/data/expo.txt")
        >>> parameters = {'loc': 0, 'scale': 2}
        >>> expo_dist = Exponential(data, parameters)
        >>> expo_dist.cdf(plot_figure=True)  # doctest: +SKIP

        .. image:: /_images/expo-random-cdf.png
            :align: center
        """
        result = super().cdf(
            parameters=parameters,
            data=data,
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
        """fit_model.

        fit_model estimates the distribution parameter based on MLM
        (Maximum likelihood method), if an objective function is entered as an input

        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for the values below (x < C), now the likeliest parameters
        are those at the max value of multiplication between two functions max(L1*L2).

        In this case, the L1 is still the product of multiplication of probability
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

        Examples
        --------
        - Instantiate the `Exponential` class only with the data.

            >>> data = np.loadtxt("examples/data/expo.txt")
            >>> expo_dist = Exponential(data)

        - Then use the `fit_model` method to estimate the distribution parameters. the method takes the method as
            parameter, the default is 'mle'. the `test` parameter is used to perform the Kolmogorov-Smirnov and chisquare
            test.

            >>> parameters = expo_dist.fit_model(method="mle", test=True)
            -----KS Test--------
            Statistic = 0.019
            Accept Hypothesis
            P value = 0.9937026761524456
            Out[14]: {'loc': 0.0009, 'scale': 2.0498075}
            >>> print(parameters)
            {'loc': 0, 'scale': 2}

        - You can also use the `lmoments` method to estimate the distribution parameters.

            >>> parameters = expo_dist.fit_model(method="lmoments", test=True)
            -----KS Test--------
            Statistic = 0.021
            Accept Hypothesis
            P value = 0.9802627322900355
            >>> print(parameters)
            {'loc': -0.00805012182182141, 'scale': 2.0587576218218215}
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
            # try:
            #     self.chisquare()
            # except ValueError:
            #     print("chisquare test failed")

        return param

    def inverse_cdf(
        self,
        cdf: Union[np.ndarray, List[float]] = None,
        parameters: Dict[str, Union[float, Any]] = None,
    ) -> np.ndarray:
        """Theoretical Estimate.

        Theoretical Estimate method calculates the theoretical values based on a given  non-exceedance probability

        Parameters
        -----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the gumbel distribution.
            - scale: [numeric]
                scale parameter of the gumbel distribution.
        cdf: [list]
            cumulative distribution function/ Non-Exceedance probability.

        Returns
        -------
        theoretical value: [numeric]
            Value based on the theoretical distribution

        Examples
        --------
        - Instantiate the Exponential class only with the data.

            >>> data = np.loadtxt("examples/data/expo.txt")
            >>> parameters = {'loc': 0, 'scale': 2}
            >>> expo_dist = Exponential(data, parameters)

        - We will generate a random numbers between 0 and 1 and pass it to the inverse_cdf method as a probabilities
            to get the data that coresponds to these probabilities based on the distribution.

            >>> cdf = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
            >>> data_values = expo_dist.inverse_cdf(cdf)
            >>> print(data_values)
            [0.21072103 0.4462871  1.02165125 1.83258146 3.21887582 4.60517019]
        """
        if parameters is None:
            parameters = self.parameters

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

        The smaller the D static, the more likely that the two samples are drawn from the same distribution
        IF Pvalue < significance level ------ reject

        Returns
        -------
            Dstatic: [numeric]
                The smaller the D static the more likely that the two samples are drawn from the same distribution
            Pvalue : [numeric]
                IF Pvalue < significance level ------ reject the null hypothesis
        """
        return super().ks()

    def chisquare(self) -> tuple:
        """chisquare test"""
        return super().chisquare()


class Normal(AbstractDistribution):
    """Normal Distribution.

    - The probability density function (PDF) of the Normal distribution is:

        .. math::
            f(x: threshold, scale) = (1/scale) e **(- (x-threshold)/scale)
          :label: normal-equation

    - The cumulative distribution functions.

        .. math::
            F(x: threshold, scale) = 1 - e **(- (x-threshold)/scale)
          :label: normal-cdf
    """

    def __init__(
        self,
        data: Union[list, np.ndarray] = None,
        parameters: Dict[str, float] = None,
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
        plot_figure: bool = False,
        parameters: Dict[str, float] = None,
        data: Union[List[float], np.ndarray] = None,
        *args,
        **kwargs,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """pdf.

        Returns the value of Gumbel's pdf with parameters loc and scale at x.

        Parameters
        -----------
        parameters: Dict[str, str], optional, default is None.
            if not provided, the parameters provided in the class initialization will be used.
            {"loc": val, "scale": val, "shape": value}

            - loc: [numeric]
                location parameter of the GEV distribution.
            - scale: [numeric]
                scale parameter of the GEV distribution.
        data : np.ndarray, default is None.
            array if you want to calculate the pdf for different data than the time series given to the constructor
            method.
        plot_figure: [bool]
            Default is False.
        kwargs:
            fig_size: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "pdf".
            fontsize: [int]
                Default is 15

        Returns
        -------
        pdf: [array]
            probability density function pdf.
        fig: matplotlib.figure.Figure, if `plot_figure` is True.
            Figure object.
        ax: matplotlib.axes.Axes, if `plot_figure` is True.
            Axes object.
        """
        result = super().pdf(
            parameters=parameters,
            data=data,
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
        plot_figure: bool = False,
        parameters: Dict[str, Union[float, Any]] = None,
        data: Union[List[float], np.ndarray] = None,
        *args,
        **kwargs,
    ) -> Union[Tuple[np.ndarray, Figure, Any], np.ndarray]:
        """cdf.

        cdf calculates the value of Normal distribution cdf with parameters loc and scale at x.

        Parameters
        ----------
        parameters: Dict[str, str], optional, default is None.
            if not provided, the parameters provided in the class initialization will be used.
            {"loc": val, "scale": val, "shape": value}

            - loc: [numeric]
                location parameter of the Normal distribution.
            - scale: [numeric]
                scale parameter of the Normal distribution.
        data : np.ndarray, default is None.
            array if you want to calculate the pdf for different data than the time series given to the constructor
            method.
        plot_figure: [bool]
            Default is False.
        kwargs:
            fig_size: [tuple]
                Default is (6, 5).
            xlabel: [str]
                Default is "Actual data".
            ylabel: [str]
                Default is "cdf".
            fontsize: [int]
                Default is 15.

        Returns
        -------
        cdf: [array]
            probability density function cdf.
        fig: matplotlib.figure.Figure, if `plot_figure` is True.
            Figure object.
        ax: matplotlib.axes.Axes, if `plot_figure` is True.
            Axes object.
        """
        result = super().cdf(
            parameters=parameters,
            data=data,
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
        """fit_model.

        fit_model estimates the distribution parameter based on MLM
        (Maximum likelihood method), if an objective function is entered as an input

        There are two likelihood functions (L1 and L2), one for values above some
        threshold (x>=C) and one for the values below (x < C), now the likeliest parameters
        are those at the max value of multiplication between two functions max(L1*L2).

        In this case, the L1 is still the product of multiplication of probability
        density function's values at xi, but the L2 is the probability that threshold
        value C will be exceeded (1-F(C)).

        Parameters
        ----------
        obj_func: [function]
            function to be used to get the distribution parameters.
        threshold: [numeric]
            Value you want to consider only the greater values.
        method: [string]
            'mle', 'mm', 'lmoments', optimization
        test: bool
            Default is True

        Returns
        -------
        parameters: [list]
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
            # try:
            #     self.chisquare()
            # except ValueError:
            #     print("chisquare test failed")

        return param

    def inverse_cdf(
        self,
        cdf: Union[np.ndarray, List[float]] = None,
        parameters: Dict[str, Union[float, Any]] = None,
    ) -> np.ndarray:
        """Theoretical Estimate.

        Theoretical Estimate method calculates the theoretical values based on a given  non exceedence probability

        Parameters
        -----------
        parameters: Dict[str, str]
            {"loc": val, "scale": val}

            - loc: [numeric]
                location parameter of the Normal distribution.
            - scale: [numeric]
                scale parameter of the Normal distribution.
        cdf: [list]
            cumulative distribution function/ Non-Exceedance probability.

        Returns
        -------
        numeric:
            Value based on the theoretical distribution
        """
        if parameters is None:
            parameters = self.parameters

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

        The smaller the D static, the more likely that the two samples are drawn from the same distribution
        IF Pvalue < significance level ------ reject

        Returns
        -------
        Dstatic: [numeric]
            The smaller the D static the more likely that the two samples are drawn from the same distribution
        Pvalue: [numeric]
            IF Pvalue < significance level ------ reject the null hypothesis
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
        """Delegate method calls to the subclass"""
        # Retrieve the attribute or method from the distribution object
        try:
            # Retrieve the attribute or method from the subclasses
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
