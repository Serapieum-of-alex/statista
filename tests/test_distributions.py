"""Test distributions module."""
from typing import List

import numpy as np
from matplotlib.figure import Figure

from statista.confidence_interval import ConfidenceInterval
from statista.distributions import (
    GEV,
    Gumbel,
    PlottingPosition,
    Exponential,
    Normal,
    Distributions,
)


class TestPlottingPosition:
    def test_plotting_position_weibul(
        self,
        time_series1: list,
    ):
        cdf = PlottingPosition.weibul(time_series1)
        assert isinstance(cdf, np.ndarray)
        rp = PlottingPosition.weibul(time_series1, return_period=True)
        assert isinstance(rp, np.ndarray)

    def test_plotting_position_rp(
        self,
        time_series1: list,
    ):
        cdf = PlottingPosition.weibul(time_series1)
        rp = PlottingPosition.return_period(cdf)
        assert isinstance(rp, np.ndarray)


class TestGumbel:
    def test_create_instance(
        self,
        time_series1: list,
    ):
        dist = Gumbel(time_series1)
        assert isinstance(dist.data, np.ndarray)
        assert isinstance(dist.data_sorted, np.ndarray)

    def test_estimate_parameter(
        self,
        time_series2: list,
        dist_estimation_parameters: List[str],
    ):
        dist = Gumbel(time_series2)
        for i in range(len(dist_estimation_parameters)):
            param = dist.fit_model(method=dist_estimation_parameters[i], test=False)
            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale"])
            assert dist.parameters.get("loc") is not None
            assert dist.parameters.get("scale") is not None

    def test_parameter_estimation_optimization(
        self,
        time_series2: list,
        dist_estimation_parameters: List[str],
        parameter_estimation_optimization_threshold: int,
    ):
        dist = Gumbel(time_series2)
        param = dist.fit_model(
            method="optimization",
            obj_func=Gumbel.objective_fn,
            threshold=parameter_estimation_optimization_threshold,
        )
        assert isinstance(param, dict)
        assert all(i in param.keys() for i in ["loc", "scale"])
        assert dist.parameters.get("loc") is not None
        assert dist.parameters.get("scale") is not None

    def test_ks(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        dist = Gumbel(time_series2)
        dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        dist.ks()
        assert dist.Dstatic
        assert dist.KS_Pvalue

    def test_chisquare(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        dist = Gumbel(time_series2)
        dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        dist.chisquare()
        assert dist.chistatic
        assert dist.chi_Pvalue

    def test_pdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        dist = Gumbel(time_series2)
        param = dist.fit_model(method=dist_estimation_parameters_ks, test=False)

        pdf, fig, ax = dist.pdf(param, plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        dist = Gumbel(time_series2)
        param = dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        cdf, fig, ax = dist.cdf(param, plot_figure=True)

        assert isinstance(cdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_theoretical_estimate(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        dist = Gumbel(time_series2)
        cdf_weibul = PlottingPosition.weibul(time_series2)
        param = dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        qth = dist.theoretical_estimate(param, cdf_weibul)
        assert isinstance(qth, np.ndarray)

    def test_confidence_interval(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        confidence_interval_alpha: float,
    ):
        dist = Gumbel(time_series2)
        cdf_weibul = PlottingPosition.weibul(time_series2)
        param = dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        upper, lower = dist.confidence_interval(
            param, cdf_weibul, alpha=confidence_interval_alpha
        )
        assert isinstance(upper, np.ndarray)
        assert isinstance(lower, np.ndarray)

    def test_probability_plot(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        confidence_interval_alpha: float,
    ):
        dist = Gumbel(time_series2)
        cdf_weibul = PlottingPosition.weibul(time_series2)
        param = dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        (fig1, fig2), (_, _) = dist.probability_plot(
            param, cdf_weibul, alpha=confidence_interval_alpha
        )
        assert isinstance(fig1, Figure)
        assert isinstance(fig2, Figure)


class TestGEV:
    def test_create_gev_instance(
        self,
        time_series1: list,
    ):
        dist = GEV(time_series1)
        assert isinstance(dist.data, np.ndarray)
        assert isinstance(dist.data_sorted, np.ndarray)

    def test_gev_estimate_parameter(
        self,
        time_series1: list,
        dist_estimation_parameters: List[str],
    ):
        dist = GEV(time_series1)
        for i in range(len(dist_estimation_parameters)):
            param = dist.fit_model(method=dist_estimation_parameters[i], test=False)

            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale", "shape"])
            assert dist.parameters.get("loc") is not None
            assert dist.parameters.get("scale") is not None
            assert dist.parameters.get("shape") is not None

    def test_gev_ks(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
    ):
        dist = GEV(time_series1)
        dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        dist.ks()
        assert dist.Dstatic
        assert dist.KS_Pvalue

    def test_gev_chisquare(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
    ):
        dist = GEV(time_series1)
        dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        dist.chisquare()
        assert dist.chistatic
        assert dist.chi_Pvalue

    def test_gev_pdf(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
    ):
        dist = GEV(time_series1)
        param = dist.fit_model(method=dist_estimation_parameters_ks, test=False)

        pdf, fig, ax = dist.pdf(param, plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_gev_cdf(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
    ):
        dist = GEV(time_series1)
        param = dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        cdf, fig, ax = dist.cdf(param, plot_figure=True)
        assert isinstance(cdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_gev_theoretical_estimate(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
    ):
        dist = GEV(time_series1)
        cdf_weibul = PlottingPosition.weibul(time_series1)
        param = dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        qth = dist.theoretical_estimate(param, cdf_weibul)
        assert isinstance(qth, np.ndarray)

    def test_gev_confidence_interval(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        confidence_interval_alpha: float,
    ):
        dist = GEV(time_series1)
        cdf_weibul = PlottingPosition.weibul(time_series1)
        param = dist.fit_model(method=dist_estimation_parameters_ks, test=False)

        func = GEV.ci_func
        upper, lower = dist.confidence_interval(
            param,
            prob_non_exceed=cdf_weibul,
            alpha=confidence_interval_alpha,
            statfunction=func,
            n_samples=len(time_series1),
        )
        assert isinstance(upper, np.ndarray)
        assert isinstance(lower, np.ndarray)

    def test_confidence_interval_directly(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        confidence_interval_alpha: float,
    ):
        dist = GEV(time_series1)
        cdf_weibul = PlottingPosition.weibul(time_series1)
        param = dist.fit_model(method=dist_estimation_parameters_ks, test=False)

        func = GEV.ci_func

        ci = ConfidenceInterval.boot_strap(
            time_series1,
            statfunction=func,
            gevfit=param,
            n_samples=len(time_series1),
            F=cdf_weibul,
            method="lmoments",
        )
        lb = ci["lb"]
        ub = ci["ub"]

        assert isinstance(lb, np.ndarray)
        assert isinstance(ub, np.ndarray)


# class TestAbstractDistrition:


class TestExponential:
    def test_create_instance(
        self,
        time_series1: list,
    ):
        expo_dist = Exponential(time_series1)
        assert isinstance(expo_dist.data, np.ndarray)
        assert isinstance(expo_dist.data_sorted, np.ndarray)

    def test_estimate_parameter(
        self,
        time_series2: list,
        dist_estimation_parameters: List[str],
    ):
        expo_dist = Exponential(time_series2)
        for i in range(len(dist_estimation_parameters)):
            param = expo_dist.fit_model(
                method=dist_estimation_parameters[i], test=False
            )
            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale"])
            assert expo_dist.parameters.get("loc") is not None
            assert expo_dist.parameters.get("scale") is not None

    def test_pdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        expo_dist = Exponential(time_series2)
        param = expo_dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        pdf, fig, ax = expo_dist.pdf(param, plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        expo_dist = Exponential(time_series2)
        param = expo_dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        cdf, fig, ax = expo_dist.cdf(param, plot_figure=True)
        assert isinstance(cdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_theoretical_estimate(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        expo_dist = Exponential(time_series2)
        cdf_weibul = PlottingPosition.weibul(time_series2)
        param = expo_dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        qth = expo_dist.theoretical_estimate(param, cdf_weibul)
        assert isinstance(qth, np.ndarray)


class TestNormal:
    def test_create_instance(
        self,
        time_series1: list,
    ):
        norm_dist = Normal(time_series1)
        assert isinstance(norm_dist.data, np.ndarray)
        assert isinstance(norm_dist.data_sorted, np.ndarray)

    def test_estimate_parameter(
        self,
        time_series2: list,
        dist_estimation_parameters: List[str],
    ):
        norm_dist = Normal(time_series2)
        for method in dist_estimation_parameters:
            param = norm_dist.fit_model(method=method, test=False)
            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale"])
            assert norm_dist.parameters.get("loc") is not None
            assert norm_dist.parameters.get("scale") is not None

    def test_pdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        norm_dist = Normal(time_series2)
        param = norm_dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        pdf, fig, ax = norm_dist.pdf(param, plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        norm_dist = Normal(time_series2)
        param = norm_dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        cdf, fig, ax = norm_dist.cdf(param, plot_figure=True)
        assert isinstance(cdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_theoretical_estimate(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        norm_dist = Normal(time_series2)
        cdf_weibul = PlottingPosition.weibul(time_series2)
        param = norm_dist.fit_model(method=dist_estimation_parameters_ks, test=False)
        qth = norm_dist.theoretical_estimate(param, cdf_weibul)
        assert isinstance(qth, np.ndarray)


class TestDistribution:
    def test_create_instance(
        self,
        time_series1: list,
    ):
        dist = Distributions("Gumbel", data=time_series1)
        assert isinstance(dist.data, np.ndarray)
        assert isinstance(dist.data_sorted, np.ndarray)

    def test_getter_method(
        self,
        time_series2: list,
        dist_estimation_parameters: List[str],
    ):
        dist = Distributions("Gumbel", data=time_series2)
        for i in range(len(dist_estimation_parameters)):
            param = dist.fit_model(method=dist_estimation_parameters[i], test=False)
            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale"])
            assert dist.parameters.get("loc") is not None
            assert dist.parameters.get("scale") is not None
