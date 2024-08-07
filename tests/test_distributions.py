"""Test distributions module."""

import matplotlib

matplotlib.use("Agg")
from typing import List, Dict

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
import pytest


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
        assert dist.parameters is None

    def test_create_instance_with_wrong_data_type(self):
        data = {"key": "value"}
        with pytest.raises(TypeError):
            dist = Gumbel(data=data)

    def test_create_instance_with_wrong_parameter_type(self):
        parameters = [1, 2, 3]
        with pytest.raises(TypeError):
            dist = Gumbel(parameters=parameters)

    def test_random(
        self,
        dist_estimation_parameters_ks: str,
        gum_dist_parameters: Dict[str, Dict[str, float]],
    ):
        # param = gum_dist_parameters[dist_estimation_parameters_ks]
        param = {"loc": 0, "scale": 1}
        dist = Gumbel(parameters=param)
        rv = dist.random(100)
        # new_dist = Gumbel(rv, parameters=param)
        assert isinstance(rv, np.ndarray)
        assert rv.shape == (100,)

    def test_fit_model(
        self,
        time_series2: list,
        dist_estimation_parameters: List[str],
        gum_dist_parameters: Dict[str, float],
    ):
        dist = Gumbel(time_series2)
        for method in dist_estimation_parameters:
            param = dist.fit_model(method=method, test=False)
            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale"])
            assert dist.parameters.get("loc") is not None
            assert dist.parameters.get("scale") is not None
            assert param == gum_dist_parameters[method]

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
        gum_dist_parameters: Dict[str, Dict[str, float]],
    ):
        param = gum_dist_parameters[dist_estimation_parameters_ks]
        dist = Gumbel(time_series2, param)
        dstatic, pvalue = dist.ks()
        assert dstatic == 0.07407407407407407
        assert pvalue == 0.9987375782247235

    def test_chisquare(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        gum_dist_parameters: Dict[str, Dict[str, float]],
    ):
        param = gum_dist_parameters[dist_estimation_parameters_ks]
        dist = Gumbel(time_series2, param)
        dstatic, pvalue = dist.chisquare()
        assert dstatic == -0.2813945052127964
        assert pvalue == 1

    def test_pdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        gum_dist_parameters: Dict[str, Dict[str, float]],
        gum_pdf: np.ndarray,
    ):
        param = gum_dist_parameters[dist_estimation_parameters_ks]
        dist = Gumbel(time_series2, param)
        pdf, fig, ax = dist.pdf(plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        np.testing.assert_almost_equal(gum_pdf, pdf)
        assert isinstance(fig, Figure)
        # test if you provide the pdf method with the data parameter
        pdf, fig, ax = dist.pdf(data=time_series2, plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        np.testing.assert_almost_equal(gum_pdf, pdf)

    def test_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        gum_dist_parameters: Dict[str, Dict[str, float]],
        gum_cdf: np.ndarray,
    ):
        param = gum_dist_parameters[dist_estimation_parameters_ks]
        dist = Gumbel(time_series2, param)
        cdf, fig, ax = dist.cdf(plot_figure=True)
        assert isinstance(cdf, np.ndarray)
        np.testing.assert_almost_equal(gum_cdf, cdf)
        assert isinstance(fig, Figure)
        # test if you provide the cdf method with the data parameter
        cdf, fig, ax = dist.cdf(data=time_series2, plot_figure=True)
        assert isinstance(cdf, np.ndarray)

    def test_inverse_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        gum_dist_parameters: Dict[str, Dict[str, float]],
        gev_inverse_cdf: np.ndarray,
        generated_cdf: List[float],
    ):
        param = gum_dist_parameters[dist_estimation_parameters_ks]
        dist = Gumbel(time_series2, param)
        qth = dist.inverse_cdf(generated_cdf)
        assert isinstance(qth, np.ndarray)
        np.testing.assert_almost_equal(gev_inverse_cdf, qth)

    def test_confidence_interval(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        confidence_interval_alpha: float,
        gum_dist_parameters: Dict[str, Dict[str, float]],
    ):
        dist = Gumbel(time_series2)
        cdf_weibul = PlottingPosition.weibul(time_series2)
        param = gum_dist_parameters[dist_estimation_parameters_ks]
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
        gum_dist_parameters: Dict[str, Dict[str, float]],
    ):
        dist = Gumbel(time_series2)
        cdf_weibul = PlottingPosition.weibul(time_series2)
        param = gum_dist_parameters[dist_estimation_parameters_ks]
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

    def test_gev_fit_model(
        self,
        time_series1: list,
        dist_estimation_parameters: List[str],
        gev_dist_parameters: Dict[str, str],
    ):
        dist = GEV(time_series1)
        for method in dist_estimation_parameters:
            param = dist.fit_model(method=method, test=False)

            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale", "shape"])
            assert dist.parameters.get("loc") is not None
            assert dist.parameters.get("scale") is not None
            assert dist.parameters.get("shape") is not None
            assert param == gev_dist_parameters[method]

    def test_gev_ks(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        gev_dist_parameters: Dict[str, Dict[str, float]],
    ):
        param = gev_dist_parameters[dist_estimation_parameters_ks]
        dist = GEV(time_series1, param)
        dstatic, pvalue = dist.ks()
        assert dstatic == 0.14814814814814814
        assert pvalue == 0.9356622290518453

    def test_gev_chisquare(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        gev_dist_parameters: Dict[str, Dict[str, float]],
    ):
        param = gev_dist_parameters[dist_estimation_parameters_ks]
        dist = GEV(time_series1, param)
        dstatic, pvalue = dist.chisquare()
        assert dstatic == -22.906818156545253
        assert pvalue == 1

    def test_gev_pdf(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        gev_dist_parameters: Dict[str, Dict[str, float]],
        gev_pdf: np.ndarray,
    ):
        param = gev_dist_parameters[dist_estimation_parameters_ks]
        dist = GEV(time_series1, param)

        pdf, fig, ax = dist.pdf(plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        np.testing.assert_almost_equal(gev_pdf, pdf)
        assert isinstance(fig, Figure)
        # test if you provide the pdf method with the data parameter
        pdf, fig, ax = dist.pdf(data=time_series1, plot_figure=True)
        assert isinstance(pdf, np.ndarray)

    def test_gev_cdf(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        gev_dist_parameters: Dict[str, Dict[str, float]],
        gev_cdf: np.ndarray,
    ):
        param = gev_dist_parameters[dist_estimation_parameters_ks]
        dist = GEV(time_series1, param)
        cdf, fig, ax = dist.cdf(plot_figure=True)
        assert isinstance(cdf, np.ndarray)
        np.testing.assert_almost_equal(gev_cdf, cdf)
        assert isinstance(fig, Figure)
        # test if you provide the cdf method with the data parameter
        cdf, fig, ax = dist.cdf(data=time_series1, plot_figure=True)
        assert isinstance(cdf, np.ndarray)

    def test_random(
        self,
        dist_estimation_parameters_ks: str,
        gum_dist_parameters: Dict[str, Dict[str, float]],
    ):
        # param = gum_dist_parameters[dist_estimation_parameters_ks]
        param = {"loc": 0, "scale": 1, "shape": 0.1}
        dist = Gumbel(parameters=param)
        rv = dist.random(100)
        # new_dist = Gumbel(rv, parameters=param)
        assert isinstance(rv, np.ndarray)
        assert rv.shape == (100,)

    def test_gev_inverse_cdf(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        gev_dist_parameters: Dict[str, Dict[str, float]],
        generated_cdf: List[float],
        gum_inverse_cdf: np.ndarray,
    ):
        param = gev_dist_parameters[dist_estimation_parameters_ks]
        dist = GEV(time_series1, param)
        qth = dist.inverse_cdf(generated_cdf)
        assert isinstance(qth, np.ndarray)
        np.testing.assert_almost_equal(gum_inverse_cdf, qth)

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
        gev_dist_parameters: Dict[str, Dict[str, float]],
    ):
        dist = GEV(time_series1)
        cdf_weibul = PlottingPosition.weibul(time_series1)
        param = gev_dist_parameters[dist_estimation_parameters_ks]
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

    def test_gev_probability_plot(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        confidence_interval_alpha: float,
        gev_dist_parameters: Dict[str, Dict[str, float]],
    ):
        dist = GEV(time_series1)
        cdf_weibul = PlottingPosition.weibul(time_series1)
        param = gev_dist_parameters[dist_estimation_parameters_ks]
        (fig1, fig2), (_, _) = dist.probability_plot(
            param, cdf_weibul, alpha=confidence_interval_alpha
        )
        assert isinstance(fig1, Figure)
        assert isinstance(fig2, Figure)


class TestExponential:
    def test_create_instance(
        self,
        time_series1: list,
    ):
        expo_dist = Exponential(time_series1)
        assert isinstance(expo_dist.data, np.ndarray)
        assert isinstance(expo_dist.data_sorted, np.ndarray)

    def test_fit_model(
        self,
        time_series2: list,
        dist_estimation_parameters: List[str],
        exp_dist_parameters: Dict[str, float],
    ):
        expo_dist = Exponential(time_series2)
        for method in dist_estimation_parameters:
            param = expo_dist.fit_model(method=method, test=False)
            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale"])
            assert expo_dist.parameters.get("loc") is not None
            assert expo_dist.parameters.get("scale") is not None
            assert param == exp_dist_parameters[method]

    def test_pdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        exp_dist_parameters: Dict[str, Dict[str, float]],
        exp_pdf: np.ndarray,
    ):
        param = exp_dist_parameters[dist_estimation_parameters_ks]
        expo_dist = Exponential(time_series2, param)
        pdf, fig, ax = expo_dist.pdf(plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        np.testing.assert_almost_equal(exp_pdf, pdf)
        assert isinstance(fig, Figure)
        # test if you provide the pdf method with the data parameter
        pdf, fig, ax = expo_dist.pdf(data=time_series2, plot_figure=True)
        assert isinstance(pdf, np.ndarray)

    def test_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        exp_dist_parameters: Dict[str, Dict[str, float]],
        exp_cdf: np.ndarray,
    ):
        param = exp_dist_parameters[dist_estimation_parameters_ks]
        expo_dist = Exponential(time_series2, param)
        cdf, fig, ax = expo_dist.cdf(plot_figure=True)
        assert isinstance(cdf, np.ndarray)
        np.testing.assert_almost_equal(exp_cdf, cdf)
        assert isinstance(fig, Figure)
        # test if you provide the cdf method with the data parameter
        cdf, fig, ax = expo_dist.cdf(data=time_series2, plot_figure=True)
        assert isinstance(cdf, np.ndarray)

    def test_inverse_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        exp_dist_parameters: Dict[str, Dict[str, float]],
        generated_cdf: List[float],
        exp_inverse_cdf: np.ndarray,
    ):
        param = exp_dist_parameters[dist_estimation_parameters_ks]
        expo_dist = Exponential(time_series2, param)
        qth = expo_dist.inverse_cdf(generated_cdf)
        assert isinstance(qth, np.ndarray)
        np.testing.assert_almost_equal(exp_inverse_cdf, qth)


class TestNormal:
    def test_create_instance(
        self,
        time_series1: list,
    ):
        norm_dist = Normal(time_series1)
        assert isinstance(norm_dist.data, np.ndarray)
        assert isinstance(norm_dist.data_sorted, np.ndarray)

    def test_fit_model(
        self,
        time_series2: list,
        dist_estimation_parameters: List[str],
        normal_dist_parameters: Dict[str, Dict[str, float]],
    ):
        norm_dist = Normal(time_series2)
        for method in dist_estimation_parameters:
            param = norm_dist.fit_model(method=method, test=False)
            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale"])
            assert norm_dist.parameters.get("loc") is not None
            assert norm_dist.parameters.get("scale") is not None
            assert param == normal_dist_parameters[method]

    def test_pdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        normal_dist_parameters: Dict[str, Dict[str, float]],
        normal_pdf: np.ndarray,
    ):
        param = normal_dist_parameters[dist_estimation_parameters_ks]
        norm_dist = Normal(time_series2, param)
        pdf, fig, ax = norm_dist.pdf(plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        np.testing.assert_almost_equal(normal_pdf, pdf)
        assert isinstance(fig, Figure)
        # test if you provide the pdf method with the data parameter
        pdf, fig, ax = norm_dist.pdf(data=time_series2, plot_figure=True)
        assert isinstance(pdf, np.ndarray)

    def test_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        normal_dist_parameters: Dict[str, Dict[str, float]],
        normal_cdf: np.ndarray,
    ):
        param = normal_dist_parameters[dist_estimation_parameters_ks]
        norm_dist = Normal(time_series2, param)
        cdf, fig, ax = norm_dist.cdf(plot_figure=True)
        assert isinstance(cdf, np.ndarray)
        np.testing.assert_almost_equal(normal_cdf, cdf)
        assert isinstance(fig, Figure)
        # test if you provide the cdf method with the data parameter
        cdf, fig, ax = norm_dist.cdf(data=time_series2, plot_figure=True)
        assert isinstance(cdf, np.ndarray)

    def test_inverse_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        normal_dist_parameters: Dict[str, Dict[str, float]],
        generated_cdf: List[float],
        normal_inverse_cdf: np.ndarray,
    ):
        param = normal_dist_parameters[dist_estimation_parameters_ks]
        norm_dist = Normal(time_series2, param)
        qth = norm_dist.inverse_cdf(generated_cdf)
        assert isinstance(qth, np.ndarray)
        np.testing.assert_almost_equal(normal_inverse_cdf, qth)


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
