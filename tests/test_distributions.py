from typing import List

import numpy as np
from matplotlib.figure import Figure

from statista.distributions import (
    GEV,
    ConfidenceInterval,
    Gumbel,
    PlottingPosition,
    Exponential,
    Normal,
)


class TestPlottingPosition:
    def test_plotting_position_weibul(
        self,
        time_series1: list,
    ):
        cdf = PlottingPosition.weibul(time_series1, return_period=False)
        assert isinstance(cdf, np.ndarray)
        rp = PlottingPosition.weibul(time_series1, return_period=True)
        assert isinstance(rp, np.ndarray)

    def test_plotting_position_rp(
        self,
        time_series1: list,
    ):
        cdf = PlottingPosition.weibul(time_series1, return_period=False)
        rp = PlottingPosition.return_period(cdf)
        assert isinstance(rp, np.ndarray)


class TestGumbel:
    def test_create_instance(
        self,
        time_series1: list,
    ):
        Gdist = Gumbel(time_series1)
        assert isinstance(Gdist.data, np.ndarray)
        assert isinstance(Gdist.data_sorted, np.ndarray)

    def test_estimate_parameter(
        self,
        time_series2: list,
        dist_estimation_parameters: List[str],
    ):
        Gdist = Gumbel(time_series2)
        for i in range(len(dist_estimation_parameters)):
            param = Gdist.estimate_parameter(
                method=dist_estimation_parameters[i], test=False
            )
            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale"])
            assert Gdist.parameters.get("loc") is not None
            assert Gdist.parameters.get("scale") is not None

    def test_parameter_estimation_optimization(
        self,
        time_series2: list,
        dist_estimation_parameters: List[str],
        parameter_estimation_optimization_threshold: int,
    ):
        Gdist = Gumbel(time_series2)
        param = Gdist.estimate_parameter(
            method="optimization",
            ObjFunc=Gumbel.ObjectiveFn,
            threshold=parameter_estimation_optimization_threshold,
        )
        assert isinstance(param, dict)
        assert all(i in param.keys() for i in ["loc", "scale"])
        assert Gdist.parameters.get("loc") is not None
        assert Gdist.parameters.get("scale") is not None

    def test_ks(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        Gdist = Gumbel(time_series2)
        Gdist.estimate_parameter(method=dist_estimation_parameters_ks, test=False)
        Gdist.ks()
        assert Gdist.Dstatic
        assert Gdist.KS_Pvalue

    def test_chisquare(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        Gdist = Gumbel(time_series2)
        Gdist.estimate_parameter(method=dist_estimation_parameters_ks, test=False)
        Gdist.chisquare()
        assert Gdist.chistatic
        assert Gdist.chi_Pvalue

    def test_pdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        Gdist = Gumbel(time_series2)
        Param = Gdist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )

        pdf, fig, ax = Gdist.pdf(Param, plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        Gdist = Gumbel(time_series2)
        Param = Gdist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )
        cdf, fig, ax = Gdist.cdf(Param, plot_figure=True)

        assert isinstance(cdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_gumbel_TheporeticalEstimate(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        Gdist = Gumbel(time_series2)
        cdf_Weibul = PlottingPosition.weibul(time_series2)
        Param = Gdist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )
        Qth = Gdist.theoretical_estimate(Param, cdf_Weibul)
        assert isinstance(Qth, np.ndarray)

    def test_gumbel_confidence_interval(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        confidence_interval_alpha: float,
    ):
        Gdist = Gumbel(time_series2)
        cdf_Weibul = PlottingPosition.weibul(time_series2)
        Param = Gdist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )
        upper, lower = Gdist.confidence_interval(
            Param, cdf_Weibul, alpha=confidence_interval_alpha
        )
        assert isinstance(upper, np.ndarray)
        assert isinstance(lower, np.ndarray)

    def test_gumbel_probapility_plot(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
        confidence_interval_alpha: float,
    ):
        Gdist = Gumbel(time_series2)
        cdf_Weibul = PlottingPosition.weibul(time_series2)
        Param = Gdist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )
        [fig1, fig2], [ax1, ax2] = Gdist.probapility_plot(
            Param, cdf_Weibul, alpha=confidence_interval_alpha
        )
        assert isinstance(fig1, Figure)
        assert isinstance(fig2, Figure)


class TestGEV:
    def test_create_gev_instance(
        self,
        time_series1: list,
    ):
        Gdist = GEV(time_series1)
        assert isinstance(Gdist.data, np.ndarray)
        assert isinstance(Gdist.data_sorted, np.ndarray)

    def test_gev_estimate_parameter(
        self,
        time_series1: list,
        dist_estimation_parameters: List[str],
    ):
        Gdist = GEV(time_series1)
        for i in range(len(dist_estimation_parameters)):
            param = Gdist.estimate_parameter(
                method=dist_estimation_parameters[i], test=False
            )

            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale", "shape"])
            assert Gdist.parameters.get("loc") is not None
            assert Gdist.parameters.get("scale") is not None
            assert Gdist.parameters.get("shape") is not None

    def test_gev_ks(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
    ):
        Gdist = GEV(time_series1)
        Gdist.estimate_parameter(method=dist_estimation_parameters_ks, test=False)
        Gdist.ks()
        assert Gdist.Dstatic
        assert Gdist.KS_Pvalue

    def test_gev_chisquare(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
    ):
        Gdist = GEV(time_series1)
        Gdist.estimate_parameter(method=dist_estimation_parameters_ks, test=False)
        Gdist.chisquare()
        assert Gdist.chistatic
        assert Gdist.chi_Pvalue

    def test_gev_pdf(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
    ):
        Gdist = GEV(time_series1)
        Param = Gdist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )

        pdf, fig, ax = Gdist.pdf(Param, plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_gev_cdf(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
    ):
        Gdist = GEV(time_series1)
        Param = Gdist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )
        cdf, fig, ax = Gdist.cdf(Param, plot_figure=True)
        assert isinstance(cdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_gev_TheporeticalEstimate(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
    ):
        Gdist = GEV(time_series1)
        cdf_Weibul = PlottingPosition.weibul(time_series1)
        Param = Gdist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )
        Qth = Gdist.theoretical_estimate(Param, cdf_Weibul)
        assert isinstance(Qth, np.ndarray)

    def test_gev_confidence_interval(
        self,
        time_series1: list,
        dist_estimation_parameters_ks: str,
        confidence_interval_alpha: float,
    ):
        Gdist = GEV(time_series1)
        cdf_Weibul = PlottingPosition.weibul(time_series1)
        Param = Gdist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )

        func = GEV.ci_func
        upper, lower = Gdist.confidence_interval(
            Param,
            F=cdf_Weibul,
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
        Gdist = GEV(time_series1)
        cdf_Weibul = PlottingPosition.weibul(time_series1)
        Param = Gdist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )

        func = GEV.ci_func

        CI = ConfidenceInterval.BootStrap(
            time_series1,
            statfunction=func,
            gevfit=Param,
            n_samples=len(time_series1),
            F=cdf_Weibul,
            method="lmoments",
        )
        LB = CI["LB"]
        UB = CI["UB"]

        assert isinstance(LB, np.ndarray)
        assert isinstance(UB, np.ndarray)


# class TestAbstractDistrition:


class TestExponential:
    def test_create_instance(
        self,
        time_series1: list,
    ):
        Edist = Exponential(time_series1)
        assert isinstance(Edist.data, np.ndarray)
        assert isinstance(Edist.data_sorted, np.ndarray)

    def test_estimate_parameter(
        self,
        time_series2: list,
        dist_estimation_parameters: List[str],
    ):
        Edist = Exponential(time_series2)
        for i in range(len(dist_estimation_parameters)):
            param = Edist.estimate_parameter(
                method=dist_estimation_parameters[i], test=False
            )
            assert isinstance(param, dict)
            assert all(i in param.keys() for i in ["loc", "scale"])
            assert Edist.parameters.get("loc") is not None
            assert Edist.parameters.get("scale") is not None

    def test_pdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        Edist = Exponential(time_series2)
        Param = Edist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )
        pdf, fig, ax = Edist.pdf(Param, plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        Edist = Exponential(time_series2)
        Param = Edist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )
        cdf, fig, ax = Edist.cdf(Param, plot_figure=True)
        assert isinstance(cdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_TheporeticalEstimate(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        Edist = Exponential(time_series2)
        cdf_Weibul = PlottingPosition.weibul(time_series2)
        Param = Edist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )
        Qth = Edist.theoretical_estimate(Param, cdf_Weibul)
        assert isinstance(Qth, np.ndarray)


class TestNormal:
    def test_create_instance(
        self,
        time_series1: list,
    ):
        Edist = Normal(time_series1)
        assert isinstance(Edist.data, np.ndarray)
        assert isinstance(Edist.data_sorted, np.ndarray)

    def test_estimate_parameter(
        self,
        time_series2: list,
        dist_estimation_parameters: List[str],
    ):
        Edist = Normal(time_series2)
        for method in dist_estimation_parameters:
            param = Edist.estimate_parameter(method=method, test=False)
            assert isinstance(param, list)
            assert Edist.loc
            assert Edist.scale

    def test_pdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        Edist = Normal(time_series2)
        Param = Edist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )
        pdf, fig, ax = Edist.pdf(Param[0], Param[1], plot_figure=True)
        assert isinstance(pdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_cdf(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        Edist = Normal(time_series2)
        Param = Edist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )
        cdf, fig, ax = Edist.cdf(Param[0], Param[1], plot_figure=True)
        assert isinstance(cdf, np.ndarray)
        assert isinstance(fig, Figure)

    def test_TheporeticalEstimate(
        self,
        time_series2: list,
        dist_estimation_parameters_ks: str,
    ):
        Edist = Normal(time_series2)
        cdf_Weibul = PlottingPosition.weibul(time_series2)
        Param = Edist.estimate_parameter(
            method=dist_estimation_parameters_ks, test=False
        )
        Qth = Edist.theoretical_estimate(Param[0], Param[1], cdf_Weibul)
        assert isinstance(Qth, np.ndarray)
