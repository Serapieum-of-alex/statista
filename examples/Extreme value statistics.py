"""Extreme value statistics"""
import matplotlib

matplotlib.use("TkAgg")
import pandas as pd

from statista.distributions import GEV, Gumbel, PlottingPosition
from statista.confidence_interval import ConfidenceInterval

time_series1 = pd.read_csv("examples/data/time_series1.txt", header=None)[0].tolist()
time_series2 = pd.read_csv("examples/data/time_series2.txt", header=None)[0].tolist()
# %%
gumbel_dist = Gumbel(time_series1)
# defult parameter estimation method is maximum liklihood method
param_mle = gumbel_dist.fit_model(method="mle")
gumbel_dist.ks()
gumbel_dist.chisquare()
print(param_mle)
# calculate and plot the pdf
pdf = gumbel_dist.pdf(param_mle, plot_figure=True)
cdf, _, _ = gumbel_dist.cdf(param_mle, plot_figure=True)
# %% lmoments
param_lmoments = gumbel_dist.fit_model(method="lmoments")
gumbel_dist.ks()
gumbel_dist.chisquare()
print(param_lmoments)
# calculate and plot the pdf
pdf = gumbel_dist.pdf(param_lmoments, plot_figure=True)
cdf, _, _ = gumbel_dist.cdf(param_lmoments, plot_figure=True)
# %%
# calculate the CDF(Non Exceedance probability) using weibul plotting position
time_series1.sort()
# calculate the F (Non Exceedence probability based on weibul)
cdf_weibul = PlottingPosition.weibul(time_series1)
# TheporeticalEstimate method calculates the theoretical values based on the Gumbel distribution
Qth = gumbel_dist.theoretical_estimate(param_lmoments, cdf_weibul)
# test = stats.chisquare(st.Standardize(Qth), st.Standardize(time_series1),ddof=5)
# calculate the confidence interval
upper, lower = gumbel_dist.confidence_interval(param_lmoments, cdf_weibul, alpha=0.1)
# ProbapilityPlot can estimate the Qth and the lower and upper confidence interval in the process of plotting
fig, ax = gumbel_dist.probapility_plot(param_lmoments, cdf_weibul, alpha=0.1)
# %%
"""
if you want to focus only on high values, you can use a threshold to make the code focus on what is higher
this threshold.
"""
threshold = 17
param_dist = gumbel_dist.fit_model(
    method="optimization", obj_func=Gumbel.objective_fn, threshold=threshold
)
print(param_dist)
gumbel_dist.probapility_plot(param_dist, cdf_weibul, alpha=0.1)
# %%
threshold = 18
param_dist = gumbel_dist.fit_model(
    method="optimization", obj_func=Gumbel.objective_fn, threshold=threshold
)
print(param_dist)
gumbel_dist.probapility_plot(param_dist, cdf_weibul, alpha=0.1)
# %% Generalized Extreme Value (GEV)
gev_dist = GEV(time_series2)
# default parameter estimation method is maximum liklihood method
gev_mle_param = gev_dist.fit_model(method="mle")
gev_dist.ks()
gev_dist.chisquare()

print(gev_mle_param)
# calculate and plot the pdf
pdf, fig, ax = gev_dist.pdf(gev_mle_param, plot_figure=True)
cdf, _, _ = gev_dist.cdf(gev_mle_param, plot_figure=True)
# %% lmoment method
gev_lmom_param = gev_dist.fit_model(method="lmoments")
print(gev_lmom_param)
# calculate and plot the pdf
pdf, fig, ax = gev_dist.pdf(gev_lmom_param, plot_figure=True)
cdf, _, _ = gev_dist.cdf(gev_lmom_param, plot_figure=True)
#%%
time_series1.sort()
# calculate the F (Non Exceedence probability based on weibul)
cdf_weibul = PlottingPosition.weibul(time_series1)
T = PlottingPosition.weibul(time_series1, return_period=True)
# TheporeticalEstimate method calculates the theoretical values based on the Gumbel distribution
Qth = gev_dist.theoretical_estimate(gev_lmom_param, cdf_weibul)

func = GEV.ci_func
upper, lower = gev_dist.confidence_interval(
    gev_lmom_param,
    prob_non_exceed=cdf_weibul,
    alpha=0.1,
    statfunction=func,
    n_samples=len(time_series1),
    method="lmoments",
)
# %%
"""
calculate the confidence interval using the boot strap method directly
"""
CI = ConfidenceInterval.boot_strap(
    time_series1,
    statfunction=func,
    gevfit=gev_lmom_param,
    n_samples=len(time_series1),
    F=cdf_weibul,
    method="lmoments",
)
LB = CI["lb"]
UB = CI["ub"]
# %%
fig, ax = gev_dist.probapility_plot(
    gev_lmom_param, cdf_weibul, func=func, n_samples=len(time_series1)
)
