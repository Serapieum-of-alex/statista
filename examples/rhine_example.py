import matplotlib

matplotlib.use("TkAgg")
import numpy as np
import pandas as pd
from statista.distributions import (
    GEV,
    Exponential,
    ConfidenceInterval,
    Gumbel,
    PlottingPosition,
)

ams = pd.read_csv("examples/data/rhine.csv")
ams.head()
ams.replace(0, np.nan, inplace=True)
ams.dropna(axis=0, inplace=True)
#%%
rees_gauge = ams.loc[:, "rees"].values
cologne_gauge = ams.loc[:, "cologne"].values
maxau_gauge = ams.loc[:, "maxau"].values
rockenau_gauge = ams.loc[:, "rockenau"].values
#%% Exponential distribution (mle)
dist_obj = Exponential(cologne_gauge)
# default parameter estimation method is maximum liklihood method
mle_param = dist_obj.estimateParameter(method="mle")
dist_obj.ks()
dist_obj.chisquare()

print(mle_param)
loc = mle_param[0]
scale = mle_param[1]
# calculate and plot the pdf
pdf, fig, ax = dist_obj.pdf(loc, scale, plot_figure=True)
cdf, _, _ = dist_obj.cdf(loc, scale, plot_figure=True)
#%% exponential distribution (lmoments)
dist_obj = Exponential(cologne_gauge)
# default parameter estimation method is maximum liklihood method
mle_param = dist_obj.estimateParameter(method="lmoments")
dist_obj.ks()
dist_obj.chisquare()

print(mle_param)
loc = mle_param[0]
scale = mle_param[1]
# calculate and plot the pdf
pdf, fig, ax = dist_obj.pdf(loc, scale, plot_figure=True)
cdf, _, _ = dist_obj.cdf(loc, scale, plot_figure=True)
#%% GEV (mle)
gev_cologne = GEV(cologne_gauge)
# default parameter estimation method is maximum liklihood method
mle_param = gev_cologne.estimateParameter(method="mle")
gev_cologne.ks()
gev_cologne.chisquare()

print(mle_param)
# shape = -1 * mle_param[0]
shape = mle_param[0]
loc = mle_param[1]
scale = mle_param[2]
# calculate and plot the pdf
pdf, fig, ax = gev_cologne.pdf(shape, loc, scale, plot_figure=True)
cdf, _, _ = gev_cologne.cdf(shape, loc, scale, plot_figure=True)
#%% cologne (lmoment)
gev_cologne = GEV(cologne_gauge)
# default parameter estimation method is maximum liklihood method
lmom_param = gev_cologne.estimateParameter(method="lmoments")
gev_cologne.ks()
gev_cologne.chisquare()

print(lmom_param)
# shape = -1 * `lmom_param[0]
shape = lmom_param[0]
loc = lmom_param[1]
scale = lmom_param[2]
# calculate and plot the pdf
pdf, fig, ax = gev_cologne.pdf(shape, loc, scale, plot_figure=True)
cdf, _, _ = gev_cologne.cdf(shape, loc, scale, plot_figure=True)
