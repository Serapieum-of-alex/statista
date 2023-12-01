import pandas as pd

rdir = rf"\\MYCLOUDEX2ULTRA\research\phd\heavy-tail-statistics"
from statista.distributions import GEV

#%%
dung = pd.read_csv(f"{rdir}/dung/pdf_obs.txt", delimiter=" ")
dung.sort_values(by="ams", inplace=True)
scale = 2132.938715
loc = 6582.059315
shape = 0.0486556
ams = dung["ams"].values


dist_positive = GEV(ams, shape, loc, scale)
pdf = dist_positive.pdf(shape, loc, scale, plot_figure=False)
dung["scipy +ve"] = pdf

shape = -0.0486556
dist_negative = GEV(ams, shape, loc, scale)
pdf = dist_negative.pdf(shape, loc, scale, plot_figure=False)
dung["scipy -ve"] = pdf
#%%
method = "lmoments"  # "mle"
parameters_lm = dist_negative.fit_model(method=method)
parameters_mle = dist_negative.fit_model(method="mle")
#%%
