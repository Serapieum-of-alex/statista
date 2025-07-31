# import os
Path = "F:/algorithms/Hydrology/HAPI/examples"
import matplotlib

matplotlib.use("TkAgg")
# functions
import Hapi.rrm.hbv_bergestrom92 as HBVLumped
import Hapi.sm.performancecriteria as PC
import pandas as pd
from Hapi.catchment import Catchment
from Hapi.rrm.routing import Routing
from Hapi.run import Run
from Hapi.sm.sensitivityanalysis import SensitivityAnalysis as SA

Parameterpath = Path + "/data/Lumped/Coello_Lumped2021-03-08_muskingum.txt"
Path = Path + "/data/Lumped/"
# %% meteorological data
start = "2009-01-01"
end = "2011-12-31"
name = "Coello"
Coello = Catchment(name, start, end)
Coello.ReadLumpedInputs(Path + "meteo_data-MSWEP.csv")

# %% Basic_inputs
# catchment area
CatArea = 1530
# temporal resolution
# [Snow pack, Soil moisture, Upper zone, Lower Zone, Water content]
InitialCond = [0, 10, 10, 10, 0]

Coello.ReadLumpedModel(HBVLumped, CatArea, InitialCond)

### parameters
# no snow subroutine
Snow = 0
# if routing using Maxbas True, if Muskingum False
Maxbas = False
Coello.ReadParameters(Parameterpath, Snow, Maxbas=Maxbas)

parameters = pd.read_csv(Parameterpath, index_col=0, header=None)
parameters.rename(columns={1: "value"}, inplace=True)

UB = pd.read_csv(Path + "/UB-1-Muskinguk.txt", index_col=0, header=None)
parnames = UB.index
UB = UB[1].tolist()
LB = pd.read_csv(Path + "/LB-1-Muskinguk.txt", index_col=0, header=None)
LB = LB[1].tolist()
Coello.ReadParametersBounds(UB, LB, Snow)

# observed flow
Coello.ReadDischargeGauges(Path + "Qout_c.csv", fmt="%Y-%m-%d")
### Routing
Route = 1
# RoutingFn=Routing.TriangularRouting2
RoutingFn = Routing.Muskingum
# %%
### run the model
Run.RunLumped(Coello, Route, RoutingFn)
# %%
Metrics = dict()

Qobs = Coello.QGauges[Coello.QGauges.columns[0]]

Metrics["RMSE"] = PC.rmse(Qobs, Coello.Qsim["q"])
Metrics["NSE"] = PC.nse(Qobs, Coello.Qsim["q"])
Metrics["NSEhf"] = PC.nse_hf(Qobs, Coello.Qsim["q"])
Metrics["KGE"] = PC.kge(Qobs, Coello.Qsim["q"])
Metrics["WB"] = PC.wb(Qobs, Coello.Qsim["q"])

print("RMSE= " + str(round(Metrics["RMSE"], 2)))
print("NSE= " + str(round(Metrics["NSE"], 2)))
print("NSEhf= " + str(round(Metrics["NSEhf"], 2)))
print("KGE= " + str(round(Metrics["KGE"], 2)))
print("WB= " + str(round(Metrics["WB"], 2)))
# %%
"""
first the Sensitivity method takes 4 arguments :
    1-parameters:previous obtained parameters
    2-LB: upper bound
    3-UB: lower bound
    4-wrapper: defined function contains the function you want to run with different
        parameters and the metric function you want to assess the first function
        based on it.

## Wrapper function definition
    define the function to the OAT sesitivity wrapper and put the parameters argument
    at the first position, and then list all the other arguments required for your function

    the following defined function contains two inner function that calculates discharge
    for lumped HBV model and calculates the RMSE of the calculated discharge.

    the first function "RUN.RunLumped" takes some arguments we need to pass it through
    the Sensitivity method [ConceptualModel,data,p2,init_st,snow,Routing, RoutingFn]
    with the same order in the defined function "wrapper"

    the second function is RMSE takes the calculated discharge from the first function
    and measured discharge array

    to define the argument of the "wrapper" function
    1- the random parameters valiable i=of the first function should be the first argument
        "wrapper(Randpar)"
    2- the first function arguments with the same order (except that the parameter
            argument is taken out and placed at the first potition step-1)
    3- list the argument of the second function with the same order that the second
    function takes them

Sensitivity method returns a dictionary with the name of the parameters
as keys,
Each parameter has a disctionary with two keys 0: list of parameters woth relative values
1: list of parameter values
"""


# For Type 1
def WrapperType1(Randpar, Route, RoutingFn, Qobs):
    Coello.Parameters = Randpar

    Run.RunLumped(Coello, Route, RoutingFn)
    rmse = PC.rmse(Qobs, Coello.Qsim["q"])
    return rmse


# For Type 2
def WrapperType2(Randpar, Route, RoutingFn, Qobs):
    Coello.Parameters = Randpar

    Run.RunLumped(Coello, Route, RoutingFn)
    rmse = PC.rmse(Qobs, Coello.Qsim["q"])
    return rmse, Coello.Qsim["q"]


Type = 2
if Type == 1:
    fn = WrapperType1
elif Type == 2:
    fn = WrapperType2

Positions = [10]


Sen = SA(
    parameters, Coello.lower_bound, Coello.upper_bound, fn, Positions, 5, Type=Type
)
Sen.one_at_a_time(Route, RoutingFn, Qobs)
# %%
From = ""
To = ""
if Type == 1:
    fig, ax1 = Sen.sobol(
        real_values=False,
        title="Sensitivity Analysis of the RMSE to models parameters",
        xlabel="Maxbas Values",
        ylabel="RMSE",
        plotting_from=From,
        plotting_to=To,
        xlabel2="time",
        ylabel2="Discharge m3/s",
        spaces=[None, None, None, None, None, None],
    )
elif Type == 2:
    fig, (ax1, ax2) = Sen.sobol(
        real_values=False,
        title="Sensitivity Analysis of the RMSE to models parameters",
        xlabel="Maxbas Values",
        ylabel="RMSE",
        plotting_from=From,
        plotting_to=To,
        xlabel2="time",
        ylabel2="Discharge m3/s",
        spaces=[None, None, None, None, None, None],
    )
    From = 0
    To = len(Qobs.values)
    ax2.plot(Qobs.values[From:To], label="Observed", color="red")
