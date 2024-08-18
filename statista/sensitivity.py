"""Sensitivity Analysis."""

from typing import List, Union
import matplotlib.pyplot as plt
import numpy as np
from pandas import DataFrame


class Sensitivity:
    """Sensitivity.

    Sensitivity class
    """

    MarkerStyleList = [
        "--o",
        ":D",
        "-.H",
        "--x",
        ":v",
        "--|",
        "-+",
        "-^",
        "--s",
        "-.*",
        "-.h",
    ]

    def __init__(
        self,
        parameter: DataFrame,
        lower_bound: List[Union[int, float]],
        upper_bound: List[Union[int, float]],
        function: callable,
        positions=None,
        n_values=5,
        return_values=1,
    ):
        """Sensitivity.

        plotting_to instantiate the Sensitivity class you have to provide the
        following parameters

        Parameters
        ----------
        parameter: [dataframe]
            dataframe with the index as the name of the parameters and one column
            with the name "value" contains the values of the parameters.
        lower_bound: [list]
            lower bound of the parameter.
        upper_bound: [list]
            upper bound of the parameter.
        function: Callable
            DESCRIPTION.
        positions: [list], optional
            position of the parameter in the list (the beginning of the list starts
            with 0), if the Position argument is an empty list, the sensitivity will
            be done for all parameters. The default is None.
        n_values: [integer], optional
            number of parameter values between the bounds you want to calculate the
            metric for, if the values do not include the value if the given parameter
            it will be appended to the values. The default is 5.
        return_values: [integer], optional
            return_values equals 1 if the function returns one value (the measured metric)
            return_values equals 2 if the function returns two values (the measured metric,
            and any calculated values you want to check how they change by changing
            the value of the parameter). The default is 1.

        Returns
        -------
        None.
        """
        self.parameter = parameter
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        assert (
            len(self.parameter) == len(self.lower_bound) == len(self.upper_bound)
        ), "The Length of the boundary should be of the same length as the length of the parameters"
        assert callable(
            function
        ), "function should be of type-callable (function that takes arguments)"
        self.function = function

        self.NoValues = n_values
        self.return_values = return_values
        # if the Position argument is empty list, the sensitivity will be done for all parameters
        if positions is None:
            self.num_parameters = len(parameter)
            self.positions = list(range(len(parameter)))
        else:
            self.num_parameters = len(positions)
            self.positions = positions

    @staticmethod
    def marker_style(style):
        """MarkerStyle.

        Marker styles for plotting

        Parameters
        ----------
        style : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        if style > len(Sensitivity.MarkerStyleList) - 1:
            style %= len(Sensitivity.MarkerStyleList)
        return Sensitivity.MarkerStyleList[style]

    def one_at_a_time(self, *args, **kwargs):
        """OAT.

        OAT one-at-a-time sensitivity analysis.

        Parameters
        ----------
        *args: [positional argument]
            arguments of the function with the same exact names inside the function.
        **kwargs
            keyword arguments of the function with the same exact names inside the function.

            parameter: [dataframe]
                parameters dataframe including the parameter values in a column with
                name 'value' and the parameters' name as index.
            LB: [List]
                parameters upper bounds.
            UB: [List]
                parameters lower bounds.
            function: [function]
                the function you want to run it several times.

        Returns
        -------
        sen: [Dict]
            for each parameter as a key, there is a list containing 4 lists,
            1-relative parameter values, 2-metric values, 3-Real parameter values
            4- addition calculated values from the function if you choose return_values=2.
        """
        self.sen = {}

        for i in range(self.num_parameters):
            k = self.positions[i]
            if self.return_values == 1:
                self.sen[self.parameter.index[k]] = [[], [], []]
            else:
                self.sen[self.parameter.index[k]] = [[], [], [], []]
            # generate 5 random values between the high and low parameter bounds
            rand_value = np.linspace(
                self.lower_bound[k], self.upper_bound[k], self.NoValues
            )
            # add the value of the calibrated parameter and sort the values
            rand_value = np.sort(np.append(rand_value, self.parameter["value"][k]))
            # store the relative values of the parameters in the first list in the dict
            self.sen[self.parameter.index[k]][0] = [
                (h / self.parameter["value"][k]) for h in rand_value
            ]

            random_param = self.parameter["value"].tolist()
            for j in range(len(rand_value)):
                random_param[k] = rand_value[j]
                # args = list(args)
                # args.insert(Position, random_param)
                if self.return_values == 1:
                    metric = self.function(random_param, *args, **kwargs)
                else:
                    metric, calculated_values = self.function(
                        random_param, *args, **kwargs
                    )
                    self.sen[self.parameter.index[k]][3].append(calculated_values)
                try:
                    # store the metric value in the second list in the dict
                    self.sen[self.parameter.index[k]][1].append(round(metric, 3))
                except TypeError:
                    message = """the Given function returns more than one value,
                    the function should return only one value for return_values=1, or
                    two values for return_values=2.
                    """
                    raise ValueError(message)
                # store the real values of the parameter in the third list in the dict
                self.sen[self.parameter.index[k]][2].append(round(rand_value[j], 4))
                print(str(k) + "-" + self.parameter.index[k] + " -" + str(j))
                print(round(metric, 3))

    def sobol(
        self,
        real_values: bool = False,
        title: str = "",  # CalculatedValues=False,
        xlabel: str = "xlabel",
        ylabel: str = "Metric values",
        labelfontsize=12,
        plotting_from="",
        plotting_to="",
        title2: str = "",
        xlabel2: str = "xlabel2",
        ylabel2: str = "ylabel2",
        spaces=List[float],
    ):
        """Sobol.

        Parameters
        ----------
        real_values: [bool], optional
            if you want to plot the real values in the x-axis not the relative
            values, works properly only if you are checking the sensitivity of
            one parameter as the range of parameters differes. The default is False.
        CalculatedValues: [bool], optional
            if you choose return_values=2 in the OAT method, then the function returns
            calculated values, and here you can True to plot it . The default is False.
        title: [string], optional
            DESCRIPTION. The default is ''.
        xlabel: [string], optional
            DESCRIPTION. The default is 'xlabel'.
        ylabel: [string], optional
            DESCRIPTION. The default is 'Metric values'.
        labelfontsize: [integer], optional
            DESCRIPTION. The default is 12.
        plotting_from : TYPE, optional
            the calculated values are in array type and From attribute is from
            where the plotting will start. The default is ''.
        plotting_to: TYPE, optional
            the calculated values are in array type and plotting_to attribute is from
            where the plotting will end. The default is ''.
        title2: TYPE, optional
            DESCRIPTION. The default is ''.
        xlabel2: TYPE, optional
            DESCRIPTION. The default is 'xlabel2'.
        ylabel2: TYPE, optional
            DESCRIPTION. The default is 'ylabel2'.
        spaces: TYPE, optional
            DESCRIPTION. The default is [None, None, None, None, None, None].

        Returns
        -------
        """
        if self.return_values == 1:
            fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(8, 6))

            for i in range(self.num_parameters):
                k = self.positions[i]
                if real_values:
                    ax.plot(
                        self.sen[self.parameter.index[k]][2],
                        self.sen[self.parameter.index[k]][1],
                        Sensitivity.marker_style(k),
                        linewidth=3,
                        markersize=10,
                        label=self.parameter.index[k],
                    )
                else:
                    ax.plot(
                        self.sen[self.parameter.index[k]][0],
                        self.sen[self.parameter.index[k]][1],
                        Sensitivity.marker_style(k),
                        linewidth=3,
                        markersize=10,
                        label=self.parameter.index[k],
                    )

            ax.set_title(title, fontsize=12)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)

            ax.tick_params(axis="both", which="major", labelsize=labelfontsize)

            ax.legend(fontsize=12)
            plt.tight_layout()
            return fig, ax
        else:  # self.return_values == 2 and CalculatedValues
            try:
                fig, (ax1, ax2) = plt.subplots(ncols=1, nrows=2, figsize=(8, 6))

                for i in range(self.num_parameters):
                    # for i in range(len(self.sen[self.parameter.index[0]][0])):
                    k = self.positions[i]
                    if real_values:
                        ax1.plot(
                            self.sen[self.parameter.index[k]][2],
                            self.sen[self.parameter.index[k]][1],
                            Sensitivity.marker_style(k),
                            linewidth=3,
                            markersize=10,
                            label=self.parameter.index[k],
                        )
                    else:
                        ax1.plot(
                            self.sen[self.parameter.index[k]][0],
                            self.sen[self.parameter.index[k]][1],
                            Sensitivity.marker_style(k),
                            linewidth=3,
                            markersize=10,
                            label=self.parameter.index[k],
                        )

                ax1.set_title(title, fontsize=12)
                ax1.set_xlabel(xlabel, fontsize=12)
                ax1.set_ylabel(ylabel, fontsize=12)
                ax1.tick_params(axis="both", which="major", labelsize=labelfontsize)

                ax1.legend(fontsize=12)

                for i in range(self.num_parameters):
                    k = self.positions[i]
                    # for j in range(self.n_values):
                    for j in range(len(self.sen[self.parameter.index[k]][0])):
                        if plotting_from == "":
                            plotting_from = 0
                        if plotting_to == "":
                            plotting_to = len(
                                self.sen[self.parameter.index[k]][3][j].values
                            )

                        ax2.plot(
                            self.sen[self.parameter.index[k]][3][j].values[
                                plotting_from:plotting_to
                            ],
                            label=self.sen[self.parameter.index[k]][2][j],
                        )

                # ax2.legend(fontsize=12)
                box = ax2.get_position()
                ax2.set_position([box.x0, box.y0, box.width * 0.8, box.height])
                ax2.legend(loc=6, fancybox=True, bbox_to_anchor=(1.015, 0.5))

                ax2.set_title(title2, fontsize=12)
                ax2.set_xlabel(xlabel2, fontsize=12)
                ax2.set_ylabel(ylabel2, fontsize=12)

                plt.subplots_adjust(
                    left=spaces[0],
                    bottom=spaces[1],
                    right=spaces[2],
                    top=spaces[3],
                    wspace=spaces[4],
                    hspace=spaces[5],
                )

            except ValueError:
                assert ValueError(
                    "To plot calculated values, you should choose return_values==2 in the sensitivity object"
                )

            plt.tight_layout()
            return fig, (ax1, ax2)
