from typing import Union, List
from pandas import DataFrame
import numpy as np


class TimeSeries(DataFrame):
    """
    A class to represent and analyze time series data using pandas DataFrame.

    Inherits from `pandas.DataFrame` and adds additional methods for statistical-analysis and visualization specific
    to time series data.

    Parameters
    ----------
    data: array-like (1D or 2D)
        The data to be converted into a time series. If 2D, each column is treated as a separate series.
    index: array-like, optional
        The index for the time series data. If None, a default RangeIndex is used.
    name : str, optional
        The name of the column in the DataFrame. Default is 'TimeSeriesData'.
    *args : tuple
        Additional positional arguments to pass to the DataFrame constructor.
    **kwargs : dict
        Additional keyword arguments to pass to the DataFrame constructor.

    Examples
    --------
    - Create a time series from a 1D array:

        >>> data = np.random.randn(100)
        >>> ts = TimeSeries(data)
        >>> print(ts.stats) # doctest: +SKIP
                  Series1
        count  100.000000
        mean     0.061816
        std      1.016592
        min     -2.622123
        25%     -0.539548
        50%     -0.010321
        75%      0.751756
        max      2.344767

    - Create a time series from a 2D array:

        >>> data_2d = np.random.randn(100, 3)
        >>> ts_2d = TimeSeries(data_2d, columns=['A', 'B', 'C'])
        >>> print(ts_2d.stats) # doctest: +SKIP
                  Series1     Series2     Series3
        count  100.000000  100.000000  100.000000
        mean     0.239437    0.058122   -0.063077
        std      1.002170    0.980495    1.000381
        min     -2.254215   -2.500011   -2.081786
        25%     -0.405632   -0.574242   -0.799128
        50%      0.308706    0.022795   -0.245399
        75%      0.879848    0.606253    0.607085
        max      2.628358    2.822292    2.538793
    """

    def __init__(
        self,
        data: Union[DataFrame, List[float], np.ndarray],
        index=None,
        columns=None,
        *args,
        **kwargs,
    ):

        if isinstance(data, np.ndarray) and data.ndim == 1:
            data = data.reshape(-1, 1)  # Convert 1D array to 2D with one column
        if columns is None:
            columns = [f"Series{i + 1}" for i in range(data.shape[1])]

        if not isinstance(data, DataFrame):
            # Convert input data to a pandas DataFrame
            data = DataFrame(data, index=index, columns=columns)

        super().__init__(data, *args, **kwargs)
        self.columns = columns

    @property
    def _constructor(self):
        """Returns the constructor of the class."""
        return TimeSeries

    @property
    def stats(self) -> DataFrame:
        """
        Returns a detailed statistical summary of the time series.

        Returns
        -------
        pandas.DataFrame
            Statistical summary including count, mean, std, min, 25%, 50%, 75%, max.

        Examples
        --------
        >>> ts = TimeSeries(np.random.randn(100))
        >>> ts.stats
        """
        return self.describe()
