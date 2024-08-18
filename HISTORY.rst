=======
History
=======

0.1.0 (2022-05-24)
------------------

* First release on PyPI.

0.1.7 (2022-12-26)
------------------

* lock numpy to version 1.23.5


0.1.8 (2023-01-31)
------------------

* bump up versions


0.2.0 (2023-02-08)
------------------

* add eva (Extreme value analysis) module
* fix bug in obtaining distribution parameters using optimization method


0.3.0 (2023-02-19)
------------------

* add documentations for both GEV and gumbel distributions.
* add lmoment parameter estimation method for all distributions.
* add exponential and normal distributions
* modify the pdf, cdf, and probability plot plots
* create separate plot and confidence_interval modules.

0.4.0 (2023-11-23)
------------------

* add Pearson 3 distribution
* Use setup.py instead of pyproject.toml.
* Correct pearson correlation coefficient and add documentation .
* replace the pdf and cdf by the methods from scipy package.

0.5.0 (2023-12-11)
------------------

* Unify the all the methods for the distributions.
* Use factory design pattern to create the distributions.
* add tests for the eva module.
* use snake_case for the methods and variables.

0.6.0 (2024-08-18)
------------------

dev
"""
* Add documentations for the `distributions`, and `eva` modules.
* Add autodoc for all modules.
* Test docstrings as part of CI and pre-commit hooks.
* Test notebooks as part of CI.
* Simplify test for the distributions module

distributions
"""""""""""""
* move the `cdf` and `parameters` for all the methods to be optional parameters.
* rename `theoretical_estimate` method to `inverse_cdf`.
* All distributions can be instantiated with the parameters and/or data.
* rename the `probability_plot` method to `plot`.
* move the `confidence_interval` plot from the `probability_plot/plot` to the method `confidence_interval` and can be
    called by activating the `plot_figure=True`.

descriptors
"""""""""""
* rename the `metrics` module to `descriptors`.
