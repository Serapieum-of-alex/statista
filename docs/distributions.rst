#############
Distributions
#############

********************************************
Generalized extreme value distribution (GEV)
********************************************

- The generalised extreme value (or generalized extreme value) distribution characterises the behaviour of ‘block
    maxima’

probability density function (pdf)
==================================

.. math::
     f(x) = \frac{1}{\sigma}\ast{Q(x)}^{\xi+1}\ast e^{-Q(x)}



- where
    - :math: `\sigma` is the scale parameter
    - :math: `\mu` is the location parameter
    - :math: `\delta` is the scale parameter

Cumulative distribution function (cdf)
======================================

.. math::
    F(x)=e^{-Q(x)}


*******************
Gumbel Distribution
*******************

- The Gumbel distribution is a special case of the `Generalized extreme value distribution (GEV)`_ when the shape
    parameter :math: `\sigma` equals zero.

probability density function (pdf)
==================================


Cumulative distribution function (cdf)
======================================

.. math::
    F(x) = {e}^{- {e}^{- (\frac{x-\mu}{\delta})} }
