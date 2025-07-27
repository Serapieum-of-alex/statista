"""Parameters estimation."""

from __future__ import annotations

from typing import Any, List, Union

import numpy as np
import scipy as sp
import scipy.special as _spsp
from numpy import ndarray

ninf = 1e-5
MAXIT = 20
EPS = 1e-6
# Euler's constant
EU = 0.577215664901532861

LMOMENTS_INVALID_ERROR = "L-Moments Invalid"


class Lmoments:
    """Lmoments.

        Hosking (1990) introduced the concept of L-moments, which are quantities that
        can be directly interpreted as scale and shape descriptors of probability distributions
        The L-moments of order r, denoted by λr.

    λ1 = α0 = β0
    λ2 = α0 - 2α1 = 2β1 - β0
    λ3 = α0 - 6α1 + 6α2 = 6β2 - 6β1 + β0
    λ4 = α0 - 12α1 + 30α2 - 20α3 = 20β3 - 30β2 + 12β1 - β0
    """

    def __init__(self, data):
        self.data = data

    def calculate(self, nmom=5):
        """Calculates the Lmoments."""
        if nmom <= 5:
            var = self._samlmusmall(nmom)
        else:
            var = self._samlmularge(nmom)

        return var

    @staticmethod
    def _comb(n, k):
        """sum [(n-j)/(j+1)]"""
        if (k > n) or (n < 0) or (k < 0):
            val = 0
        else:
            val = 1
            for j in range(min(k, n - k)):
                val = (val * (n - j)) // (j + 1)  # // is floor division
        return val

    def _samlmularge(self, nmom: int = 5) -> list[ndarray | float | int | Any]:
        """Large sample L moment."""

        x = self.data
        if nmom <= 0:
            raise ValueError("Invalid number of Sample L-Moments")

        x = sorted(x)
        n = len(x)

        if n < nmom:
            raise ValueError("Insufficient length of data for specified nmoments")

        # Calculate first order
        coef_l1 = 1.0 / self._comb(n, 1)
        sum_l1 = sum(x)
        lmoments = [coef_l1 * sum_l1]

        if nmom == 1:
            return lmoments[0]

        # Setup comb table, where comb[i][x] refers to comb(x,i)
        comb = []
        for i in range(1, nmom):
            comb.append([])
            for j in range(n):
                comb[-1].append(self._comb(j, i))

        for mom in range(2, nmom + 1):
            coefl = 1.0 / mom * 1.0 / self._comb(n, mom)
            xtrans = []
            for i in range(0, n):
                coef_temp = []
                for _ in range(0, mom):
                    coef_temp.append(1)

                for j in range(0, mom - 1):
                    coef_temp[j] = coef_temp[j] * comb[mom - j - 2][i]

                for j in range(1, mom):
                    coef_temp[j] = coef_temp[j] * comb[j - 1][n - i - 1]

                for j in range(0, mom):
                    coef_temp[j] = coef_temp[j] * self._comb(mom - 1, j)

                for j in range(0, int(0.5 * mom)):
                    coef_temp[j * 2 + 1] = -coef_temp[j * 2 + 1]
                coef_temp = sum(coef_temp)
                xtrans.append(x[i] * coef_temp)

            if mom > 2:
                lmoments.append(coefl * sum(xtrans) / lmoments[1])
            else:
                lmoments.append(coefl * sum(xtrans))
        return lmoments

    def _samlmusmall(self, nmom: int = 5) -> list[ndarray | float | int | Any]:
        """Small sample L-Moments."""
        sample = self.data

        if nmom <= 0:
            raise ValueError("Invalid number of Sample L-Moments")

        sample = sorted(sample)
        n = len(sample)

        if n < nmom:
            raise ValueError("Insufficient length of data for specified nmoments")

        # coefl1 = 1.0 / self._comb(n, 1)  # coefl1 = 1/n
        # suml1 = sum(sample)
        # l_moment_1 = coefl1 * suml1  # l_moment_1 = mean(sample)
        l_moment_1 = np.mean(sample)
        if nmom == 1:
            return [l_moment_1]

        # comb terms appear elsewhere, this will decrease calc time
        # for nmom > 2, and shouldn't decrease time for nmom == 2
        # comb(sample,1) = sample
        # for i in range(1,n+1):
        # #        comb1.append(comb(i-1,1))
        # #        comb2.append(comb(n-i,1))
        # Can be simplified to comb1 = range(0,n)

        comb1 = range(0, n)
        comb2 = range(n - 1, -1, -1)

        coefl2 = 0.5 * 1.0 / self._comb(n, 2)
        xtrans = []
        for i in range(0, n):
            coef_temp = comb1[i] - comb2[i]
            xtrans.append(coef_temp * sample[i])

        l_moment_2 = coefl2 * sum(xtrans)

        if nmom == 2:
            return [l_moment_1, l_moment_2]

        # Calculate Third order
        # comb terms appear elsewhere, this will decrease calc time
        # for nmom > 2, and shouldn't decrease time for nmom == 2
        # comb3 = comb(i-1,2)
        # comb4 = comb3.reverse()
        comb3 = []
        comb4 = []
        for i in range(0, n):
            comb_temp = self._comb(i, 2)
            comb3.append(comb_temp)
            comb4.insert(0, comb_temp)

        coefl3 = 1.0 / 3 * 1.0 / self._comb(n, 3)
        xtrans = []
        for i in range(0, n):
            coef_temp = comb3[i] - 2 * comb1[i] * comb2[i] + comb4[i]
            xtrans.append(coef_temp * sample[i])

        l_moment_3 = coefl3 * sum(xtrans) / l_moment_2

        if nmom == 3:
            return [l_moment_1, l_moment_2, l_moment_3]

        # Calculate Fourth order
        # comb5 = comb(i-1,3)
        # comb6 = comb(n-i,3)
        comb5 = []
        comb6 = []
        for i in range(0, n):
            comb_temp = self._comb(i, 3)
            comb5.append(comb_temp)
            comb6.insert(0, comb_temp)

        coefl4 = 1.0 / 4 * 1.0 / self._comb(n, 4)
        xtrans = []
        for i in range(0, n):
            coef_temp = (
                comb5[i] - 3 * comb3[i] * comb2[i] + 3 * comb1[i] * comb4[i] - comb6[i]
            )
            xtrans.append(coef_temp * sample[i])

        l_moment_4 = coefl4 * sum(xtrans) / l_moment_2

        if nmom == 4:
            return [l_moment_1, l_moment_2, l_moment_3, l_moment_4]

        # Calculate Fifth order
        comb7 = []
        comb8 = []
        for i in range(0, n):
            comb_temp = self._comb(i, 4)
            comb7.append(comb_temp)
            comb8.insert(0, comb_temp)

        coefl5 = 1.0 / 5 * 1.0 / self._comb(n, 5)
        xtrans = []
        for i in range(0, n):
            coef_temp = (
                comb7[i]
                - 4 * comb5[i] * comb2[i]
                + 6 * comb3[i] * comb4[i]
                - 4 * comb1[i] * comb6[i]
                + comb8[i]
            )
            xtrans.append(coef_temp * sample[i])

        l_moment_5 = coefl5 * sum(xtrans) / l_moment_2

        if nmom == 5:
            return [l_moment_1, l_moment_2, l_moment_3, l_moment_4, l_moment_5]

    @staticmethod
    def gev(lmoments: List[Union[float, int]]) -> List[Union[float, int]]:
        """Generalized Extreme Value distribution.

            Estimate the generalized extreme value distribution parameters using Lmoments method.

        Parameters
        ----------
        lmoments: List
            list of l moments

        Returns
        -------
        List of distribution parameters
        """
        dl2 = np.log(2)
        dl3 = np.log(3)
        # COEFFICIENTS OF RATIONAL-FUNCTION APPROXIMATIONS FOR XI
        a0 = 0.28377530
        a1 = -1.21096399
        a2 = -2.50728214
        a3 = -1.13455566
        a4 = -0.07138022
        b1 = 2.06189696
        b2 = 1.31912239
        b3 = 0.25077104
        c1 = 1.59921491
        c2 = -0.48832213
        c3 = 0.01573152
        d1 = -0.64363929
        d2 = 0.08985247

        t3 = lmoments[2]
        # if std <= 0 or third moment > 1
        if lmoments[1] <= 0 or abs(t3) >= 1:
            raise ValueError(LMOMENTS_INVALID_ERROR)

        if t3 <= 0:
            G = (a0 + t3 * (a1 + t3 * (a2 + t3 * (a3 + t3 * a4)))) / (
                1 + t3 * (b1 + t3 * (b2 + t3 * b3))
            )
            if t3 >= -0.8:
                shape = G
                GAM = np.exp(sp.special.gammaln(1 + G))
                scale = lmoments[1] * G / (GAM * (1 - 2 ** (-G)))
                loc = lmoments[0] - scale * (1 - GAM) / G
                para = [shape, loc, scale]
                return para

            if t3 <= -0.97:
                G = 1 - np.log(1 + t3) / dl2

            T0 = (t3 + 3) * 0.5

            for _ in range(1, MAXIT):
                X2 = 2 ** (-G)
                X3 = 3 ** (-G)
                XX2 = 1 - X2
                XX3 = 1 - X3
                T = XX3 / XX2
                DERIV = (XX2 * X3 * dl3 - XX3 * X2 * dl2) / (XX2**2)
                GOLD = G
                G = G - (T - T0) / DERIV
                if abs(G - GOLD) <= EPS * G:
                    shape = G
                    GAM = np.exp(sp.special.gammaln(1 + G))
                    scale = lmoments[1] * G / (GAM * (1 - 2 ** (-G)))
                    loc = lmoments[0] - scale * (1 - GAM) / G
                    para = [shape, loc, scale]
                    return para
            raise Exception("Iteration has not converged")
        else:
            Z = 1 - t3
            G = (-1 + Z * (c1 + Z * (c2 + Z * c3))) / (1 + Z * (d1 + Z * d2))
            if abs(G) < ninf:
                # Gumbel
                scale = lmoments[1] / dl2
                loc = lmoments[0] - EU * scale
                para = [0, loc, scale]
            else:
                # GEV
                shape = G
                GAM = np.exp(sp.special.gammaln(1 + G))
                scale = lmoments[1] * G / (GAM * (1 - 2 ** (-G)))
                loc = lmoments[0] - scale * (1 - GAM) / G
                # multiply the shape by -1 to follow the + ve shape parameter equation (+ve value means heavy tail)
                # para = [-1 * shape, loc, scale]
                para = [shape, loc, scale]

            return para

    @staticmethod
    def gumbel(lmoments: List[Union[float, int]]) -> List[Union[float, int]]:
        """ "Gumbel" distribution.

        Parameters
        ----------
        lmoments: List
            list of l moments

        Returns
        -------
        List of distribution parameters
        """
        if lmoments[1] <= 0:
            raise ValueError(LMOMENTS_INVALID_ERROR)
        else:
            para2 = lmoments[1] / np.log(2)
            para1 = lmoments[0] - EU * para2
            para = [para1, para2]
            return para

    @staticmethod
    def exponential(lmoments: List[Union[float, int]]) -> List[Union[float, int]]:
        """Exponential distribution.

        Parameters
        ----------
        lmoments: List
            list of l moments

        Returns
        -------
        List of distribution parameters
        """
        if lmoments[1] <= 0:
            print(LMOMENTS_INVALID_ERROR)
            para = None
        else:
            para = [lmoments[0] - 2 * lmoments[1], 2 * lmoments[1]]

        return para

    @staticmethod
    def gamma(lmoments: List[Union[float, int]]) -> List[Union[float, int]]:
        """Gamma distribution.

        Parameters
        ----------
        lmoments: List
            list of l moments

        Returns
        -------
        List of distribution parameters
        """
        A1 = -0.3080
        A2 = -0.05812
        A3 = 0.01765
        B1 = 0.7213
        B2 = -0.5947
        B3 = -2.1817
        B4 = 1.2113

        if lmoments[0] <= lmoments[1] or lmoments[1] <= 0:
            print(LMOMENTS_INVALID_ERROR)
            para = None
        else:
            CV = lmoments[1] / lmoments[0]
            if CV >= 0.5:
                T = 1 - CV
                ALPHA = T * (B1 + T * B2) / (1 + T * (B3 + T * B4))
            else:
                T = np.pi * CV**2
                ALPHA = (1 + A1 * T) / (T * (1 + T * (A2 + T * A3)))

            para = [ALPHA, lmoments[0] / ALPHA]
        return para

    @staticmethod
    def generalized_logistic(
        lmoments: List[Union[float, int]],
    ) -> List[Union[float, int]]:
        """Generalized logistic distribution.

        Parameters
        ----------
        lmoments: List
            list of l moments

        Returns
        -------
        List of distribution parameters
        """
        SMALL = 1e-6

        g = -lmoments[2]
        if lmoments[1] <= 0 or abs(g) >= 1:
            print(LMOMENTS_INVALID_ERROR)
            para = None
        else:
            if abs(g) <= SMALL:
                para = [lmoments[0], lmoments[1], 0]
                return para

            gg = g * np.pi / sp.sin(g * np.pi)
            a = lmoments[1] / gg
            para1 = lmoments[0] - a * (1 - gg) / g
            para = [para1, a, g]
        return para

    @staticmethod
    def generalized_normal(
        lmoments: List[Union[float, int]] | None,
    ) -> List[Union[float, int]] | None:
        """Generalized Normal distribution.

        Args:
            lmoments (List):
                list of l moments

        Returns:
            List of distribution parameters
        """
        a0 = 0.20466534e01
        a1 = -0.36544371e01
        a2 = 0.18396733e01
        a3 = -0.20360244e00
        b1 = -0.20182173e01
        b2 = 0.12420401e01
        b3 = -0.21741801e00

        t3 = lmoments[2]
        if lmoments[1] <= 0 or abs(t3) >= 1:
            print(LMOMENTS_INVALID_ERROR)
            return None

        if abs(t3) >= 0.95:
            para = [0, -1, 0]
            return para

        tt = t3**2
        g = (
            -t3
            * (a0 + tt * (a1 + tt * (a2 + tt * a3)))
            / (1 + tt * (b1 + tt * (b2 + tt * b3)))
        )
        e = sp.exp(0.5 * g**2)
        a = lmoments[1] * g / (e * sp.special.erf(0.5 * g))
        u = lmoments[0] + a * (e - 1) / g
        para = [u, a, g]
        return para

    @staticmethod
    def generalized_pareto(
        lmoments: List[Union[float, int]],
    ) -> list[float] | None:
        """Generalized Pareto distribution.

        Parameters
        ----------
        lmoments: List
            list of l moments

        Returns
        -------
        List of distribution parameters
        """
        t3 = lmoments[2]
        if lmoments[1] <= 0:
            print(LMOMENTS_INVALID_ERROR)
            return None

        if abs(t3) >= 1:
            print(LMOMENTS_INVALID_ERROR)
            return None

        g = (1 - 3 * t3) / (1 + t3)

        para3 = g
        para2 = (1 + g) * (2 + g) * lmoments[1]
        para1 = lmoments[0] - para2 / (1 + g)
        para = [para1, para2, para3]
        return para

    @staticmethod
    def normal(lmoments: List[Union[float, int]]) -> List[Union[float, int]] | None:
        """Normal distribution.

        Parameters
        ----------
        lmoments: List
            list of l moments

        Returns
        -------
        List of distribution parameters
        """
        if lmoments[1] <= 0:
            print(LMOMENTS_INVALID_ERROR)
            return None
        else:
            para = [lmoments[0], lmoments[1] * np.sqrt(np.pi)]
            return para

    @staticmethod
    def pearson_3(lmoments: List[Union[float, int]]) -> List[Union[float, int]]:
        """Pearson III (PE3) distribution.

        Parameters
        ----------
        lmoments: List
            list of l moments

        Returns
        -------
        List of distribution parameters
        """
        small = 1e-6
        # Constants used in Minimax Approx:

        c1 = 0.2906
        c2 = 0.1882
        c3 = 0.0442
        d1 = 0.36067
        d2 = -0.59567
        d3 = 0.25361
        d4 = -2.78861
        d5 = 2.56096
        d6 = -0.77045

        t3 = abs(lmoments[2])
        if lmoments[1] <= 0 or t3 >= 1:
            para = [0] * 3
            print(LMOMENTS_INVALID_ERROR)
            return para

        if t3 <= small:
            para = [lmoments[0], lmoments[1] * np.sqrt(np.pi), 0]
            return para

        if t3 >= (1.0 / 3):
            t = 1 - t3
            alpha = t * (d1 + t * (d2 + t * d3)) / (1 + t * (d4 + t * (d5 + t * d6)))
        else:
            t = 3 * np.pi * t3 * t3
            alpha = (1 + c1 * t) / (t * (1 + t * (c2 + t * c3)))

        rtalph = np.sqrt(alpha)
        beta = (
            np.sqrt(np.pi)
            * lmoments[1]
            * sp.exp(_spsp.gammaln(alpha) - _spsp.gammaln(alpha + 0.5))
        )
        para = [lmoments[0], beta * rtalph, 2 / rtalph]
        if lmoments[2] < 0:
            para[2] = -para[2]

        return para

    @staticmethod
    def wakeby(lmoments: List[Union[float, int]]) -> List[Union[float, int]] | None:
        """wakeby distribution.

        Args:
            lmoments (List):
                list of l moments

        Args:
            List of distribution parameters
        """
        if lmoments[1] <= 0:
            print("Invalid L-Moments")
            return None
        if abs(lmoments[2]) >= 1 or abs(lmoments[3]) >= 1 or abs(lmoments[4]) >= 1:
            print("Invalid L-Moments")
            return None

        alam1 = lmoments[0]
        alam2 = lmoments[1]
        alam3 = lmoments[2] * alam2
        alam4 = lmoments[3] * alam2
        alam5 = lmoments[4] * alam2

        xn1 = 3 * alam2 - 25 * alam3 + 32 * alam4
        xn2 = -3 * alam2 + 5 * alam3 + 8 * alam4
        xn3 = 3 * alam2 + 5 * alam3 + 2 * alam4
        xc1 = 7 * alam2 - 85 * alam3 + 203 * alam4 - 125 * alam5
        xc2 = -7 * alam2 + 25 * alam3 + 7 * alam4 - 25 * alam5
        xc3 = 7 * alam2 + 5 * alam3 - 7 * alam4 - 5 * alam5

        xa = xn2 * xc3 - xc2 * xn3
        xb = xn1 * xc3 - xc1 * xn3
        xc = xn1 * xc2 - xc1 * xn2
        disc = xb * xb - 4 * xa * xc
        skip20 = 0
        if disc < 0:
            pass
        else:
            disc = np.sqrt(disc)
            root1 = 0.5 * (-xb + disc) / xa
            root2 = 0.5 * (-xb - disc) / xa
            b = max(root1, root2)
            d = -min(root1, root2)
            if d >= 1:
                pass
            else:
                a = (
                    (1 + b)
                    * (2 + b)
                    * (3 + b)
                    / (4 * (b + d))
                    * ((1 + d) * alam2 - (3 - d) * alam3)
                )
                c = (
                    -(1 - d)
                    * (2 - d)
                    * (3 - d)
                    / (4 * (b + d))
                    * ((1 - b) * alam2 - (3 + b) * alam3)
                )
                xi = alam1 - a / (1 + b) - c / (1 - d)
                if c >= 0 and a + c >= 0:
                    skip20 = 1

        if skip20 == 0:
            d = -(1 - 3 * lmoments[2]) / (1 + lmoments[2])
            c = (1 - d) * (2 - d) * lmoments[1]
            b = 0
            a = 0
            xi = lmoments[0] - c / (1 - d)
            if d <= 0:
                a = c
                b = -d
                c = 0
                d = 0

        para = [xi, a, b, c, d]
        return para
