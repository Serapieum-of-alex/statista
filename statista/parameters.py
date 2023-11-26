"""L moments."""
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
        pass

    def Lmom(self, nmom=5):
        """Calculates the Lmoments."""
        if nmom <= 5:
            var = self._samlmusmall(nmom)
        else:
            var = self._samlmularge(nmom)

        return var

    @staticmethod
    def _comb(N, k):
        """sum [(N-j)/(j+1)]"""
        if (k > N) or (N < 0) or (k < 0):
            val = 0
        else:
            val = 1
            for j in range(min(k, N - k)):
                val = (val * (N - j)) // (j + 1)  # // is floor division
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
        coefl1 = 1.0 / self._comb(n, 1)
        suml1 = sum(x)
        lmoments = [coefl1 * suml1]

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
                coeftemp = []
                for _ in range(0, mom):
                    coeftemp.append(1)

                for j in range(0, mom - 1):
                    coeftemp[j] = coeftemp[j] * comb[mom - j - 2][i]

                for j in range(1, mom):
                    coeftemp[j] = coeftemp[j] * comb[j - 1][n - i - 1]

                for j in range(0, mom):
                    coeftemp[j] = coeftemp[j] * self._comb(mom - 1, j)

                for j in range(0, int(0.5 * mom)):
                    coeftemp[j * 2 + 1] = -coeftemp[j * 2 + 1]
                coeftemp = sum(coeftemp)
                xtrans.append(x[i] * coeftemp)

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
        # Can be simplifed to comb1 = range(0,n)

        comb1 = range(0, n)
        comb2 = range(n - 1, -1, -1)

        coefl2 = 0.5 * 1.0 / self._comb(n, 2)
        xtrans = []
        for i in range(0, n):
            coeftemp = comb1[i] - comb2[i]
            xtrans.append(coeftemp * sample[i])

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
            combtemp = self._comb(i, 2)
            comb3.append(combtemp)
            comb4.insert(0, combtemp)

        coefl3 = 1.0 / 3 * 1.0 / self._comb(n, 3)
        xtrans = []
        for i in range(0, n):
            coeftemp = comb3[i] - 2 * comb1[i] * comb2[i] + comb4[i]
            xtrans.append(coeftemp * sample[i])

        l_moment_3 = coefl3 * sum(xtrans) / l_moment_2

        if nmom == 3:
            return [l_moment_1, l_moment_2, l_moment_3]

        # Calculate Fourth order
        # comb5 = comb(i-1,3)
        # comb6 = comb(n-i,3)
        comb5 = []
        comb6 = []
        for i in range(0, n):
            combtemp = self._comb(i, 3)
            comb5.append(combtemp)
            comb6.insert(0, combtemp)

        coefl4 = 1.0 / 4 * 1.0 / self._comb(n, 4)
        xtrans = []
        for i in range(0, n):
            coeftemp = (
                comb5[i] - 3 * comb3[i] * comb2[i] + 3 * comb1[i] * comb4[i] - comb6[i]
            )
            xtrans.append(coeftemp * sample[i])

        l_moment_4 = coefl4 * sum(xtrans) / l_moment_2

        if nmom == 4:
            return [l_moment_1, l_moment_2, l_moment_3, l_moment_4]

        # Calculate Fifth order
        comb7 = []
        comb8 = []
        for i in range(0, n):
            combtemp = self._comb(i, 4)
            comb7.append(combtemp)
            comb8.insert(0, combtemp)

        coefl5 = 1.0 / 5 * 1.0 / self._comb(n, 5)
        xtrans = []
        for i in range(0, n):
            coeftemp = (
                comb7[i]
                - 4 * comb5[i] * comb2[i]
                + 6 * comb3[i] * comb4[i]
                - 4 * comb1[i] * comb6[i]
                + comb8[i]
            )
            xtrans.append(coeftemp * sample[i])

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
        DL2 = np.log(2)
        DL3 = np.log(3)
        # COEFFICIENTS OF RATIONAL-FUNCTION APPROXIMATIONS FOR XI
        A0 = 0.28377530
        A1 = -1.21096399
        A2 = -2.50728214
        A3 = -1.13455566
        A4 = -0.07138022
        B1 = 2.06189696
        B2 = 1.31912239
        B3 = 0.25077104
        C1 = 1.59921491
        C2 = -0.48832213
        C3 = 0.01573152
        D1 = -0.64363929
        D2 = 0.08985247

        T3 = lmoments[2]
        # if std <= 0 or third moment > 1
        if lmoments[1] <= 0 or abs(T3) >= 1:
            raise ValueError("L-Moments Invalid")

        if T3 <= 0:
            G = (A0 + T3 * (A1 + T3 * (A2 + T3 * (A3 + T3 * A4)))) / (
                1 + T3 * (B1 + T3 * (B2 + T3 * B3))
            )
            if T3 >= -0.8:
                shape = G
                GAM = np.exp(sp.special.gammaln(1 + G))
                scale = lmoments[1] * G / (GAM * (1 - 2 ** (-G)))
                loc = lmoments[0] - scale * (1 - GAM) / G
                para = [shape, loc, scale]
                return para

            if T3 <= -0.97:
                G = 1 - np.log(1 + T3) / DL2

            T0 = (T3 + 3) * 0.5

            for _ in range(1, MAXIT):
                X2 = 2 ** (-G)
                X3 = 3 ** (-G)
                XX2 = 1 - X2
                XX3 = 1 - X3
                T = XX3 / XX2
                DERIV = (XX2 * X3 * DL3 - XX3 * X2 * DL2) / (XX2**2)
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
            Z = 1 - T3
            G = (-1 + Z * (C1 + Z * (C2 + Z * C3))) / (1 + Z * (D1 + Z * D2))
            if abs(G) < ninf:
                # Gumbel
                scale = lmoments[1] / DL2
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
            raise ValueError("L-Moments Invalid")
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
            print("L-Moments Invalid")
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
            print("L-Moments Invalid")
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
        lmoments: List[Union[float, int]]
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

        G = -lmoments[2]
        if lmoments[1] <= 0 or abs(G) >= 1:
            print("L-Moments Invalid")
            para = None
        else:
            if abs(G) <= SMALL:
                para = [lmoments[0], lmoments[1], 0]
                return para

            GG = G * np.pi / sp.sin(G * np.pi)
            A = lmoments[1] / GG
            para1 = lmoments[0] - A * (1 - GG) / G
            para = [para1, A, G]
        return para

    @staticmethod
    def generalized_normal(
        lmoments: List[Union[float, int]]
    ) -> List[Union[float, int]]:
        """Generalized Normal distribution.

        Parameters
        ----------
        lmoments: List
            list of l moments

        Returns
        -------
        List of distribution parameters
        """
        A0 = 0.20466534e01
        A1 = -0.36544371e01
        A2 = 0.18396733e01
        A3 = -0.20360244e00
        B1 = -0.20182173e01
        B2 = 0.12420401e01
        B3 = -0.21741801e00
        SMALL = 1e-8

        T3 = lmoments[2]
        if lmoments[1] <= 0 or abs(T3) >= 1:
            print("L-Moments Invalid")
            return
        if abs(T3) >= 0.95:
            para = [0, -1, 0]
            return para

        if abs(T3) <= SMALL:
            para = [lmoments[0], lmoments[1] * np.sqrt(np.pi), 0]

        TT = T3**2
        G = (
            -T3
            * (A0 + TT * (A1 + TT * (A2 + TT * A3)))
            / (1 + TT * (B1 + TT * (B2 + TT * B3)))
        )
        E = sp.exp(0.5 * G**2)
        A = lmoments[1] * G / (E * sp.special.erf(0.5 * G))
        U = lmoments[0] + A * (E - 1) / G
        para = [U, A, G]
        return para

    @staticmethod
    def generalized_pareto(
        lmoments: List[Union[float, int]]
    ) -> List[Union[float, int]]:
        """Generalized Pareto distribution.

        Parameters
        ----------
        lmoments: List
            list of l moments

        Returns
        -------
        List of distribution parameters
        """
        T3 = lmoments[2]
        if lmoments[1] <= 0:
            print("L-Moments Invalid")
            return
        if abs(T3) >= 1:
            print("L-Moments Invalid")
            return

        G = (1 - 3 * T3) / (1 + T3)

        PARA3 = G
        PARA2 = (1 + G) * (2 + G) * lmoments[1]
        PARA1 = lmoments[0] - PARA2 / (1 + G)
        para = [PARA1, PARA2, PARA3]
        return para

    # def kappa(lmoments):
    #
    #     MAXSR = 10
    #     HSTART = 1.001
    #     BIG = 10
    #     OFLEXP = 170
    #     OFLGAM = 53
    #
    #     T3 = lmoments[2]
    #     T4 = lmoments[3]
    #     para = [0] * 4
    #     if lmoments[1] <= 0:
    #         print("L-Moments Invalid")
    #         return
    #     if abs(T3) >= 1 or abs(T4) >= 1:
    #         print("L-Moments Invalid")
    #         return
    #
    #     if T4 <= (5 * T3 * T3 - 1) / 4:
    #         print("L-Moments Invalid")
    #         return
    #
    #     if T4 >= (5 * T3 * T3 + 1) / 6:
    #         print("L-Moments Invalid")
    #         return
    #
    #     G = (1 - 3 * T3) / (1 + T3)
    #     H = HSTART
    #     Z = G + H * 0.725
    #     Xdist = BIG
    #
    #     # Newton-Raphson Iteration
    #     for it in range(1, MAXIT + 1):
    #         for i in range(1, MAXSR + 1):
    #             if G > OFLGAM:
    #                 print("Failed to converge")
    #                 return
    #             if H > 0:
    #                 U1 = sp.exp(_spsp.gammaln(1 / H) - _spsp.gammaln(1 / H + 1 + G))
    #                 U2 = sp.exp(_spsp.gammaln(2 / H) - _spsp.gammaln(2 / H + 1 + G))
    #                 U3 = sp.exp(_spsp.gammaln(3 / H) - _spsp.gammaln(3 / H + 1 + G))
    #                 U4 = sp.exp(_spsp.gammaln(4 / H) - _spsp.gammaln(4 / H + 1 + G))
    #             else:
    #                 U1 = sp.exp(_spsp.gammaln(-1 / H - G) - _spsp.gammaln(-1 / H + 1))
    #                 U2 = sp.exp(_spsp.gammaln(-2 / H - G) - _spsp.gammaln(-2 / H + 1))
    #                 U3 = sp.exp(_spsp.gammaln(-3 / H - G) - _spsp.gammaln(-3 / H + 1))
    #                 U4 = sp.exp(_spsp.gammaln(-4 / H - G) - _spsp.gammaln(-4 / H + 1))
    #
    #             ALAM2 = U1 - 2 * U2
    #             ALAM3 = -U1 + 6 * U2 - 6 * U3
    #             ALAM4 = U1 - 12 * U2 + 30 * U3 - 20 * U4
    #             if ALAM2 == 0:
    #                 print("Failed to Converge")
    #                 return
    #             TAU3 = ALAM3 / ALAM2
    #             TAU4 = ALAM4 / ALAM2
    #             E1 = TAU3 - T3
    #             E2 = TAU4 - T4
    #
    #             DIST = max(abs(E1), abs(E2))
    #             if DIST < Xdist:
    #                 Success = 1
    #                 break
    #             else:
    #                 DEL1 = 0.5 * DEL1
    #                 DEL2 = 0.5 * DEL2
    #                 G = XG - DEL1
    #                 H = XH - DEL2
    #
    #         if Success == 0:
    #             print("Failed to converge")
    #             return
    #
    #         # Test for convergence
    #         if DIST < EPS:
    #             para[3] = H
    #             para[2] = G
    #             TEMP = _spsp.gammaln(1 + G)
    #             if TEMP > OFLEXP:
    #                 print("Failed to converge")
    #                 return
    #             GAM = sp.exp(TEMP)
    #             TEMP = (1 + G) * sp.log(abs(H))
    #             if TEMP > OFLEXP:
    #                 print("Failed to converge")
    #                 return
    #
    #             HH = sp.exp(TEMP)
    #             para[1] = lmoments[1] * G * HH / (ALAM2 * GAM)
    #             para[0] = lmoments[0] - para[1] / G * (1 - GAM * U1 / HH)
    #             return (para)
    #         else:
    #             XG = G
    #             XH = H
    #             XZ = Z
    #             Xdist = DIST
    #             RHH = 1 / (H ** 2)
    #             if H > 0:
    #                 U1G = -U1 * _spsp.psi(1 / H + 1 + G)
    #                 U2G = -U2 * _spsp.psi(2 / H + 1 + G)
    #                 U3G = -U3 * _spsp.psi(3 / H + 1 + G)
    #                 U4G = -U4 * _spsp.psi(4 / H + 1 + G)
    #                 U1H = RHH * (-U1G - U1 * _spsp.psi(1 / H))
    #                 U2H = 2 * RHH * (-U2G - U2 * _spsp.psi(2 / H))
    #                 U3H = 3 * RHH * (-U3G - U3 * _spsp.psi(3 / H))
    #                 U4H = 4 * RHH * (-U4G - U4 * _spsp.psi(4 / H))
    #             else:
    #                 U1G = -U1 * _spsp.psi(-1 / H - G)
    #                 U2G = -U2 * _spsp.psi(-2 / H - G)
    #                 U3G = -U3 * _spsp.psi(-3 / H - G)
    #                 U4G = -U4 * _spsp.psi(-4 / H - G)
    #                 U1H = RHH * (-U1G - U1 * _spsp.psi(-1 / H + 1))
    #                 U2H = 2 * RHH * (-U2G - U2 * _spsp.psi(-2 / H + 1))
    #                 U3H = 3 * RHH * (-U3G - U3 * _spsp.psi(-3 / H + 1))
    #                 U4H = 4 * RHH * (-U4G - U4 * _spsp.psi(-4 / H + 1))
    #
    #             DL2G = U1G - 2 * U2G
    #             DL2H = U1H - 2 * U2H
    #             DL3G = -U1G + 6 * U2G - 6 * U3G
    #             DL3H = -U1H + 6 * U2H - 6 * U3H
    #             DL4G = U1G - 12 * U2G + 30 * U3G - 20 * U4G
    #             DL4H = U1H - 12 * U2H + 30 * U3H - 20 * U4H
    #             D11 = (DL3G - TAU3 * DL2G) / ALAM2
    #             D12 = (DL3H - TAU3 * DL2H) / ALAM2
    #             D21 = (DL4G - TAU4 * DL2G) / ALAM2
    #             D22 = (DL4H - TAU4 * DL2H) / ALAM2
    #             DET = D11 * D22 - D12 * D21
    #             H11 = D22 / DET
    #             H12 = -D12 / DET
    #             H21 = -D21 / DET
    #             H22 = D11 / DET
    #             DEL1 = E1 * H11 + E2 * H12
    #             DEL2 = E1 * H21 + E2 * H22
    #
    #             ##          TAKE NEXT N-R STEP
    #             G = XG - DEL1
    #             H = XH - DEL2
    #             Z = G + H * 0.725
    #
    #             ##          REDUCE STEP IF G AND H ARE OUTSIDE THE PARAMETER _spACE
    #             FACTOR = 1
    #             if G <= -1:
    #                 FACTOR = 0.8 * (XG + 1) / DEL1
    #             if H <= -1:
    #                 FACTOR = min(FACTOR, 0.8 * (XH + 1) / DEL2)
    #             if Z <= -1:
    #                 FACTOR = min(FACTOR, 0.8 * (XZ + 1) / (XZ - Z))
    #             if H <= 0 and G * H <= -1:
    #                 FACTOR = min(FACTOR, 0.8 * (XG * XH + 1) / (XG * XH - G * H))
    #
    #             if FACTOR == 1:
    #                 pass
    #             else:
    #                 DEL1 = DEL1 * FACTOR
    #                 DEL2 = DEL2 * FACTOR
    #                 G = XG - DEL1
    #                 H = XH - DEL2
    #                 Z = G + H * 0.725

    @staticmethod
    def normal(lmoments: List[Union[float, int]]) -> List[Union[float, int]]:
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
            print("L-Moments Invalid")
            return
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
        Small = 1e-6
        # Constants used in Minimax Approx:

        C1 = 0.2906
        C2 = 0.1882
        C3 = 0.0442
        D1 = 0.36067
        D2 = -0.59567
        D3 = 0.25361
        D4 = -2.78861
        D5 = 2.56096
        D6 = -0.77045

        T3 = abs(lmoments[2])
        if lmoments[1] <= 0 or T3 >= 1:
            para = [0] * 3
            print("L-Moments Invalid")
            return para

        if T3 <= Small:
            para = []
            para.append(lmoments[0])
            para.append(lmoments[1] * np.sqrt(np.pi))
            para.append(0)
            return para

        if T3 >= (1.0 / 3):
            T = 1 - T3
            Alpha = T * (D1 + T * (D2 + T * D3)) / (1 + T * (D4 + T * (D5 + T * D6)))
        else:
            T = 3 * np.pi * T3 * T3
            Alpha = (1 + C1 * T) / (T * (1 + T * (C2 + T * C3)))

        RTALPH = np.sqrt(Alpha)
        BETA = (
            np.sqrt(np.pi)
            * lmoments[1]
            * sp.exp(_spsp.gammaln(Alpha) - _spsp.gammaln(Alpha + 0.5))
        )
        para = []
        para.append(lmoments[0])
        para.append(BETA * RTALPH)
        para.append(2 / RTALPH)
        if lmoments[2] < 0:
            para[2] = -para[2]

        return para

    @staticmethod
    def wakeby(lmoments: List[Union[float, int]]) -> List[Union[float, int]]:
        """wakeby distribution.

        Parameters
        ----------
        lmoments: List
            list of l moments

        Returns
        -------
        List of distribution parameters
        """
        if lmoments[1] <= 0:
            print("Invalid L-Moments")
            return ()
        if abs(lmoments[2]) >= 1 or abs(lmoments[3]) >= 1 or abs(lmoments[4]) >= 1:
            print("Invalid L-Moments")
            return ()

        ALAM1 = lmoments[0]
        ALAM2 = lmoments[1]
        ALAM3 = lmoments[2] * ALAM2
        ALAM4 = lmoments[3] * ALAM2
        ALAM5 = lmoments[4] * ALAM2

        XN1 = 3 * ALAM2 - 25 * ALAM3 + 32 * ALAM4
        XN2 = -3 * ALAM2 + 5 * ALAM3 + 8 * ALAM4
        XN3 = 3 * ALAM2 + 5 * ALAM3 + 2 * ALAM4
        XC1 = 7 * ALAM2 - 85 * ALAM3 + 203 * ALAM4 - 125 * ALAM5
        XC2 = -7 * ALAM2 + 25 * ALAM3 + 7 * ALAM4 - 25 * ALAM5
        XC3 = 7 * ALAM2 + 5 * ALAM3 - 7 * ALAM4 - 5 * ALAM5

        XA = XN2 * XC3 - XC2 * XN3
        XB = XN1 * XC3 - XC1 * XN3
        XC = XN1 * XC2 - XC1 * XN2
        DISC = XB * XB - 4 * XA * XC
        skip20 = 0
        if DISC < 0:
            pass
        else:
            DISC = np.sqrt(DISC)
            ROOT1 = 0.5 * (-XB + DISC) / XA
            ROOT2 = 0.5 * (-XB - DISC) / XA
            B = max(ROOT1, ROOT2)
            D = -min(ROOT1, ROOT2)
            if D >= 1:
                pass
            else:
                A = (
                    (1 + B)
                    * (2 + B)
                    * (3 + B)
                    / (4 * (B + D))
                    * ((1 + D) * ALAM2 - (3 - D) * ALAM3)
                )
                C = (
                    -(1 - D)
                    * (2 - D)
                    * (3 - D)
                    / (4 * (B + D))
                    * ((1 - B) * ALAM2 - (3 + B) * ALAM3)
                )
                XI = ALAM1 - A / (1 + B) - C / (1 - D)
                if C >= 0 and A + C >= 0:
                    skip20 = 1

        if skip20 == 0:
            # IFAIL = 1
            D = -(1 - 3 * lmoments[2]) / (1 + lmoments[2])
            C = (1 - D) * (2 - D) * lmoments[1]
            B = 0
            A = 0
            XI = lmoments[0] - C / (1 - D)
            if D <= 0:
                A = C
                B = -D
                C = 0
                D = 0

        para = [XI, A, B, C, D]
        return para

    # TODO: add the function lmrgum
    # def weibull(lmoments):
    #     if len(lmoments) < 3:
    #         print("Insufficient L-Moments: Need 3")
    #         return
    #     if lmoments[1] <= 0 or lmoments[2] >= 1 or lmoments[2] <= -lmoments.lmrgum([0, 1], 3)[2]:
    #         print("L-Moments Invalid")
    #         return
    #     pg = Lmoments.GEV([-lmoments[0], lmoments[1], -lmoments[2]])
    #     delta = 1 / pg[2]
    #     beta = pg[1] / pg[2]
    #     out = [-pg[0] - beta, beta, delta]
    #     return (out)
