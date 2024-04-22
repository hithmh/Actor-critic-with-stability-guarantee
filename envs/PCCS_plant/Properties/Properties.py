# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:12:02 2017

@author: decardin
"""

from casadi import *
import numpy as np
from scipy.optimize import fsolve


class Properties:
    # Declare constants
    MW = np.array([28.0134, 44.01, 61.08, 18.01528])
    R = 8.314  # Universal gas constant, m^3.kPa/kmol/K
    NC = 4  # Number of components

    # Create dummy symbolics to build functions
    x = SX.sym('x', 4)  # Concentration
    z = SX.sym('z', 4)  # Additional inputs concentration
    T = SX.sym('T')
    P = SX.sym('P')

    # Methods to determine properties

    def moleFraction(self, x):
        func = self.x / (sum1(self.x) + 1E-100)
        f = Function('moleFraction', [self.x], [func])
        return f(x)

    def avgMolWeight(self, x):
        x1 = self.moleFraction(self.x)
        func = sum1(x1 * self.MW)
        f = Function('avgMolWeight', [self.x], [func])
        return f(x)

    #######################################################################
    # Liquid phase properties 
    #######################################################################

    def liqDensityMol(self, T, P, x):
        x1 = self.moleFraction(self.x)
        mw = self.MW
        nc = self.NC
        vm = SX.zeros(nc)
        rho = SX.zeros(nc)
        vm[1] = 0.04747
        rho[2] = 1.19093 + (-0.00042999 * self.T) + (-0.000000566060 * self.T ** 2)
        rho[3] = 0.863559 + (0.00121494 * self.T) + (-0.00000257080 * self.T ** 2)

        vm[2:nc] = mw[2:nc] / (rho[2:nc] + 1e-100)

        func = (1 / (sum1(x1 * vm) + 1E-100 + x1[2] * x1[3] * (-1.8218))) * 1000.0
        f = Function('liqDensityMol', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kmol/m^3 or mol/L or M

    def liqDensityMass(self, T, P, x):
        rho_mol = self.liqDensityMol(self.T, self.P, self.x)
        avgmw = self.avgMolWeight(self.x)
        func = avgmw * rho_mol
        f = Function('liqDensityMass', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kg/m^3 or g/L

    def liqViscosityComp(self, T, P, x):
        b = np.array([
            [-14.09345, 1331.0784, 0.0], [-19.355128, 4568.5591, 0.0],
            [-12.260477, 1515.6766, 0.0]
        ])
        x1 = self.moleFraction(self.x)
        nc = self.NC
        visc = SX.zeros(nc)
        for i in np.arange(1, nc):
            visc[i] = np.exp((b[i - 1][0]) + (b[i - 1][1] / self.T) + (b[i - 1][2] * np.log(self.T)))
        func = visc
        f = Function('liqViscosityComp', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kg·m^−1·s^−1 or Pa.s

    def liqViscosityMixture(self, T, P, x):
        visc = self.liqViscosityComp(self.T, self.P, self.x)
        x1 = self.moleFraction(self.x)
        func = sum1(x1 * visc)
        f = Function('liqViscosityMixture', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kg·m^−1·s^−1

    def henryConstant(self, T, P, x):
        nc = self.NC
        mw = self.MW
        x1 = self.moleFraction(self.x)
        HeC_H = 2820000 * np.exp(-2044.0 / self.T)
        HeN_H = 8550000 * np.exp(-2284.0 / self.T)
        HeN_M = 120700 * np.exp(-1136.0 / self.T)
        rho = SX.zeros(nc)
        rho[2] = 1.19093 + (-0.00042999 * self.T) + (-0.000000566060 * self.T ** 2)
        rho[3] = 0.863559 + (0.00121494 * self.T) + (-0.000000257080 * self.T ** 2)
        vm = SX.zeros(nc)
        vm[1] = 0.04747
        vm[2:nc] = mw[2:nc] / rho[2:nc]
        v = SX.zeros(nc)
        v[2:nc] = (x1[2:nc] * vm[2:nc]) / sum1(x1[2:nc] * vm[2:nc])
        H_E = v[2] * v[3] * (4.793 - (0.007446 * T) - (2.201 * v[3]))
        HeN_MM = np.exp((v[2] * np.log(HeN_M)) + (v[3] * np.log(HeN_H)) + H_E)

        func = HeN_MM * (HeC_H / HeN_H) * 1000.0
        f = Function('henryConstant', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # Pa.m^3/kmol

    def liqDiffusivity(self, T, P, x):
        x1 = self.moleFraction(self.x)
        D_CO2_H2O = 0.00000235 * np.exp(-2119.0 / self.T)
        D_N2O_H2O = 0.00000507 * np.exp(-2371.0 / self.T)
        conc_MEA = self.liqDensityMol(self.T, self.P, self.x) * x1[2]
        D_N2O_MEAaq = 0.000001 * (5.07 + (0.865 * conc_MEA) + (0.278 * conc_MEA * conc_MEA)) * np.exp(
            (-2371.0 - (93.4 * conc_MEA)) / self.T)
        D_MEA_MEAaq = np.exp(-13.275 - (2198.3 / self.T) - (0.078142 * conc_MEA))
        D_CO2_MEAaq = (D_CO2_H2O / D_N2O_H2O) * D_N2O_MEAaq

        func = SX.zeros(4)
        func[0] = 0.00000000188
        func[1] = D_CO2_MEAaq
        func[2] = D_MEA_MEAaq
        func[3] = 0.0 + 1e-100
        f = Function('liqDiffusivity', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # m^2/s

    def liqDiffusivityCarbamate(self, T, P, x):
        mu = self.liqViscosityComp(self.T, self.P, self.x)
        D_MEACOO = np.exp((-22.64) - (1000 / self.T) - (0.7 * np.log(mu[2])))

        func = D_MEACOO
        f = Function('liqDiffusivityCarbamate', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # m^2/s

    def vapourPressure(self, T, P, x):
        b = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [47.017, -2839.0, 0.0, 0.0, -3.8639, 2.8112E-16, 6.0],
                      [172.78, -13492.0, 0.0, 0.0, -21.914, 1.3779E-5, 2.0],
                      [72.55, -7206.7, 0.0, 0, -7.1385, 4.046E-6, 2.0]])
        nc = self.NC
        vp = SX.zeros(nc)

        for i in range(nc):
            vp[i] = np.exp(
                b[i][0] + (b[i][1] / (self.T + b[i][2])) + (b[i][3] * self.T) + (b[i][4] * np.log(self.T)) + (
                            b[i][5] * pow(self.T, b[i][6])))

        func = vp
        f = Function('vapourPressure', [self.T, self.P, self.x], [func])
        return f(T, P, x)

    def liqHeatCapacityMass(self, T, P, x):
        b = np.array([[0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0],
                      [2.6161, 0.003706, 0.000003787, 0.0, 0.0],
                      [4.2107, -0.00001696, 0.00002568, -0.0000001095, 0.0000000003038]])
        T = T - 273.15

        func = (b[:, 0]) + (b[:, 1] * self.T) + (b[:, 2] * self.T ** 2.0) + (b[:, 3] * self.T ** 3.0) + (
                    b[:, 4] * self.T ** 4.0)
        f = Function('liqHeatCapacityMass', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kJ/kg/K

    def liqHeatCapacityMol(self, T, P, x):
        x1 = self.moleFraction(self.x)
        mw = self.MW
        Cp = self.liqHeatCapacityMass(self.T, self.P, self.x)

        func = Cp * mw
        f = Function('liqHeatCapacityMol', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kJ/kmol/K

    def liqAvgHeatCapacityMol(self, T, P, x):
        Cp_mol = self.liqHeatCapacityMol(self.T, self.P, self.x)
        x1 = self.moleFraction(self.x)

        func = sum1(Cp_mol * x1)
        f = Function('liqAvgHeatCapacityMol', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kJ/kmol/K

    def liqVapourizationHeat(self, T, P, x):
        nc = self.NC
        b = np.array([[1.0, 1.0, 1.0, 1.0, 1.0],
                      [304.2, 17165.880, 194.7, 0.3576292, 0.0],
                      [614.45, 54835.8, 399.82, 0.3287809, -0.0856624],
                      [647.3, 40683.136, 373.2, 0.31064607, 0.0]])
        vh = SX.zeros(nc)
        for i in np.arange(2, nc):
            vh[i] = b[i, 1] * pow(((1 - (self.T / b[i, 0])) / (1 - (b[i, 2] / b[i, 0]))),
                                  (b[i, 3] + (b[i, 4] * (1 - (self.T / b[i, 0])))))

        func = vh
        f = Function('liqVapourizationHeat', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kJ/kmol

    def liqActivityCoefficient(self, T, P, x):
        a = np.array([[-0.7975, 202.381], [-0.1864, 356.193]])
        x1 = self.moleFraction(self.x)
        nc = self.NC
        A = np.exp(a[:, 0] + (a[:, 1] / self.T))
        ac = SX.zeros(nc)
        ac[0] = 1.0
        ac[1] = 1.0
        ac[2] = np.exp(-np.log(x1[2] + (A[0] * x1[3])) + x1[3] * (
                    (A[0] / (x1[2] + (A[0] * x1[3]))) - (A[1] / (x1[3] + (A[1] * x1[2])))))
        ac[3] = np.exp(-np.log(x1[3] + (A[1] * x1[2])) - x1[2] * (
                    (A[0] / (x1[2] + (A[0] * x1[3]))) - (A[1] / (x1[3] + (A[1] * x1[2])))))

        func = ac
        f = Function('liqActivityCoefficient', [self.T, self.P, self.x], [func])
        return f(T, P, x)

    def liqConcentration(self, T, P, x):
        x1 = self.moleFraction(self.x)
        cc = np.array([[2.8898, -3635.09, 0, 0],
                       [2.1211, -8189.38, 0, -0.007484],
                       [231.456, -12092.10, -36.7816, 0],
                       [216.049, -12431.70, -35.4819, 0],
                       [132.899, -13445.9, -22.4773, 0]])
        Keq = np.exp((cc[:, 0]) + (cc[:, 1] / self.T) + (cc[:, 2] * np.log(self.T)) + (cc[:, 3] * self.T))
        Keq7 = Keq[3] / (Keq[1]+1E-100)
        Keq8 = 1 / Keq[0]
        conc = self.liqDensityMol(self.T, self.P, self.x) * 0.001 * x1
        alpha = conc[1] / (conc[2]+1E-100)

        a = Keq8 * Keq8
        b = (2 * Keq8 * Keq8 * conc[2] * alpha) + (2 * Keq7 * Keq8)
        c = (Keq7 * Keq7) - (Keq7) + (2 * Keq7 * Keq8 * conc[2] * alpha) + (
                    Keq8 * Keq8 * conc[2] * conc[2] * alpha * alpha) - (Keq7 * Keq8 * conc[2])
        d = (-Keq7 * Keq7 * conc[2]) - (Keq7 * conc[2] * alpha) - (Keq7 * Keq8 * conc[2] * conc[2] * alpha)
        e = (Keq7 * Keq7 * conc[2] * conc[2] * alpha) - (Keq7 * Keq7 * conc[2] * conc[2] * alpha * alpha)

        p = ((8 * a * c) - (3.0 * b ** 2.0)) / ((8.0 * a ** 2.0)+1E-100)
        q = ((b ** 3.0) - (4.0 * a * b * c) + (8 * a * a * d)) / ((8 * a ** 3.0)+1E-100)
        D0 = (c ** 2.0) - (3 * b * d) + (12 * a * e)
        D1 = (2 * c ** 3.0) - (9 * b * c * d) + (27 * b * b * e) + (27 * a * d * d) - (72 * a * c * e)
        Q = fmax((D1 + np.sqrt(fmax((D1*D1) - (4*D0*D0*D0), 1E-100)))/2.0, 1E-100)**(1.0/3.0)
        # Q = ((D1 + np.sqrt(1E-50)) / 2.0) ** (1.0 / 3.0)
        S = (1.0 / 2.0) * np.sqrt(
            fmax((-2.0 * p / 3.0) + (1.0 / (3.0 * a + 1E-100)) * (Q + (D0 / (Q + 1E-100))), 1E-100))

        r = SX.zeros(4)
        # r[0] = (-b / (4 * a)) - S + ((1.0 / 2.0) * np.sqrt((-4 * S * S) - (2 * p) + (q / S)))
        # r[1] = (-b / (4 * a)) - S - ((1.0 / 2.0) * np.sqrt((-4 * S * S) - (2 * p) + (q / S)))
        # r[2] = (-b / (4 * a)) + S + ((1.0 / 2.0) * np.sqrt((-4 * S * S) - (2 * p) - (q / S)))
        r = (-b / (4 * a + 1E-100)) + S - (
                    (1.0 / 2.0) * np.sqrt(fmax((-4 * S * S) - (2 * p) - (q / (S + 1E-100)), 1E-100)))
        # r = np.roots([A,B,C,D,E])

        ext1 = r
        ext2 = (Keq8 * ext1 * (ext1 + (conc[2] * alpha))) / (Keq7+1E-100)
        #        for i in range(r.size):
        #            if r[i] > 0:
        #                ext1 = r[i]
        #                ext2 = (Keq8*ext1*(ext1 + (conc[2]*alpha)))/Keq7
        #                if (ext1 >= 0 and ext2 >= 0 and (ext1+ext2) <= (conc[2]*alpha) and (ext1+ext2) <= (conc[2]*(1-alpha)) ):
        #                    break
        #                else:
        #                    ext1 = 0
        #                    ext2 = 0
        C_MEAf = (conc[2] * (1 - alpha) - ext1 - ext2)
        C_MEAH = ((conc[2] * alpha) + ext1)
        C_MEACOO = ext2
        Keq1 = Keq[2] / ((Keq[0] * Keq[1])+1E-100)
        C_CO2f = (C_MEAH * C_MEACOO) / ((Keq1 * C_MEAf * C_MEAf)+1E-100)

        lc = SX.zeros(2)
        # C_MEAf = 5.986 - (16.87*alpha) - (0.003431*self.T) + (0.02604*alpha*self.T)
        # C_CO2f = 0.006598 - (0.1326*alpha) - (0.000021*self.T) + (0.000430*alpha*self.T)
        # lc = np.array([C_MEAf, C_CO2f])
        lc[0] = 1000.0 * C_MEAf
        lc[1] = 1000.0 * C_CO2f

        # func = lc
        func = lc
        f = Function('liqConcentration', [self.T, self.P, self.x], [func])
        return f(T, P, x)

    def KeqOverall(self, T, P, x):
        cc = np.array([[2.8898, -3635.09, 0, 0],
                       [2.1211, -8189.38, 0, -0.007484],
                       [231.456, -12092.10, -36.7816, 0],
                       [216.049, -12431.70, -35.4819, 0],
                       [132.899, -13445.9, -22.4773, 0]])
        Keq = np.exp((cc[:, 0]) + (cc[:, 1] / self.T) + (cc[:, 2] * np.log(self.T)) + (cc[:, 3] * self.T))

        func = (Keq[2]) / (Keq[0] * Keq[1])
        f = Function('KeqOverall', [self.T, self.P, self.x], [func])
        return f(T, P, x)

    def Kvalue1(self, T, P, x):
        x1 = self.moleFraction(self.x)
        nc = self.NC

        K = SX.zeros(nc)

        f1 = np.exp(
            385.738 - 64.49 * np.log(self.T) - 5.026 * x1[1] - 0.006 * self.T * x1[2] + 0.129 * self.T * x1[1] * x1[2])
        f2 = np.exp(490.131 - 82.103 * np.log(self.T) + 2.562 * x1[1] + 1.597 * x1[2] + 0.022 * self.T * x1[1] * x1[2])
        f3 = np.exp(531.525 - 89.121 * np.log(self.T) + 10.441 * x1[1] + 4.158 * x1[2] - 0.062 * T * x1[1] * x1[2])
        f4 = np.exp(
            505.895 - 84.832 * np.log(self.T) + 12.072 * x1[1] + 0.006 * self.T * x1[2] + 0.046 * self.T * x1[1] * x1[
                2])

        f5 = np.exp(
            -315.529 + 52.289 * np.log(self.T) - 3.471 * x1[1] - 0.003 * self.T * x1[2] + 0.112 * self.T * x1[1] * x1[
                2])
        f6 = np.exp(-341.477 + 56.65 * np.log(self.T) - 4.008 * x1[1] - 1.762 * x1[2] + 0.098 * self.T * x1[1] * x1[2])
        f7 = np.exp(
            -339.542 + 56.331 * np.log(self.T) - 3.901 * x1[1] - 0.003 * self.T * x1[2] + 0.026 * self.T * x1[1] * x1[
                2])

        z1 = if_else(logic_and(self.T > 388, self.T <= 389), f3, f4)
        z2 = if_else(logic_and(self.T > 387, self.T <= 388), f2, z1)
        z3 = if_else(logic_and(self.T >= 385, self.T <= 387), f1, z2)

        z4 = if_else(logic_and(self.T > 387, self.T <= 389), f6, f7)
        z5 = if_else(logic_and(self.T >= 385, self.T <= 387), f5, z4)

        K[0] = 0.0
        K[1] = z3
        K[2] = z5
        K[3] = -71.927 + 12.203 * np.log(self.T) + 0.34 * x1[1] + 0.265 * x1[3]

        func = K
        f = Function('Kvalue1', [self.T, self.P, self.x], [func])
        return f(T, P, x)

    def Kvalue2(self, T, P, x):
        Pvap = self.vapourPressure(self.T, self.P, self.x)
        gamma = self.liqActivityCoefficient(self.T, self.P, self.x)
        H = self.henryConstant(self.T, self.P, self.x)
        R = self.R
        nc = self.NC
        K = SX.zeros(nc)
        for i in range(nc):
            if (i == 1):
                K[i] = (H * gamma[i]) / (R * self.T * 1000.0)
            else:
                K[i] = (gamma[i] * Pvap[i]) / self.P

        func = K
        f = Function('Kvalue2', [self.T, self.P, self.x], [func])
        return f(T, P, x)

    def Kvalue3(self, T, P, x):
        nc = self.NC
        K = SX.zeros(nc)
        Pvap = self.vapourPressure(self.T, self.P, self.x)
        gamma = self.liqActivityCoefficient(self.T, self.P, self.x)
        H = self.henryConstant(self.T, self.P, self.x)
        conc = self.liqDensityMol(self.T, self.P, self.x) * self.x

        K[0] = 1e-100
        K[1] = np.exp(
            -2276 + 30.32 * conc[1] + 11.65 * self.T - 0.3293 * conc[1] ** 2 - 0.01491 * self.T ** 2 - 0.07459 * conc[
                1] * self.T)  # np.exp(-1306 + (143.93*alpha) + (6.67*self.T) - (7.040*alpha*alpha) - (0.00852*self.T*self.T) - (0.3546*alpha*self.T))
        K[2] = (gamma[2] * Pvap[
            2]) / self.P  # np.exp(369 - (51.86*alpha) - (1.916*self.T) + (1.093*alpha*alpha) + (0.002479*self.T*self.T) + (0.1303*alpha*self.T))
        K[3] = (gamma[3] * Pvap[
            3]) / self.P  # np.exp(-37.58 + (1.584*alpha) + (0.1795*self.T) - (0.11589*alpha*alpha) - (0.000213*self.T*self.T) - (0.003869*alpha*self.T))

        func = K
        f = Function('Kvalue3', [self.T, self.P, self.x], [func])
        return f(T, P, x)

    def Kvalue4(self, T, P, x):
        Pvap = self.vapourPressure(self.T, self.P, self.x)
        alpha = self.x[1] / (self.x[2] + 1e-100)
        gamma = self.liqActivityCoefficient(self.T, self.P, self.x)
        gamma[1] = np.exp(-556.059 + 93.0892 * np.log(self.T) + 467.901 * alpha - 1.1964 * self.T * alpha)
        H = self.henryConstant(self.T, self.P, self.x)
        R = self.R
        nc = self.NC
        K = SX.zeros(nc)
        for i in range(nc):
            if (i == 1):
                K[i] = (H * gamma[i]) / (R * self.T * 1000.0)
            else:
                K[i] = (gamma[i] * Pvap[i]) / self.P

        func = K
        f = Function('Kvalue4', [self.T, self.P, self.x], [func])
        return f(T, P, x)

    def Kvalue5(self, T, P, x):
        x1 = self.moleFraction(self.x)
        nc = self.NC

        K = SX.zeros(nc)

        f1 = np.exp(
            385.738 - 64.49 * np.log(self.T) - 5.026 * x1[1] - 0.006 * self.T * x1[2] + 0.129 * self.T * x1[1] * x1[2])
        f2 = np.exp(490.131 - 82.103 * np.log(self.T) + 2.562 * x1[1] + 1.597 * x1[2] + 0.022 * self.T * x1[1] * x1[2])
        f3 = np.exp(531.525 - 89.121 * np.log(self.T) + 10.441 * x1[1] + 4.158 * x1[2] - 0.062 * T * x1[1] * x1[2])
        f4 = np.exp(
            505.895 - 84.832 * np.log(self.T) + 12.072 * x1[1] + 0.006 * self.T * x1[2] + 0.046 * self.T * x1[1] * x1[
                2])

        f5 = np.exp(
            -315.529 + 52.289 * np.log(self.T) - 3.471 * x1[1] - 0.003 * self.T * x1[2] + 0.112 * self.T * x1[1] * x1[
                2])
        f6 = np.exp(-341.477 + 56.65 * np.log(self.T) - 4.008 * x1[1] - 1.762 * x1[2] + 0.098 * self.T * x1[1] * x1[2])
        f7 = np.exp(
            -339.542 + 56.331 * np.log(self.T) - 3.901 * x1[1] - 0.003 * self.T * x1[2] + 0.026 * self.T * x1[1] * x1[
                2])

        z1 = if_else(logic_and(self.T >= 388, self.T <= 389), f3, f4)
        z2 = if_else(logic_and(self.T > 387, self.T <= 388), f2, z1)
        z3 = if_else(logic_and(self.T >= 385, self.T <= 387), f1, z2)

        z4 = if_else(logic_and(self.T >= 387, self.T <= 389), f6, f7)
        z5 = if_else(logic_and(self.T >= 385, self.T <= 387), f5, z4)

        K[0] = 0.0
        K[1] = f4
        K[2] = f7
        K[3] = -71.927 + 12.203 * np.log(self.T) + 0.34 * x1[1] + 0.265 * x1[3]

        func = K
        f = Function('Kvalue5', [self.T, self.P, self.x], [func])
        return f(T, P, x)

    def Kvalue6(self, T, P, x):
        x1 = self.moleFraction(self.x)
        nc = self.NC
        kk = 1
        K = SX.zeros(nc)

        f1 = np.exp(
            385.738 - 64.49 * np.log(self.T) - 5.026 * x1[1] - 0.006 * self.T * x1[2] + 0.129 * self.T * x1[1] * x1[2])
        f2 = np.exp(490.131 - 82.103 * np.log(self.T) + 2.562 * x1[1] + 1.597 * x1[2] + 0.022 * self.T * x1[1] * x1[2])
        f3 = np.exp(531.525 - 89.121 * np.log(self.T) + 10.441 * x1[1] + 4.158 * x1[2] - 0.062 * T * x1[1] * x1[2])
        f4 = np.exp(
            505.895 - 84.832 * np.log(self.T) + 12.072 * x1[1] + 0.006 * self.T * x1[2] + 0.046 * self.T * x1[1] * x1[
                2])

        f5 = np.exp(
            -315.529 + 52.289 * np.log(self.T) - 3.471 * x1[1] - 0.003 * self.T * x1[2] + 0.112 * self.T * x1[1] * x1[
                2])
        f6 = np.exp(-341.477 + 56.65 * np.log(self.T) - 4.008 * x1[1] - 1.762 * x1[2] + 0.098 * self.T * x1[1] * x1[2])
        f7 = np.exp(
            -339.542 + 56.331 * np.log(self.T) - 3.901 * x1[1] - 0.003 * self.T * x1[2] + 0.026 * self.T * x1[1] * x1[
                2])

        z1 = f3 + ((1 + np.tanh(kk * (T - 388.99999))) / 2) * (
                    f4 - f3)  # if_else(logic_and(self.T > 388, self.T <= 389), f3, f4)
        z2 = f2 + ((1 + np.tanh(kk * (T - 387.99999))) / 2) * (
                    z1 - f2)  # if_else(logic_and(self.T > 387, self.T <= 388), f2, z1)
        z3 = f1 + ((1 + np.tanh(kk * (T - 386.99999))) / 2) * (
                    z2 - f1)  # if_else(logic_and(self.T >= 385, self.T <= 387), f1, z2)

        z4 = f6 + ((1 + np.tanh(kk * (T - 388.99999))) / 2) * (
                    f7 - f6)  # if_else(logic_and(self.T > 387, self.T <= 389), f6, f7)
        z5 = f5 + ((1 + np.tanh(kk * (T - 386.99999))) / 2) * (
                    z4 - f5)  # if_else(logic_and(self.T >= 385, self.T <= 387), f5, z4)

        K[0] = 0.0
        K[1] = z3
        K[2] = z5
        K[3] = -71.927 + 12.203 * np.log(self.T) + 0.34 * x1[1] + 0.265 * x1[3]

        func = K
        f = Function('Kvalue6', [self.T, self.P, self.x], [func])
        return f(T, P, x)

    def vapourFraction(self, T, P, x, z):
        TT = MX.sym('TT')
        PP = MX.sym('PP')
        xx = MX.sym('xx')
        zz = MX.sym('zz')
        K = self.Kvalue1(T, P, x)
        # K = np.array([0,3.44764878289669,0.0172864141217366,1.02174783817333])
        z1 = self.moleFraction(z)
        #        func = -(((z1[1])/(K[3]+K[2]-1))+((z1[3]+z1[2])/(K[1]-1)))+1
        #        f = Function('VapourFraction', [self.T, self.P, self.x, self.z],[func])
        #        return f(T, P, x, z)
        v = SX.sym('v')
        res = sum1(z1 / ((1.0 / (K - 1.0)) + v))
        rf = Function('r', [v], [res])
        opts = {"abstol": 1e-14, "linear_solver": "csparse"}
        vfrac = rootfinder('s', 'kinsol', rf, opts)
        vv = MX.sym('vv')
        ff = vfrac(vv)
        rt = Function('rt', [vv], [ff])
        # v = fsolve(func, 0.3)
        return rt(0.3)

    #######################################################################
    # Gas phase properties 
    #######################################################################

    def gasDensityMol(self, T, P, x):
        R = self.R

        func = (self.P / (R * self.T)) * 0.001
        f = Function('gasDensityMol', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kmol/m^3

    def gasDensityMass(self, T, P, x):
        x1 = self.moleFraction(self.x)
        rho_mol = self.gasDensityMol(self.T, self.P, self.x)
        avgmw = self.avgMolWeight(self.x)

        func = rho_mol * avgmw
        f = Function('gasDensityMass', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kg/m^3

    def gasViscosityComp(self, T, P, x):
        x1 = self.moleFraction(self.x)
        mw = self.MW
        b = np.array([[0.000000656, 0.6081, 54.714], [0.000002148, 0.46, 290], [0.00000021602, 0.7105, 229.78],
                      [0.00000017851, 0.813, 304.72]])
        mu = (b[:, 0] * pow(self.T, b[:, 1])) / (1 + (b[:, 2] / self.T))

        func = mu
        f = Function('gasViscosityComp', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kg·m^−1·s^−1 or Pa.s

    def gasViscosityMixture(self, T, P, x):
        x1 = self.moleFraction(self.x)
        mw = self.MW
        mu = self.gasViscosityComp(self.T, self.P, self.x)
        nc = self.NC
        A = SX.zeros((nc, nc))
        for i in range(nc):
            for j in range(nc):
                if (j < i):
                    A[i, j] = ((mu[j] * mw[i]) / (mu[i] * mw[j])) * A[j, i]
                else:
                    A[i, j] = (pow((1 + (pow((mu[i] / mu[j]), 0.5)) * (pow(mw[j] / mw[i], 0.25))), 2)) / (
                        pow((8 * (1 + (mw[i] / mw[j]))), 0.5))
        mut = 0
        for i in range(nc):
            mut = mut + (x1[i] * mu[i] / (sum1(x1 * A[i, :].T)+1E-100))

        func = mut
        f = Function('gasViscosityMixture', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kg·m^−1·s^−1 or Pa.s

    def gasDiffusivity(self, T, P, x):
        x1 = self.moleFraction(self.x)
        nc = self.NC
        mw = self.MW
        P = P * 0.00001
        dv = np.array([18.5, 26.9, 58.62, 13.1])
        MWij = SX.zeros((nc, nc))
        Dij = SX.zeros((nc, nc))
        for i in range(nc):
            for j in range(nc):
                MWij[i, j] = (2 * mw[i] * mw[j]) / (mw[i] + mw[j])
                Dij[i, j] = (0.00143 * pow(self.T + 1E-100, 1.75)) / (P * pow(MWij[i, j] + 1E-100, 0.5) * (
                    pow((pow(dv[i] + 1E-100, 0.333) + pow(dv[j] + 1E-100, 0.333)), 2.0)))
        Dg = SX.zeros(nc)
        for i in range(nc):
            xjDij = 0
            for j in range(nc):
                if (i != j):
                    xjDij = xjDij + (x1[j] * Dij[i, j])
            Dg[i] = (1 / (1 - x1[i]+1E-100)) * xjDij * 0.0001

        func = Dg
        f = Function('gasDiffusivity', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # m^2/s

    def gasAvgDiffusivity(self, T, P, x):
        D = self.gasDiffusivity(self.T, self.P, self.x)
        x1 = self.moleFraction(self.x)

        func = sum1(D * x1)
        f = Function('gasAvgDiffusivity', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # m^2/s

    def gasThermalConductivity(self, T, P, x):
        x1 = self.moleFraction(self.x)
        mw = self.MW
        nc = self.NC
        b = np.array(
            [[0.00033143, 0.7722, 16.323, 373.72], [3.69, -0.3838, 964, 1860000.0], [-0.0011442, 0.6373, -2418.1, 0.0],
             [0.0000693, 1.1254, 847.68, -150000.0]])
        cond = (b[:, 0] * pow(self.T, b[:, 1])) / (1 + (b[:, 2] / self.T) + (b[:, 3] / (pow(self.T, 2.0))))
        A = SX.zeros((nc, nc))
        for i in range(nc):
            for j in range(nc):
                if (j < i):
                    A[i, j] = ((cond[j] * mw[i]) / (cond[i] * mw[j])) * A[j, i]
                else:
                    A[i, j] = (pow((1 + (pow((cond[i] / cond[j]), 0.5)) * (pow(mw[j] / mw[i]+1E-100, 0.25))), 2.0)) / (
                        pow((8 * (1 + (mw[i] / mw[j])))+1E-100, 0.5))
        condt = 0
        for i in range(nc):
            condt = condt + (x1[i] * cond[i] / sum1(x1 * A[i, :].T))

        func = condt * 0.001
        f = Function('gasThermalConductivity', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kW/m/K or kJ/s/m/K

    def gasHeatCapacityMol(self, T, P, x):
        b = np.array([[31149.792, -13.565232, 0.02679552, -0.0000117, 0.0, 0.0],
                      [19795, 73.436472, -0.056019, 0.0000172, 0.0, 0.0],
                      [13207.4, 281.577, -0.1513066, 0.0000313, 0.0, 0.0],
                      [33738.112, -7.0175634, 0.0272961, -0.0000167, 0.0000000043, -0.000000000000417]])

        func = ((b[:, 0]) + (b[:, 1] * self.T) + (b[:, 2] * self.T ** 2) + (b[:, 3] * self.T ** 3) + (
                    b[:, 4] * self.T ** 4)) * 0.001
        f = Function('gasHeatCapacityMol', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kJ/kmol/K

    def gasAvgHeatCapacityMol(self, T, P, x):
        Cpmol = self.gasHeatCapacityMol(self.T, self.P, self.x)
        x1 = self.moleFraction(self.x)

        func = sum1(Cpmol * x1)
        f = Function('gasAvgHeatCapacityMol', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kJ/kmol/K

    def gasAvgHeatCapacityMass(self, T, P, x):
        Cpmolavg = self.gasAvgHeatCapacityMol(self.T, self.P, self.x)
        mwavg = self.avgMolWeight(x)

        func = Cpmolavg / mwavg
        f = Function('gasAvgHeatCapacityMass', [self.T, self.P, self.x], [func])
        return f(T, P, x)  # kJ/kg/K
