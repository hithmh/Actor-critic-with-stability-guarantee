# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 10:42:21 2017

@author: decardin
"""

import numpy as np
from ..Models import Stream as Stream
from ..Properties import Properties
from casadi import *


class Column:

    def __init__(self, Height, Diameter, colType):

        # Properties
        self.nComponents = 4
        self.R = 8.314
        self.g = 9.80665
        self.DHRXN = 82000.0
        self.aP = 143.9
        self.dP = 0.038

        # Column specs
        self.Height = Height
        self.Diameter = Diameter
        self.colType = colType
        self.nStages = 5
        self.dz = Height / self.nStages
        self.A = (np.pi * Diameter ** 2.0) / 4.0
        self.Ac = np.pi * self.dz * self.Diameter
        
        self.liqin = Stream.Stream()
        self.liqout = Stream.Stream()
        self.gasin = Stream.Stream()
        self.gasout = Stream.Stream()

    def groupStates(self, ns, nc, x):

        xL = SX.zeros((ns + 1, nc))
        xG = SX.zeros((ns + 1, nc))
        TL = SX.zeros(ns + 1)
        TG = SX.zeros(ns + 1)

        for i in range(nc):
            xL[1:ns + 1, i] = x[(i) * ns:((i + 1) * ns)]
            xG[0:ns, i] = x[(i) * ns + (nc * ns) + ns:((i + 1) * ns + (nc * ns) + ns)]
        TL[1:ns + 1] = x[nc * ns:(5 * ns)]
        TG[0:ns] = x[nc * ns + (nc * ns) + ns:(5 * ns + (nc * ns) + ns)]

        return xL, TL, xG, TG

    def gasMassTransferCoefficient(self, rho, u, aP, mu, diff, dP, T):
        return 5.23 * (pow(((rho * u) / (aP * mu)) + 1e-100, 0.7)) * (pow((mu / (rho * diff)) + 1e-100, 0.333)) * (
            pow(diff / (aP * dP * dP * self.R * T), 1.0))

    def liqMassTransferCoefficient(self, rho, u, aP, mu, diff, dP, aW):
        return 0.0051 * (pow(((mu * self.g) / rho) + 1e-100, 0.333)) * (
            pow(((rho * u) / (aW * mu)) + 1e-100, 0.667)) * (pow((mu / (rho * diff)) + 1e-100, -0.5)) * (
                   pow((aP * dP) + 1e-100, 0.4))

    def wettedSurfaceArea(self, rho, u, aP, mu, dP):
        L = rho * u
        return aP * (1.0 - np.exp(-1.45 * (pow((L / (mu * aP)) + 1e-100, 0.1)) * (
            pow(((aP * L * L) / (self.g * rho * rho)) + 1e-100, -0.05)) * (
                                      pow(((L * L) / (rho * aP * 0.04)) + 1e-100, 0.2)) * (
                                      pow((0.075 / 0.04) + 1e-100, 0.75))))

    def getValue(self, x, u, d):
        #
        # General properties
        props = Properties.Properties()
        aP = self.aP
        dP = self.dP
        pd = (self.gasin.P - self.liqin.P)
        nc = self.nComponents
        ns = self.nStages

        # Group states into phases and assign inlets
        xL, TL, xG, TG = self.groupStates(ns, nc, x)
        xL[0, :] = self.liqin.comp[:nc]
        TL[0] = self.liqin.T
        xG[ns, :] = self.gasin.comp[:nc]
        TG[ns] = self.gasin.T

        # Create holders for derivaties
        xLdot = SX.zeros((ns, nc))
        xGdot = SX.zeros((ns, nc))
        TLdot = SX.zeros(ns)
        TGdot = SX.zeros(ns)

        for i in range(ns):
            # General properties
            P = self.liqin.P + (pd / ns) * (i)

            # Liquid phase properties
            muL = props.liqViscosityMixture(TL[i + 1], P, xL[i + 1, :].T)
            rhoL_mass = props.liqDensityMass(TL[i + 1], P, xL[i + 1, :].T)
            rhoL_mol = props.liqDensityMol(TL[i + 1], P, xL[i + 1, :].T)
            uL = self.liqin.Q / self.A
            diffL = props.liqDiffusivity(TL[i + 1], P, xL[i + 1, :].T)
            aGL = self.wettedSurfaceArea(rhoL_mass, uL, aP, muL, dP)
            k2 = 97700000000 * np.exp(-4955 / TL[i + 1])
            He = props.henryConstant(TL[i + 1], P, xL[i + 1, :].T)
            DHVAP = props.liqVapourizationHeat(TL[i + 1], P, xL[i + 1, :].T)
            Ps = props.vapourPressure(TL[i + 1], P, xL[i + 1, :].T)
            gamma = props.liqActivityCoefficient(TL[i + 1], P, xL[i + 1, :].T)
            concLf = props.liqConcentration(TL[i + 1], P, xL[i + 1,
                                                          :].T)  # np.array([1.578946472, 0.000308133]) [2.34045046580154,3.24208335748558e-05]
            D_MEACOO = props.liqDiffusivityCarbamate(TL[i + 1], P, xL[i + 1, :].T)
            KeqOverall = props.KeqOverall(TL[i + 1], P, xL[i + 1, :].T)

            # Gas phase properties
            muG = props.gasViscosityMixture(TG[i], P, xG[i, :].T)
            rhoG_mol = props.gasDensityMol(TG[i], P, xG[i, :].T)
            rhoG_mass = props.gasDensityMass(TG[i], P, xG[i, :].T)
            uG = self.gasin.Q / self.A
            diffG = props.gasDiffusivity(TG[i], P, xG[i, :].T)
            lambdaG = props.gasThermalConductivity(TG[i], P, xG[i, :].T)

            # Calculate mass transfer coefficients and component partial pressures
            kg = SX.zeros(nc)
            kl = SX.zeros(nc)
            Pe = SX.zeros(nc)

            for j in range(nc):
                kg[j] = self.gasMassTransferCoefficient(rhoG_mass, uG, aP, muG, diffG[j], dP, TG[i])
                kl[j] = self.liqMassTransferCoefficient(rhoL_mass, uL, aP, muL, diffL[j], dP, aGL)
                if j == 1:
                    Pe[j] = gamma[j] * concLf[1] * He
                else:
                    Pe[j] = Ps[j] * (xL[i + 1, j] / rhoL_mol + 1e-100) * gamma[j]

            # kg = np.array([1.07566712194441e-05,8.36936918459324e-06,6.61170204091475e-06,1.11922853032279e-05])
            # kl = np.array([9.36514879856339e-05,0.000101967104028575,7.34912001124136e-05,0.00000000000000000001])

            CpGavmass = props.gasAvgHeatCapacityMass(TG[i], P, xG[i, :].T)
            CpGavmol = props.gasAvgHeatCapacityMol(TG[i], P, xG[i, :].T)
            CpLavmol = props.liqAvgHeatCapacityMol(TL[i + 1], P, xL[i + 1, :].T)
            Dgav = sum1(xG[i, :].T * diffG) / rhoG_mol
            kgav = sum1(xG[i, :].T * kg) / rhoG_mol
            hgl = kgav * self.R * TG[i] * pow(CpGavmass * rhoG_mass + 1e-100, 0.333) * pow((lambdaG / Dgav) + 1e-100,
                                                                                           0.667)

            if self.colType == 'absorber':
                E = np.sqrt(fmax(k2 * concLf[0] * diffL[1], 1E-100)) / (kl[1] + 1e-100)
            else:
                E = 1.0 + (((D_MEACOO / diffL[1] + 1e-100) * (np.sqrt(fmax(KeqOverall, 1E-100))) * 0.001 * concLf[
                    0]) / (1 + (
                        (2 * D_MEACOO / diffL[2]) * (np.sqrt(fmax(KeqOverall * 0.001 * concLf[1], 1E-100))))) * (
                                   np.sqrt(fmax(0.001 * concLf[1], 1E-100)) + np.sqrt(fmax(0.001 * concLf[1], 1E-100))))

            KG = SX.zeros(nc)
            N = SX.zeros(nc)
            for j in range(nc):
                if j == 0:
                    KG[j] = 0
                elif j == 1:
                    KG[j] = (1 / ((1 / (0.001 * kg[j] + 1e-100)) + (He / (E * kl[j])) + 1e-100))
                else:
                    KG[j] = 1 / (1 / (0.001 * kg[j]) + 1e-100)
                N[j] = KG[j] * (((xG[i, j] / (rhoG_mol + 1e-100)) * P) - Pe[j])

            # Calculate derivatives
            for j in range(nc):
                xLdot[i, j] = -uL * ((xL[i + 1, j] - xL[i, j]) / self.dz) + (aGL * (N[j]))
                xGdot[i, j] = -uG * ((xG[i, j] - xG[i + 1, j]) / self.dz) - (aGL * (N[j]))
            qL = -hgl * (TL[i + 1] - TG[i]) + ((self.DHRXN * N[1]) + (DHVAP[3] * N[3])) - (
                        0.430 * 0.003 * (TL[i + 1] - 293.15))
            qG = hgl * (TL[i + 1] - TG[i])
            TLdot[i] = -uL * ((TL[i + 1] - TL[i]) / self.dz) + ((aGL * qL) / (CpLavmol * rhoL_mol + 1e-100))
            TGdot[i] = -uG * ((TG[i] - TG[i + 1]) / self.dz) + ((aGL * qG) / (CpGavmol * rhoG_mol + 1e-100))

        # Update output streams
        self.liqout.Q = self.liqin.Q
        self.liqout.T = TL[ns]
        self.liqout.P = self.gasin.P
        self.liqout.comp[:nc] = xL[ns, :].T
        self.gasout.Q = self.gasin.Q
        self.gasout.T = TG[0]
        self.gasout.P = self.liqin.P
        self.gasout.comp[:nc] = xG[0, :].T

        # Reshape arrays
        xLdot = reshape(xLdot, (-1, 1))
        xGdot = reshape(xGdot, (-1, 1))
        TLdot = reshape(TLdot, (-1, 1))
        TGdot = reshape(TGdot, (-1, 1))
        xdot = vertcat(xLdot, TLdot, xGdot, TGdot)
        return xdot
