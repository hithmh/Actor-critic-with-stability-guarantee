# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 16:29:45 2017

@author: decardin
"""
import numpy as np
from ..Models import Stream
from ..Properties import Properties
from casadi import *
from copy import deepcopy


class Reboiler:

    def __init__(self):
        # Streams
        self.liqin = Stream.Stream()
        self.liqout = Stream.Stream()
        self.gasout = Stream.Stream()
        self.DHRXN = 82000
        # self.D = 6.1
        self.D = 0.43
        self.Preb = 160000
        self.Lreb = 1
        self.A = (np.pi * self.D ** 2) / 4
        self.V = self.A * self.Lreb
        self.hout = 0.4300
        self.Tamb = 273.15 + 25
        self.Dc = np.pi * self.D * self.Lreb

    def getValueO(self, x, z, u, d):
        props = Properties.Properties()
        # Preb = self.Preb
        # Lreb = self.Lreb
        # D = self.D
        # A = self.A
        # V = self.V

        Tin = self.liqin.T
        concin = self.liqin.comp
        Pin = self.liqin.P
        Q = self.liqin.Q  # xm = x[0:4]
        T = x[0]
        Qreb = u
        P = self.Preb
        zin = props.moleFraction(concin)

        # v, K, x1, y = self.Flash(T, P, xm, z)
        xi = z[0:4]
        v = z[4]  # K = z[8:12]; ; yi = z[4:8]

        K = props.Kvalue6(T, P, xi)
        yi = K * xi

        # Since hold up is constant xi can be calculated directly from the material balance equation
        rhoin = props.liqDensityMol(Tin, Pin, zin)
        xm = np.array([0, 1.36, 5, 39.09])
        # xi = ((zin - (v*yi)))/(1-v)
        # K = props.Kvalue1(T, P, xi)

        # xdot = SX.zeros(1)
        Cpin = props.liqAvgHeatCapacityMol(Tin, Pin, zin)
        Cpx = props.liqAvgHeatCapacityMol(T, P, xi)
        Cpy = props.gasAvgHeatCapacityMol(T, P, yi)
        Cpm = props.liqAvgHeatCapacityMol(T, P, xi)

        rhox = props.liqDensityMol(T, P, xi)
        rhoy = props.gasDensityMol(T, P, yi)
        rhom = props.liqDensityMol(T, P, xi)
        DHVAP = props.liqVapourizationHeat(T, P, xi)

        q = ((self.DHRXN * yi[1]) + (DHVAP[3] * yi[3]) + DHVAP[2] * yi[2])
        xdot = ((Q * rhoin * (Cpin * (Tin - 0) - v * Cpy * (T - 0) - (1 - v) * Cpx * (
                T - 0) - v * q)) + Qreb) / (
                       self.V * Cpm * rhom + 1e-100)

        self.liqout.Q = Q * rhoin * (1.0 - v) / (rhox + 1e-100)
        self.liqout.T = T
        self.liqout.P = P
        self.liqout.comp = rhox * xi

        self.gasout.Q = (Q * rhoin * v) / (rhoy + 1e-100)
        self.gasout.T = T
        self.gasout.P = P
        self.gasout.comp = rhoy * yi

        return xdot

    def getValueA(self, x, z, u, d):
        props = Properties.Properties()

        concin = deepcopy(self.liqin.comp)  # Pin = self.liqin.P;
        T = x[0]
        xi = z[0:4]
        vf = z[4]  # K = z[8:12]; ; yi = z[4:8];
        P = self.Preb
        zi = props.moleFraction(concin)
        K = props.Kvalue6(T, P, xi)
        res1 = xi - (zi / (1 + vf * (K - 1) + 1e-100))
        res2 = sum1((zi * (K - 1)) / (1 + (vf * (K - 1)) + 1e-100))

        return deepcopy(vertcat(res1, res2))

        # res = sum1(z1/((1/(K - 1)) + vf))
