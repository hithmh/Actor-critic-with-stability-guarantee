# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 16:34:50 2017

@author: decardin
"""
from casadi import *
import mpctools as mpc
import numpy as np
import HeatExchanger
import Stream
from Utilities.RK4Integrator import rk4

Delta = 1
Nx = 2
Nz = 5
Nu = 1
Nd = 1

hrx = HeatExchanger.HeatExchanger()
x0 = np.array([350., 350]) # x[0] = tube out/lean amine, x[1] = shell out/rich amine
z0 = np.array([88.6215, 43.243, 84.5749, 45.1368,4])
x = np.array([0,1.36,5,39.09])
tubein = Stream.Stream(6.292384000000001E-4, 389, 150000, np.array([0,1.36,5,39.09]))
shellin = Stream.Stream(6.292384000000001E-4, 330.7, 101325, np.array([0,2.36,5,39.09]))
hrx.shellin = shellin
hrx.tubein = tubein

x = SX.sym('x',Nx)
z = SX.sym('z',Nz)
#
#ode = hrx.getValueF(x,z,0,0)
#alg = hrx.getValueG(x,z,0,0)
#dae = {'x':x, 'z':z, 'ode':ode, 'alg':alg}
#F = integrator('F', 'idas', dae)
#r = F(x0=x0, z0 = z0)
#print(r['xf'])
#f = Function('f', [x, z], [ode, alg], ['x', 'z'], ['ode', 'alg'])

# Create discrete simulator
nn = 3600
xx = np.zeros((nn,2))
xx[0] = x0
sim = mpc.DiscreteSimulator(hrx.getValue, Delta, [Nx, Nu, Nd], ["x", "u", "d"])
for i in range(1,nn):
    #print(sim.sim(x0, 0, 0))
    xnext = sim.sim(xx[i-1], 0, 0)
    xx[i] = xnext
    print(xnext)

#t, xx = rk4(hrx.getValue, 0, 3600, 0.5, x0, 0, 0)

#print(xx)