# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 03:02:57 2017

@author: decardin
"""

import numpy as np
import Reboiler
import Stream
from Utilities.RK4Integrator import rk4
from copy import copy
import mpctools as mpc
from casadi import *

reboiler = Reboiler.Reboiler()

shellincomp = np.array([0.0, 2.367149984438293, 4.99916910874868, 38.7018863262854])
liqin = Stream.Stream(6.292384000000001E-4,330.7,103825.0,shellincomp)

Q = SX.sym('Q')
T = SX.sym('T')
P = SX.sym('P')
x = SX.sym('x',4)
# liqin = Stream.Stream(Q, T, P, x)

reboiler.liqin = liqin

# x0 = np.array([0, 1.36, 5, 39.09, 387])


U = np.array([250])

## Symbolic variables for DAE system
x = SX.sym('x')
z = SX.sym('z', 13)
u = SX.sym('u')
d = SX.sym('d')

ode = reboiler.getValueO(x, z, u, d)
alg = reboiler.getValueA(x, z, u, d)
dae = {'x':x, 'z':z, 'p':vertcat(u,d) ,'ode':ode, 'alg':alg}
opts = {"tf":50} # interval length
f = integrator('f', 'idas', dae, opts)

x0 = np.array([389])
z0 = np.array([0, 0.0147604103, 0.0504165221, 0.934823068, 0, 0.0841595048, 
      0.000826652134, 0.915013843, 0, 2.31465854662427, 0.0201259574490178, 
      1.06789481184371, 0.299649717])
#z0 = np.zeros(13)

r = f(x0=x0, z0=vertcat(*z0), p = vertcat(U, 0))
print(r['xf'])
# f = Function('f', [x, z], [ode, alg], ['x', 'z'], ['ode', 'alg'])
#
#Delta = 0.5
Nx = x0.size
Nu = 1
Nd = 1
Nz = 13
##
#print (reboiler.getValueA(x0, z0, U, 0))
#
### Create discrete simulator
nn = 500
xx = np.zeros((nn,Nx))
zz = np.zeros((nn,Nz))
xx[0] = x0
zz[0] = z0
#sim = mpc.DiscreteSimulator(reboiler.getValue, Delta, [Nx, Nu, Nd], ["x", "u", "d"])
for i in range(1,nn):
    print i
    xnext = f(x0=xx[i-1], z0=vertcat(*zz[i-1]), p = vertcat(U,0))
    xx[i] = xnext['xf'] 
    zz[i] = (xnext['zf'].T);
    print(xnext['xf'])
#    # print(sim.sim(x0, 0, 0))
#    xf = sim.sim(xx[i-1], U, 0)

#    xx[i] = xnext
print(xnext['zf'])