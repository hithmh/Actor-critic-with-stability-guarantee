# -*- coding: utf-8 -*-
"""
Created on Wed Jul 05 16:27:48 2017

@author: decardin
"""

import numpy as np
import Column
import Stream
from Utilities.RK4Integrator import rk4
from copy import copy
import mpctools as mpc

absorber = Column.Column(6.1, 0.43, "absorber")

fluegasComposition = np.array([0.03571, 0.00781, 0, 0.00112])
fluegas = Stream.Stream(0.0898912, 319.7, 103825, fluegasComposition);

xLIn = np.array([0,1.36,5,39.09]) 
leanAmine = Stream.Stream(6.292384000000001E-4, 314, 101325, xLIn)

absorber.liqin = copy(leanAmine)
absorber.gasin = copy(fluegas)

x = SX.sym('x',50)
u = SX.sym('u')
d = SX.sym('d')

ode = absorber.getValue(x, u, d)

dae = {'x':x, 'p':vertcat(u,d), 'ode':ode}

f = integrator('f', 'idas', dae)

x0 = np.array([0,0,0,0,0,1.36,1.36,1.36,1.36,1.36,5,5,5,5,5,39.09,39.09,39.09,39.09,39.09,314,314,314,314,314,0.03372,0.03372,0.03372,0.03372,0.03372,0.00715,0.00715,0.00715,0.00715,0.00715,0,0,0,0,0,0.00112,0.00112,0.00112,0.00112,0.00112,319.7,319.7,319.7,319.7,319.7])

r = f(x0=x0, p = vertcat(6.292384000000001E-4, 0))
print(r['xf'])

Nx = x0.size
Nu = 1
Nd = 1

#
nn = 3600
xx = np.zeros((nn,Nx))
#zz = np.zeros((nn,Nz))
xx[0] = x0
#zz[0] = z0
#
for i in range(1,nn):
    print '========================================'
    print i
    xnext = f(x0=xx[i-1], p = vertcat(6.292384000000001E-4,0))
    xx[i] = xnext['xf'].T
#    zz[i] = xnext['zf'].T
#    print f.jacobian()
print(xnext['xf'])
##    # print(sim.sim(x0, 0, 0))
##    xf = sim.sim(xx[i-1], U, 0)
#
##    xx[i] = xnext
#print(xnext['zf'])

##x0 = np.array([0,0,0,0,0,1.44459227487252,1.60146857274116,1.84822417743143,2.11025008688042,2.34563402351909,5.00032350345555,5.00076639488095,5.00099893104737,5.00054982050483,4.99923714283984,39.2310864573074,39.4125109706027,39.5078681785470,39.3496358863361,38.7486840123154,317.650833025950,323.304585662174,329.780096026845,333.508356821323,331.425442350628,0.0357099999999999,0.0357100000000000,0.0357100000000000,0.0357100000000000,0.0357100000000000,0.000709853411044116,0.00131849544824657,0.00246411692862605,0.00426158622071689,0.00614346760148703,4.64343140479811e-06,7.00820711382569e-06,1.03187329247432e-05,1.21476616138372e-05,9.12381956171690e-06,0.00324896739897706,0.00427690382417716,0.00562454145867799,0.00636289294228698,0.00530046203330021,318.621724572462,324.291217021985,330.225922691497,332.998377948704,329.784281927166])
##xxx = absorber.getValue(x0,0,0)
##t, xx = rk4(absorber.getValue, 0, 3600, 0.5, x0, 0, 0)
#Delta = 0.5
#Nx = x0.size
#Nu = 1
#Nd = 1
##
### Create discrete simulator
#nn = 3600
#xx = np.zeros((nn,Nx))
#xx[0] = x0
#sim = mpc.DiscreteSimulator(absorber.getValue, Delta, [Nx, Nu, Nd], ["x", "u", "d"])
#for i in range(1,nn):
#    #print(sim.sim(x0, 0, 0))
#    xnext = sim.sim(xx[i-1], 0, 0)
#    xx[i] = xnext
#    # print(xnext)
#
##def groupStates(ns, nc, x):
##        
##    xL = np.zeros((ns+1,nc))
##    xG = np.zeros((ns+1,nc))
##    TL = np.zeros(ns+1)
##    TG = np.zeros(ns+1)
##        
##    for i in range(nc):
##        xL[1:ns+1,i] = x[(i)*ns:((i+1)*ns)]
##        xG[0:ns,i] = x[(i)*ns+(nc*ns)+ns:((i+1)*ns+(nc*ns)+ns)]
##    TL[1:ns+1] = x[(5-1)*ns:(5*ns)]
##    TG[0:ns] = x[(5-1)*ns+(nc*ns)+ns:(5*ns+(nc*ns)+ns)]
##        
##    return xL, TL, xG, TG
##
##xL, TL, xG, TG = groupStates(5, 4, x0)