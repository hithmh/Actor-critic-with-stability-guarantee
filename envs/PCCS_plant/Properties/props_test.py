# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:45:04 2017

@author: decardin
"""

from casadi import *
import Properties
import numpy as np
# from scipy.optimize import fsolve

props = Properties.Properties()
x = np.array([0,1.36,5,39.09])
x1 = np.array([0,2.5,5,39.09])
x2 = np.array([0,1.36,5,39.09])
xg = np.array([0, 1, 0, 0])
 

# y = props.a((298.15+298.15)/2, 101325, x2)
#y
#print y
xz = SX.sym('xz',4)
xx = SX.sym('x',4)
T = SX.sym('T')
P = SX.sym('P')

xt = np.array([0.03571, 0.00781, 0, 0.00112])
#func = props.avgMolWeight(xz)
#f = Function('f',[xx], [func]) # for testing mole fractions
#print(f(x))
#func = props.vapourFraction(387, 101325, x2, x1)
#func = props.vapourFraction(T, P, x2, x1)
xx = np.array([0,0.008884, 0.000081, 0.021233])
x = np.array([1.5e-100, 1.5, 5, 39.09])
#func = props.gasAvgHeatCapacityMol(385, 150000, xg)
func = props.liqConcentration(300, 101325, x2)
#zz = np.zeros(7)
#for i in range(7):
#    T = 385 + i
#    zz[i] = props.henryConstant(T, 170000, x)
    
    
    
#f = Function('f', [T, P, xx], [func])
#print(f(385, 170000, x2,x1))
# print(f(T, P, xx))
# func = props.gasViscosity(298.15, 101325, xg)
print(func)
