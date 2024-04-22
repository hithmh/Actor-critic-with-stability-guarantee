# -*- coding: utf-8 -*-
"""
Created on Tue Jul 04 11:13:53 2017

@author: decardin
"""
import numpy as np

# This solves an ODE using the 4th order Runge-Kutta method (RK4) for IVPs

def rk4(f, t0, tf, h, x0, u, d):
    """
    This uses 4th order runge kutta method to numerically integrate an ode.
    f = the ode to be integrated
    t0 = initial time
    tf = final time
    h = step size
    u = all controlled inputs
    d = all disturbances
    """
    n = int(np.round((tf-t0)/h))
    x = np.zeros((n+1,x0.size))
    t = np.zeros(n+1)
    
    t[0] = t0
    x[0] = x0
    
    for i in range(n):
        t[i+1] = t[i] + h
        k1 = f(x[i], u, d)
        k2 = f(x[i] + (k1*(h/2)), u, d)
        k3 = f(x[i] + (k2*(h/2)), u, d)
        k4 = f(x[i] + (k3*h), u, d)
        x[i+1] = x[i] + ((h/6)*(k1 + (2*(k2+k3)) + k4))
    return t, x