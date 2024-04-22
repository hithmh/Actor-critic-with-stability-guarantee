# -*- coding: utf-8 -*-
"""
Created on Tue Aug 01 02:39:41 2017

@author: decardin
"""

from casadi import *
import numpy as np

class Solver:
    
    def __init__(self, N=None):
        self.c = N["c"]
        self.Nx = N["Nx"]
        self.Nz = N["Nz"]
        self.Nk = N["Nk"]
        self.Nt = N["Nt"]
        self.Nu = N["Nu"]
        
        # Declare variables
        self.t = SX.sym('t')     # time
        self.u = SX.sym('u',self.Nu)  # Control inputs
        self.xd = SX.sym('xd',self.Nx)  # Differential states
        self.xz = SX.sym('xz',self.Nz)  # Algebraic states
    
    def colloc(self, c, Nt, Nk):
        
        nicp = 1

        # Degree of intepolating polynomial
        deg = c - 1
        
        # Radau collocation points
        cp = "radau"
        
        # Size of finite elements
        h = Nt/Nk/nicp
        
        # Coefficients of the collocation equation
        C = np.zeros((deg+1,deg+1))
        
        # Coefficients of the continuity equation
        D = np.zeros(deg+1)
        
        # Collocation point
        tau = SX.sym("tau")
        
        # All collocation time points
        tau_root = [0] + collocation_points(deg, cp)
        
        Tt = np.zeros((Nk,deg+1))
        for i in range(Nk):
            for j in range(deg+1):
                Tt[i][j] = h*(i + tau_root[j])
                
        # For all collocation points: eq 10.4 or 10.17 in Biegler's book
        # Construct Lagrange polynomials to get the polynomial basis at the collocation point
        for j in range(deg+1):
            L = 1
            for j2 in range(deg+1):
                if j2 != j:
                    L *= (tau-tau_root[j2])/(tau_root[j]-tau_root[j2])
        
            # Evaluate the polynomial at the final time to get the coefficients of the continuity equation
            lfcn = Function('lfcn', [tau],[L])
            D[j] = lfcn(1.0)
        
            # Evaluate the time derivative of the polynomial at all collocation points to get the coefficients of the continuity equation
            tfcn = Function('tfcn', [tau],[tangent(L,tau)])
            for j2 in range(deg+1):
                C[j][j2] = tfcn(tau_root[j2])
                
        return C, D, Tt
        
        
        