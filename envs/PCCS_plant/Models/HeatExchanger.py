# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 12:02:17 2017

@author: decardin
"""
import numpy as np
from ..Models import Stream
from ..Properties import Properties
from copy import deepcopy
from casadi import *


class HeatExchanger:
    
    # Tube side parameters
    ID_tube = 0.01483
    OD_tube = 0.01905
    rho_wall = 7817.0
    Cp_wall = 461.0
    U_tube = 850.0
    nTubes = 30.0
    LTube = 3.0
    V_tube = (np.pi*ID_tube*ID_tube/4.0)*LTube*nTubes
    k = 16.2
    
    # Shell side parameters
    LDsR = 7.0
    D_shell = LTube/LDsR
    U_shell = 850.0
    V_shell = ((np.pi*D_shell*D_shell/4.0)*LTube) - V_tube
    
    # Streams
    tubein = Stream.Stream()
    tubeout = Stream.Stream()
    shellin = Stream.Stream()
    shellout = Stream.Stream()
    
    # Nx = 2; Nz = 5; u = d = 0
    
    #def __init__(self):
        
    def getValue(self, x, u, d):
        props = Properties.Properties()
        Cp_tube = props.liqAvgHeatCapacityMol((self.tubein.T + x[0])/2, self.tubein.P, self.tubein.comp)
        rho_tube = props.liqDensityMol((self.tubein.T + x[0])/2, self.tubein.P, self.tubein.comp)
        Cp_shell = props.liqAvgHeatCapacityMol((self.shellin.T + x[1])/2, self.shellin.P, self.shellin.comp)
        rho_shell = props.liqDensityMol((self.shellin.T + x[1])/2, self.shellin.P, self.shellin.comp)
        Tlmtd = ((x[0] - self.shellin.T) - (self.tubein.T - x[1]))/np.log((x[0] - self.shellin.T)/(self.tubein.T - x[1]+1e-100))
        UA = self.nTubes/((1/(self.U_tube*np.pi*self.ID_tube*self.LTube)) + (1/(self.U_shell*np.pi*self.OD_tube*self.LTube+1e-100)) + (np.log(fmax((self.OD_tube/self.ID_tube),1e-100))/(2*np.pi*self.LTube*self.k)) )
        Q = 0.001*UA*Tlmtd
        
        xdot = deepcopy(x)
        xdot[0] = (self.tubein.Q/self.V_tube)*(self.tubein.T - x[0]) - (Q/((Cp_tube*rho_tube*self.V_tube)+1e-100))
        xdot[1] = (self.shellin.Q/self.V_shell)*(self.shellin.T - x[1]) + (Q/((Cp_shell*rho_shell*self.V_shell)+1e-100))
        
        self.shellout = deepcopy(self.shellin)
        self.shellout.T = x[1]
        
        self.tubeout = deepcopy(self.tubein)
        self.tubeout.T = x[0]
        return xdot
    
    def getValueF(self, x, z, u, d):        
        
        Cp_tube = z[0]
        rho_tube = z[1]
        Cp_shell = z[2]
        rho_shell = z[3]
        Q = z[4]
        
        xdot = x
        xdot[0] = (self.tubein.Q/self.V_tube)*(self.tubein.T - x[0]) - (Q/((Cp_tube*rho_tube*self.V_tube)))
        xdot[1] = (self.shellin.Q/self.V_shell)*(self.shellin.T - x[1]) + (Q/((Cp_shell*rho_shell*self.V_shell)))
        
        return xdot
    
    def getValueG(self, x, z, u, d):
        
        props = Properties.Properties()
        
        #a = np.zeros(5)
        
        Cp_tube = props.liqAvgHeatCapacityMol((self.tubein.T + x[0])/2, self.tubein.P, self.tubein.comp)
        rho_tube = props.liqDensityMol((self.tubein.T + x[0])/2, self.tubein.P, self.tubein.comp)
        Cp_shell = props.liqAvgHeatCapacityMol((self.shellin.T + x[1])/2, self.shellin.P, self.shellin.comp)
        rho_shell = props.liqDensityMol((self.shellin.T + x[1])/2, self.shellin.P, self.shellin.comp)
        Tlmtd = ((x[0] - self.shellin.T) - (self.tubein.T - x[1]))/np.log((x[0] - self.shellin.T)/(self.tubein.T - x[1]))
        UA = self.nTubes/( (1/(self.U_tube*np.pi*self.ID_tube*self.LTube) ) + (1/(self.U_shell*np.pi*self.OD_tube*self.LTube)) + (np.log(self.OD_tube/self.ID_tube)/(2*np.pi*self.LTube*self.k)) )
        Q = 0.001*UA*Tlmtd
        
        return np.array([z[0] - Cp_tube, z[1] - rho_tube, z[2] - Cp_shell, z[3] - rho_shell, z[4] - Q])