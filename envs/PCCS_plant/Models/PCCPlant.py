# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 14:58:17 2017

@author: decardin

This file constains all the units in the plant. This includes Absorber,
Desorber, Lean-Rich Heat Exchanger and Reboiler. Key assuptions are described 
here,
ABSORBER: 
"""
import numpy as np
from casadi import *
from ..Models import Stream as Stream
from ..Properties import Properties
from ..Models import Column as Column
# import Column as Desorber
from ..Models import HeatExchanger as HeatExchanger
from ..Models import Reboiler as Reboiler
from copy import deepcopy
# from operating_points import opt_pts


class PCCPlant:
    """
    The following constants are declared:
        nComponents = 4 : Number of components in the system
        R = 8.314 : Universal gas constant
        DHRXN = 82000 : Heat of reaction
        aP 
        
    """
    # Constants or general parameters
    nComponents = 4
    R = 8.314
    g = 8.80665
    DHRXN = 82000
    aP = 143.9
    dP = 0.038
    
    # xss = np.load('xss.npy')
    # zss = np.load('xss.npy')
    # uss = np.load('uss.npy') # 0.581191*0.94
    # yss = np.load('yss.npy')

    def __init__(self):
        # Streams
        self.fluegas = Stream.Stream(47, 319.7, 103825, np.array([0.04129, 0.00223, 0, 0.00112]))
        # self.fluegas = Stream.Stream(0.0898912, 319.7, 103825, np.array([0.03571, 0.00781, 0, 0.00112]))
        self.leanAmine = Stream.Stream(6.292384000000001E-4, 314, 101325, np.array([0.0, 0.865, 5, 39.09])) #0.173 mol CO2/mol amine, based on Luo and Wang delta between rich and lean amine
        # self.leanAmine = Stream.Stream(6.292384000000001E-4, 314, 101325, np.array([0.0, 1.36, 5, 39.09]))

        # Absorber specs
        self.absHeight = 12.5
        # self.absHeight = 6.1
        self.absNStages = 5
        self.absDiameter = 4.2
        # self.absDiameter = 0.43
        self.absNStates = self.absNStages * 2 * (self.nComponents + 1)
        self.absorber = Column.Column(self.absHeight, self.absDiameter, 'absorber')
        self.absorber.gasin = deepcopy(self.fluegas)

        # Desorber specs
        self.desHeight = 12.5
        # self.desHeight = 6.1
        self.desNStages = 5
        self.desDiameter = 4.9
        # self.desDiameter = 0.43
        self.desNStates = self.desNStages * 2 * (self.nComponents + 1)
        self.desorber = Column.Column(self.desHeight, self.desDiameter, 'desorber')

        # Heat exchanger specs
        self.hrxNStates = 2
        self.hrx = HeatExchanger.HeatExchanger()

        # Reboiler specs
        self.rebNStates = 1
        self.reboiler = Reboiler.Reboiler()
        
        #Heat capacity of amine
        self.CP_amine = 3.9 #kJ/kg C
        self.CP_seawater = 4.18 #kJ/kg C
        self.T_amine_desorber = 100 + 273 #K
        self.sw_inlet = 50 + 273 #K
        self.sw_outlet = 35 + 273 #K
        
        #Heat capacity of amine
        self.engine_cap = 10800 * 2 #kW, times 2 because there are 2 main engines as per Luo and Wang 2017 paper
        self.SFOC = 177.5 #g/kWh
        self.air_intake = 6.27 #kg/kWh

    # ODEs
    def getODE(self, x, z, u, p):
        # Get and Split the states into various equipment
        abs_x = x[0:self.absNStates]
        des_x = x[self.absNStates:self.absNStates + self.desNStates]
        hrx_x = x[self.absNStates + self.desNStates:self.absNStates + self.desNStates + self.hrxNStates]
        reb_x = x[self.absNStates + self.desNStates + self.hrxNStates:self.absNStates + self.desNStates + self.hrxNStates + self.rebNStates]

        # Get and split inputs
        abs_u = u[0] / 1000  # Lean amine flow volumetric flow
        fuel_u = u[1]
        steam_mass_flow = fuel_u * 42200/2762.83 #Input u is fuel flow rate of boiler in kg/h * calorific value / spec enthalpy at 6 barg and 165 C (https://www.tlv.com/global/SG/calculator/steam-table-pressure.html) 
        steam_heat = steam_mass_flow*(2762.83-697.5)/3600/1000 #2762.83 is spec enthalpy of steam, 697.5 is specific enthalpy of water at 7 bar. Assuming 16.26 kg/h as input fuel, the energy from steam is 0.1425 MW

        # Get and split disturbances
        abs_d = p[0]  # engine load ratio (between 0.15 to 1)
        #abs_d = 1.0
        
        #Flue gas flow rate components
        CO2_flow = (self.SFOC * self.engine_cap * abs_d / 1000000) * 86.66 / 100 * 44 / 12 # %C = 86.6, MW C = 12, MW CO2 = 44, unit is ton/h
        H2O_flow = (self.SFOC * self.engine_cap * abs_d / 1000000) * 12.29 / 100 * 18 / 2 # %H2 = 12.29, MW H2 = 2, MW H2O = 18, unit is ton/h
        N2_flow = self.air_intake * self.engine_cap * abs_d * 92.5 / 100 / 1000 # %N2 = 92.5, MW N = 12, MW CO2 = 44, unit is ton/h
        total_flue_gas = (CO2_flow + H2O_flow + N2_flow) * 1000 / 0.7 /3600 #convert unit from ton/h to m3/s, density of flue gas average is 0.7 kg/m3
        
        self.absorber.gasin.Q = total_flue_gas # unit is m3/s
        # self.absorber.gasin.Q = 0.0898912 * abs_d
        
        # Waste heat calculation
        waste_heat = self.absorber.gasin.Q * 1.12 * (360-150) * 0.748 / 1000 #( 1.12 is cp of flue gas, 0.748 is density of flue gas, 360 is Tin, and 150 is Tout. https://www.pipeflowcalculations.com/tables/flue-gas.xhtml)
        # Total heat
        reb_u = (steam_heat + waste_heat) * 1000 # Reboiler heat input
        # reb_u = u[1] * 1000  # Reboiler heat input
        
        # Absorption column
        self.absorber.liqin.Q = abs_u
        self.absorber.liqin.P = 101325.0  # Pressure is reduced
        # self.absorber.liqin.T = 314.0  # Lean amine is cooled
        T_MEA_in = self.T_amine_desorber + (u[2] * self.CP_seawater * (self.sw_outlet-self.sw_inlet)/(u[0]*self.CP_amine))


        self.absorber.liqin.T = T_MEA_in  # Lean amine is cooled using u[2] = 19.568, the value of T = 314K or 41 C. Also change the value in Column.py
        self.absorber.liqin.comp[0] = 1e-100  # Nitrogen concentration
        self.absorber.liqin.comp[1] = z[5]
        self.absorber.liqin.comp[2] = 5
        self.absorber.liqin.comp[3] = 39.09

        abs_xdot = self.absorber.getValue(abs_x, abs_u, abs_d)

        # Heat exchanger
        self.hrx.shellin = deepcopy(self.absorber.liqout)
        self.hrx.tubein = deepcopy(self.absorber.liqin)
        self.hrx.tubein.T = reb_x[0]
        hrx_xdot = self.hrx.getValue(hrx_x, 0, 0)

        # Reboiler
        self.desorber.liqin = deepcopy(self.hrx.shellout)
        self.desorber.liqin.P = 149750.0  # Pump to increase head
        self.desorber.gasin = deepcopy(self.fluegas)
        self.desorber.getValue(des_x, 0, 0)
        self.reboiler.liqin = deepcopy(self.desorber.liqout)
        reb_xdot = self.reboiler.getValueO(reb_x, z[0:5], reb_u, 0)

        # Desorber
        self.desorber.gasin = deepcopy(self.reboiler.gasout)
        self.desorber.gasin.Q = z[6]
        des_xdot = self.desorber.getValue(des_x, 0.0, 0.0)

        # self.desorber.gasin.P = 170000
        # self.desorber.gasin.Q = z[13]
        # self.desorber.gasin.comp = z[15:19]
        # self.desorber.gasin.T = reb_x[0]

        return vertcat(abs_xdot, des_xdot, hrx_xdot, reb_xdot)

    def getALG(self, x, z):
        # abs_x = x[0:self.absNStates]
        # des_x = x[self.absNStates:self.absNStates+self.desNStates]
        # hrx_x = x[self.absNStates+self.desNStates:self.absNStates+self.desNStates+self.hrxNStates]
        reb_x = x[
                self.absNStates + self.desNStates + self.hrxNStates:self.absNStates + self.desNStates + self.hrxNStates + self.rebNStates]

        res0 = self.reboiler.getValueA(reb_x, z[0:5], 0.0, 0.0)
        # res1 = z[13] - self.reboiler.gasout.Q
        res1 = z[5] - self.reboiler.liqout.comp[1]
        res2 = z[6] - self.reboiler.gasout.Q
        # res3 = z[15:19] - self.reboiler.gasout.comp
        return vertcat(res0, res1, res2)
