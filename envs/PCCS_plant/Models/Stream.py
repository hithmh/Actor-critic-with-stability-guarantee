# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 14:25:22 2017

@author: decardin
"""
from casadi import *
import numpy as np
from copy import deepcopy

class Stream:
    
    
    def __init__(self, Q = None, T = None, P = None, comp = None):
        if Q != None and T != None and P != None:   
            self.Q = Q
            self.T = T
            self.P = P
            self.comp = deepcopy(comp)
        else:
            self.Q = 1.0
            self.T = 298.15
            self.P = 101325.0
            self.comp = SX.zeros(4)
                