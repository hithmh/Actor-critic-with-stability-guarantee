# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 11:06:40 2017

@author: Benjamin
"""

from casadi import *

class Kvalue(Callback):
   def __init__(self, name, opts={}):
       # Create dummy symbolics to build functions
       x = SX.sym('x',4)   # Concentration
       T = SX.sym('T')
       P = SX.sym('P')
       
       K = SX.zeros(4)
       alpha = x[1]/(x[2]+1e-100)
        
       K[0] = 1e-100
       K[1] = np.exp(-1306 + (143.93*alpha) + (6.67*T) - (7.040*alpha*alpha) - (0.00852*T*T) - (0.3546*alpha*T))
       K[2] = np.exp(369 - (51.86*alpha) - (1.916*T) + (1.093*alpha*alpha) + (0.002479*T*T) + (0.1303*alpha*T))
       K[3] = np.exp(-37.58 + (1.584*alpha) + (0.1795*T) - (0.11589*alpha*alpha) - (0.000213*T*T) - (0.003869*alpha*T))
        
       func = K
       f = Function('Kvalue3', [T, P, x],[func])
       self.f = f
       
       self.construct(name, opts)
   
   # Number of inputs and outputs
   def get_n_in(self): return 3
   def get_n_out(self): return 1
   
   # Initialize the object
   def init(self):
       print("Initializing object")
       
   # Evaluate numerically
   def eval(self, arg):
       T = arg[0]
       P = arg[1]
       x = arg[2]
       
       f = self.f(T, P, x)
       return [f]
   
# Test
x1 = np.array([0,2.5,5,39.09])
T = 387
P = 170000

f = Kvalue('f')
res = f(T, P, x1)
print(res)
       
       
           