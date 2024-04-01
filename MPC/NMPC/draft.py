import numpy as np

# initial conditions
CA0     = 2.    
CB0     = 0.    
CC0     = 0.    
Vol0    = 0.5   
CBin0   = 10.
x0      = np.array([CA0,CB0,CC0,Vol0,CBin0]) # initial state of plant

# initial state estimates
CAhat0      = 1.5    
CBhat0      = 0.    
CChat0      = 0.    
Volhat0     = 0.2   
CBinhat0    = 2.
xhat0       = np.array([CAhat0,CBhat0,CChat0,Volhat0,CBinhat0])    # initial state estimate

Sigmaxhat0 = 0.01*np.diag([CAhat0,0.1,0.1,Volhat0,CBinhat0*100.]) # initial state covariance estimate

print("initial state covariance estimate: ", Sigmaxhat0)