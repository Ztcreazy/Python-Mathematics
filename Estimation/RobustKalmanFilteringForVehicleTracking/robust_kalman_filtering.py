"""
We will try to pinpoint the location of a moving vehicle with high accuracy 
from noisy sensor data. We’ll do this by modeling the vehicle state 
as a discrete-time linear dynamical system. Standard Kalman filtering can be used 
to approach this problem when the sensor noise is assumed to be Gaussian. We’ll use 
robust Kalman filtering to get a more accurate estimate of the vehicle state 
for a non-Gaussian case with outliers.
"""
# https://www.cvxpy.org/examples/applications/robust_kalman.html
"""
Problem statement
x_(t+1) = Ax_t + Bw_t <---- input
y_t     = Cx_t + v_t
A Kalman filter estimates x_t by solving the optimization problem
minimize sum from t = 0 to N-1 
( ||w_t||_2 ^2 + tau *||v_t||_2 ^2 )

subject to x_(t+1) = Ax_t + B*w_t, t = 0, 1, ..., N-1
           y_t     = Cx_t + v_t,   t = 0, 1, ..., N-1 

where tau is a tuning parameter

This model performs well when w_t and v_t are Gaussian. However, the quadratic objective 
can be influenced by large outliers, which degrades the accuracy of the recovery. 
To improve estimation in the presence of outliers, we can use robust Kalman filtering.

objective minimize ||w_t||_ ^2 + tau *phi_rho(vt)
where phi_rho is the Huber function
phi_rho(a) = ||a||_ ^2                ||a||_2 <= rho 
           = 2 *rho *||a||_2 - rho^2, ||a||_2 > rho  

The Huber penalty function penalizes estimation error linearly outside of a ball of radius rho, 
whereas in standard Kalman filtering, all errors are penalized quadratically. 
Thus, large errors are penalized less harshly, making this model more robust to outliers           

"""

# Problem Data
# We generate the data for the vehicle tracking problem. We’ll have N=1000, 
# w_t a standard Gaussian, and v_t a standard Gaussian, 
# except 20% of the points will be outliers with sigma = 20.

import numpy as np
from plot import *

n = 1000 # number of timesteps
T = 50 # time will vary from 0 to T with step delt
ts, delt = np.linspace(0,T,n,endpoint=True, retstep=True)
gamma = .05 # damping, 0 is no damping

A = np.zeros((4,4))
B = np.zeros((4,2))
C = np.zeros((2,4))

A[0,0] = 1
A[1,1] = 1
A[0,2] = (1-gamma*delt/2)*delt
A[1,3] = (1-gamma*delt/2)*delt
A[2,2] = 1 - gamma*delt
A[3,3] = 1 - gamma*delt

B[0,0] = delt**2/2
B[1,1] = delt**2/2
B[2,0] = delt
B[3,1] = delt

C[0,0] = 1
C[1,1] = 1

sigma = 20
p = .20
np.random.seed(6)

x = np.zeros((4,n+1))
x[:,0] = [0,0,0,0]
y = np.zeros((2,n))

# generate random input and noise vectors
w = np.random.randn(2,n)
v = np.random.randn(2,n)

# add outliers to v
np.random.seed(0)
inds = np.random.rand(n) <= p
# print("inds: ", inds)
v[:,inds] = sigma*np.random.randn(2,n)[:,inds]
# print("v[:,inds]:", v)

# simulate the system forward in time
for t in range(n):
    y[:,t] = C.dot(x[:,t]) + v[:,t]
    x[:,t+1] = A.dot(x[:,t]) + B.dot(w[:,t])

x_true = x.copy()
w_true = w.copy()

plot_state(ts,(x_true,w_true))
plot_positions([x_true,y], ['True', 'Observed'],[-4,14,-5,20],'rkf1.pdf')

"""%%time"""
import time
import cvxpy as cp

start_time = time.time()
x = cp.Variable(shape=(4, n+1))
w = cp.Variable(shape=(2, n))
v = cp.Variable(shape=(2, n))

# the standard Kalman filtering
tau = .08

obj = cp.sum_squares(w) + tau*cp.sum_squares(v) # * Use * for matrix-scalar and vector-scalar multiplication
obj = cp.Minimize(obj)

constr = []
for t in range(n):
    constr += [ x[:,t+1] == A@x[:,t] + B@w[:,t] , # Use @ for matrix-matrix and matrix-vector multiplication
                y[:,t]   == C@x[:,t] + v[:,t]   ]
    
cp.Problem(obj, constr).solve(verbose=True)

end_time = time.time()
print("elapsed time: ", end_time - start_time)

x = np.array(x.value)
w = np.array(w.value)

plot_state(ts,(x_true,w_true),(x,w))
plot_positions([x_true,y], ['True', 'Noisy'], [-4,14,-5,20])
plot_positions([x_true,x], ['True', 'KF recovery'], [-4,14,-5,20], 'rkf2.pdf')

print("optimal objective value: {}".format(obj.value))



# robust Kalman filtering
start_time2 = time.time()

x2 = cp.Variable(shape=(4, n+1))
w2 = cp.Variable(shape=(2, n))
v2 = cp.Variable(shape=(2, n))

tau2 = 2
rho = 2

obj2 = cp.sum_squares(w2)
obj2 += cp.sum([tau2*cp.huber(cp.norm(v2[:,t]),rho) for t in range(n)])
obj2 = cp.Minimize(obj2)

constr2 = []
for t in range(n):
    constr2 += [ x2[:,t+1] == A@x2[:,t] + B@w2[:,t] ,
                y[:,t]   == C@x2[:,t] + v2[:,t]   ]

cp.Problem(obj2, constr2).solve(verbose=True)

end_time2 = time.time()
print("elapsed time: ", end_time2 - start_time2)

x2 = np.array(x2.value)
w2 = np.array(w2.value)

plot_state(ts,(x_true,w_true),(x2,w2))
plot_positions([x_true,y], ['True', 'Noisy'], [-4,14,-5,20])
plot_positions([x_true,x2], ['True', 'Robust KF recovery'], [-4,14,-5,20],'rkf3.pdf')

print("optimal objective value: {}".format(obj2.value))
