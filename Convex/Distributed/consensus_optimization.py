"""
Suppose we have a convex optimization problem with N terms in the objective
minimize  sum from i=1 to N: f_i(x)

For example, we might be fitting a model to data and f_i is the loss function 
for the ith block of training data.

We can convert this problem into consensus form
minimize  sum from i=1 to N: f_i(x)
subject to                   x_i = z

We interpret the x_i as local variables, since they are particular to a given f_i. 
The variable z, by contrast, is global. The constraints x_i = z enforce consistency, 
or consensus.

We can solve a problem in consensus form using the Alternating Direction Method of Multipliers 
(ADMM). 
Each iteration of ADMM reduces to the following updates:
x_i ^(k+1) = argmin_x ( f_i(x_i) + (rho/2) *||x_i - xbar^k + u_i ^k||_2 ^2 )
u_i ^(k+1) = u_i ^k + x_i ^(k+1) - xbar^(k+1)

where xbar^k = (1/N) sum from i=1 to N: x_i ^k
The following code carries out consensus ADMM, using CVXPY to solve the local subproblems.
We split the x_i variables across N different worker processes. 
The workers update the x_i in parallel. A master process then gathers 
and averages the x_i and broadcasts xbar back to the workers. 
The workers update u_i locally.

"""
from cvxpy import *
import numpy as np
from multiprocessing import Process, Pipe

# Number of terms f_i.
N = ...
# A list of all the f_i.
f_list = ...

def run_worker(f, pipe):
    
    xbar = Parameter(n, value=np.zeros(n))
    u = Parameter(n, value=np.zeros(n))
    f += (rho/2)*sum_squares(x - xbar + u)

    prox = Problem( Minimize(f) )

    # ADMM loop.
    while True:
        prox.solve()
        pipe.send(x.value)
        xbar.value = pipe.recv()
        u.value += x.value - xbar.value

# Setup the workers.
pipes = []
procs = []
for i in range(N):
    local, remote = Pipe()
    pipes += [local]
    procs += [Process(target=run_process, args=(f_list[i], remote))]
    procs[-1].start()

# ADMM loop.
for i in range(MAX_ITER):
    # Gather and average xi
    xbar = sum(pipe.recv() for pipe in pipes)/N
    # Scatter xbar
    for pipe in pipes:
        pipe.send(xbar)

[p.terminate() for p in procs]        
