# Method of multipliers
# minimize      f(x)
# subject to    A *x = b
# where f is convex, x ----> R^n is the optimization variable
# and A ----> R^(m*n) and b ----> R^m are problem data
"""
the augmented Lagrangian
L_rho(x, y) = f(x) + y.T(A *x - b) + (rho/2) *||A *x - b||_2 ^2
The dual function associated with the augmented Lagrangian is:
g_rho(y) = inf_x L_rho(x, y)
We maximize the dual function using gradient ascent. Each step of gradient ascent 
reduces to the x and y updates
x^(k+1) := argmin_x ( f(x) + (y^k).T *(A *x - b) + (rho/2) *||A *x - b||_2 ^2 )
y^(k+1) := y ^k + rho *( A *x^(k+1) - b )
"""
import cvxpy as cp
import numpy as np
np.random.seed(1)

# Initialize data.
MAX_ITERS = 10
rho = 1.0
n = 20
m = 10
A = np.random.randn(m,n)
b = np.random.randn(m)

# Initialize problem.
x = cp.Variable( shape=n )
f = cp.norm(x, 1)

# Solve with CVXPY.
cp.Problem(cp.Minimize(f), [A @ x == b]).solve( solver=cp.ECOS )
print("Optimal value from CVXPY: {}".format(f.value))

# Solve with method of multipliers.
resid = A@x - b



# !!!! L_rho(x, y) = f(x) + y.T(A *x - b) + (rho/2) *||A *x - b||_2 ^2
# cp.Parameter 
y = cp.Parameter( shape=(m) ); y.value = np.zeros(m)
aug_lagr = f + y.T @ resid + (rho/2)*cp.sum_squares(resid)

for t in range(MAX_ITERS):
    cp.Problem(cp.Minimize(aug_lagr)).solve( solver=cp.ECOS ) # "solver=cp.ECOS"
    y.value += rho*resid.value

print("Optimal value from method of multipliers: {}".format(f.value))
