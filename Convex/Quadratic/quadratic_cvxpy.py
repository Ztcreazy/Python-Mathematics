"""
A quadratic program is an optimization problem with a quadratic objective 
and affine equality and inequality constraints.

When we solve a quadratic program, in addition to a solution x*, we obtain a dual solution 
lambda* corresponding to the inequality constraints. A positive entry lambda_i* indicates that 
the constraint g_i.T *x < h_i holds with equality for x* and suggests that 
changing h_i would change the optimal value.
"""
import cvxpy as cp
import numpy as np

m = 15
n = 10
p = 5

np.random.seed(1)
P = np.random.randn(n, n)
P = P.T @P

q = np.random.randn(n)

G = np.random.randn(m, n)
h = G @np.random.randn(n)

A = np.random.randn(p, n)
b = np.random.randn(p)

# Define and solve the CVXPY problem.
x = cp.Variable(n)
prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                 [G @ x <= h,
                  A @ x == b])
prob.solve()

# Print result.
print("The optimal value is", prob.value)
print("A solution x is")
print(x.value)
print("A dual solution corresponding to the inequality constraints is")
print(prob.constraints[0].dual_value)
