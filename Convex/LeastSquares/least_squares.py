# least squares
# linear regression
# measurements A---->R^(m*n) b---->R^(m)
# x---->R^(n)
# Ax ----> b
# ||Ax - b||_2 ^(2)
import cvxpy as cp
import numpy as np

m = 20
n = 15
np.random.seed(1)
A = np.random.randn(m, n)
b = np.random.randn(m)
# print("A: ", A, "b: ", b)

x = cp.Variable(n)
cost = cp.sum_squares(A @ x - b)
prob = cp.Problem(cp.Minimize(cost))
prob.solve()

print("The optimal value is", prob.value)
print("The optimal x is")
print(x.value)
print("The norm of the residual is ", cp.norm(A @ x - b, p=2).value)