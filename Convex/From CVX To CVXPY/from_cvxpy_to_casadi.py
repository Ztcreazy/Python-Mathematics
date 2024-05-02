import cvxpy as cp
import numpy as np

np.random.seed(1)
m = 30; n = 10

# A1 = np.random.randn(m, n); A2 = np.random.randn(m, n)
# b1 = np.random.rand(m, 1); b2 = np.random.rand(m, 1)
# c1 = np.random.randn(n, 1); c2 = np.random.randn(n, 1)
# D1 = np.random.randn(2, n); D2 = np.random.randn(2, n)

# MATLAB ----> Python
from scipy.io import loadmat
data_A1 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/A1.mat') 
data_A2 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/A2.mat')
data_b1 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/b1.mat')
data_b2 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/b2.mat')
data_c1 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/c1.mat')
data_c2 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/c2.mat')
data_D1 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/D1.mat')
data_D2 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/D2.mat')

A1 = data_A1['A1']; A2 = data_A2['A2']; 
b1 = data_b1['b1']; b2 = data_b2['b2']; 
c1 = data_c1['c1']; c2 = data_c2['c2']; 
D1 = data_D1['D1']; D2 = data_D2['D2']; 

x1 = cp.Variable(n); x2 = cp.Variable(n)

objective = cp.Minimize(cp.sum(c1.T @ x1) + cp.sum(c2.T @ x2) + 0.1 * cp.sum_squares(x1) + 0.1 * cp.sum_squares(x2))
constraints = [A1 @ x1 <= b1.flatten(), A2 @ x2 <= b2.flatten(), D1 @ x1 + D2 @ x2 <= 0]

problem = cp.Problem(objective, constraints)
problem.solve()

print("The optimal value is", problem.value) # The optimal value is -2.7137379897903235
# print("A solution x1 is")
# print(x1.value)
# print("A solution x2 is")
# print(x2.value)
print("A dual solution corresponding to the inequality constraints is")
print(problem.constraints[2].dual_value)
# A dual solution corresponding to the inequality constraints is [0.         0.72291799]



# from cvxpy to casadi
from casadi import *

n = 10
y1 = SX.sym('y1', n)
y2 = SX.sym('y2', n)

obj = dot(c1, y1) + dot(c2, y2) + 0.1 * dot(y1, y1) + 0.1 * dot(y2, y2)

constr1_ca = A1 @ y1 - b1; constr2_ca = A2 @ y2 - b2
constr_dual = D1 @ y1 + D2 @ y2
constr = vertcat(constr1_ca, constr2_ca, constr_dual)

opt_prob = {'x': vertcat(y1, y2), 'f': obj, 'g': constr}
solver = nlpsol('solver', 'ipopt', opt_prob)



import time
start_time = time.time()

num_constr = constr.shape[0]
sol = solver(lbg=-inf, ubg=0, lbx=-inf*(2*n), ubx=inf*(2*n))

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")

y_opt = sol['x'].full()
opt_val = sol['f'].full()

y1_opt = y_opt[:n]
y2_opt = y_opt[n:]

# print('Optimal solution for y1:')
# print(y1_opt)
# print('Optimal solution for y2:')
# print(y2_opt)
print('Optimal value from CasADi:') # Optimal value from CasADi: [[-2.71373805]]
print(opt_val)

lambda_opt = sol['lam_g'].full()
print('Lambda coupled from CasADi:') # Lambda coupled from CasADi:[[1.80287834e-09] 
                                                                 # [7.22917983e-01]]
print(lambda_opt[-2:])
