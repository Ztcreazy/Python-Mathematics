# primal decomposition
import numpy as np
import cvxpy as cp

np.random.seed(1)
m = 30; n = 10

saved_optval = -2.713737968086698

# A1 = np.random.randn(m, n); A2 = np.random.randn(m, n)
# b1 = np.random.rand(m, 1); b2 = np.random.rand(m, 1)
# c1 = np.random.randn(n, 1); c2 = np.random.randn(n, 1)
# D1 = np.random.randn(2, n); D2 = np.random.randn(2, n)

from scipy.io import loadmat
data_A1 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/A1.mat') 
data_A2 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/A2.mat')
data_b1 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/b1.mat')
data_b2 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/b2.mat')
data_c1 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/c1.mat')
data_c2 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/c2.mat')
data_D1 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/D1.mat')
data_D2 = loadmat('C:/Users/14404/OneDrive/Desktop/Optimization/Decompostion/D2.mat')

A1 = data_A1['A1']; A2 = data_A2['A2']; b1 = data_b1['b1']; b2 = data_b2['b2']; 
c1 = data_c1['c1']; c2 = data_c2['c2']; D1 = data_D1['D1']; D2 = data_D2['D2']; 

niters = 120
ts_cvx_p = np.zeros((2, niters))
t = np.zeros((2, 1))

Primal = True; Dual = False
ADMM = False; All = False

import matplotlib.pyplot as plt

niters = 119 # 120-1
ts_cvx_p = np.zeros((2, niters))
t = np.zeros((2, 1))
f_cvx_p = np.zeros(niters)

import time
start_time = time.time()

if Primal or All:
    for i in range(niters):
        x1_cvx_p = cp.Variable(n)
        # l1_cvx_p = cp.Variable()
        prob1 = cp.Problem(cp.Minimize(cp.sum(c1.T @ x1_cvx_p) + 0.1 * cp.sum_squares(x1_cvx_p)),
                        [A1 @ x1_cvx_p <= b1.flatten(), D1 @ x1_cvx_p <= t.flatten()])
        prob1.solve()
        f1_cvx_p = prob1.value
        l1_cvx_p = prob1.constraints[1].dual_value # print("l1_cvx_p: ", l1_cvx_p)
        l1_cvx_p = np.array(l1_cvx_p)
        l1_cvx_p_reshaped = l1_cvx_p.reshape(2, 1)
        # print("l1_cvx_p_reshaped: ", l1_cvx_p_reshaped)
        


        x2_cvx_p = cp.Variable(n)
        # l2_cvx_p = cp.Variable()
        prob2 = cp.Problem(cp.Minimize(cp.sum(c2.T @ x2_cvx_p) + 0.1 * cp.sum_squares(x2_cvx_p)),
                        [A2 @ x2_cvx_p <= b2.flatten(), D2 @ x2_cvx_p <= -t.flatten()])
        prob2.solve()
        f2_cvx_p = prob2.value
        l2_cvx_p = prob2.constraints[1].dual_value # print("l2_cvx_p: ", l2_cvx_p)
        l2_cvx_p = np.array(l2_cvx_p)
        l2_cvx_p_reshaped = l2_cvx_p.reshape(2, 1)
        # print("l2_cvx_p_reshaped: ", l2_cvx_p_reshaped)


        alpha_cvx_p = 0.1
        t = t - alpha_cvx_p * (l2_cvx_p_reshaped - l1_cvx_p_reshaped)

        f_cvx_p_i = f1_cvx_p + f2_cvx_p
        print("f1_cvx_p + f2_cvx_p: ", f_cvx_p_i)
        f_cvx_p[i] = f_cvx_p_i
        ts_cvx_p[:, i] = t.flatten()

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")

    plt.figure(1)
    plt.semilogy(f_cvx_p - saved_optval, linewidth=1.5)
    plt.xlabel('k')
    plt.ylabel('f_cvx_p - fmin')
    plt.grid(True)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    plt.figure(2)
    plt.plot(ts_cvx_p[0,:], 'b', linewidth=1.5)
    plt.plot(ts_cvx_p[1,:], 'r', linewidth=1.5)
    plt.xlabel('k')
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()



if Primal or All:

    import casadi as ca

    alpha_ca_p = 0.1
    t = np.zeros((2, 1))
    ts_ca_p = np.zeros((2, niters))
    f_ca_p = np.zeros(niters)

    for i in range(niters):

        p1 = ca.SX.sym('p1', n)
        obj1_ca_p = c1.T @ p1 + 0.1 * ca.dot(p1, p1)
        constr1_ca = A1 @ p1 - b1.flatten()
        constr_dual1_ca_p = D1 @ p1 - t.flatten()

        opt_prob1 = {'x': p1, 'f': obj1_ca_p, 'g': ca.vertcat(constr1_ca, constr_dual1_ca_p)}
        
        solver1 = ca.nlpsol('solver', 'ipopt', opt_prob1)
        sol1 = solver1(lbg=-np.inf, ubg=0, lbx=-np.inf, ubx=np.inf)

        p2 = ca.SX.sym('p2', n)
        obj2_ca_p = c2.T @ p2 + 0.1 * ca.dot(p2, p2)
        constr2_ca = A2 @ p2 - b2.flatten()
        constr_dual2_ca_p = D2 @ p2 + t.flatten()

        opt_val1_ca_p = sol1['f'].full().item()
        lambda_opt1_ca_p = sol1['lam_g'].full().flatten()

        lambda1_ca_p = lambda_opt1_ca_p[-2:]
        lambda1_ca_p = np.array(lambda1_ca_p)
        lambda1_ca_p_reshaped = lambda1_ca_p.reshape(2, 1)
        print("lambda1_ca_p_reshaped: ", lambda1_ca_p_reshaped)

        opt_prob2 = {'x': p2, 'f': obj2_ca_p, 'g': ca.vertcat(constr2_ca, constr_dual2_ca_p)}
        
        solver2 = ca.nlpsol('solver', 'ipopt', opt_prob1)
        sol2 = solver2(lbg=-np.inf, ubg=0, lbx=-np.inf, ubx=np.inf)

        opt_val2_ca_p = sol2['f'].full().item()
        lambda_opt2_ca_p = sol2['lam_g'].full().flatten()

        lambda2_ca_p = lambda_opt2_ca_p[-2:]
        lambda2_ca_p = np.array(lambda2_ca_p)
        lambda2_ca_p_reshaped = lambda2_ca_p.reshape(2, 1)
        print("lambda2_ca_p_reshaped: ", lambda2_ca_p_reshaped)

        t = t - alpha_ca_p * (lambda2_ca_p_reshaped.astype(float) - lambda1_ca_p_reshaped.astype(float))
        print("t: ", t)

        f_ca_p[i] = opt_val1_ca_p + opt_val2_ca_p
        print("f_ca_p: ", opt_val1_ca_p + opt_val2_ca_p)
        ts_ca_p[:, i] = t.flatten()

    plt.figure(3)
    plt.semilogy(f_ca_p - saved_optval, linewidth=1.5)
    plt.xlabel('k')
    plt.ylabel('f_ca_p - fmin')
    plt.grid(True)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()

    plt.figure(4)
    plt.plot(ts_ca_p[0], 'b', linewidth=1.5, label='t1')
    plt.plot(ts_ca_p[1], 'r', linewidth=1.5, label='t2')
    plt.xlabel('k')
    plt.grid(True)
    plt.legend()
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()