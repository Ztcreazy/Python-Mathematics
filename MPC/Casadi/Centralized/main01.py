import sympy as sp

# Define symbolic variables
x1, y1, v1, beta1, M1, L1, L2, alpha1, theta1 = sp.symbols('x1 y1 v1 beta1 M1 L1 L2 alpha1 theta1')
x2, y2, v2, beta2, M2, L3, L4, alpha2, theta2 = sp.symbols('x2 y2 v2 beta2 M2 L3 L4 alpha2 theta2')

# Define the symbolic functions
f1 = v1 * sp.cos(beta1) * (1 + M1/L1 * sp.tan(beta1) * sp.tan(alpha1)) * sp.cos(theta1)
f2 = v1 * sp.cos(beta1) * (1 + M1/L1 * sp.tan(beta1) * sp.tan(alpha1)) * sp.sin(theta1)
f3 = v1 * (sp.sin(beta1)/L2 - M1/(L1*L2) * sp.cos(beta1) * sp.tan(alpha1))
f4 = v1 * (sp.tan(alpha1)/L1 - sp.sin(beta1)/L2 + M1/(L1*L2) * sp.cos(beta1) * sp.tan(alpha1))

f5 = v2 * sp.cos(beta2) * (1 + M2/L3 * sp.tan(beta2) * sp.tan(alpha2)) * sp.cos(theta2)
f6 = v2 * sp.cos(beta2) * (1 + M2/L3 * sp.tan(beta2) * sp.tan(alpha2)) * sp.sin(theta2)
f7 = v2 * (sp.sin(beta2)/L4 - M2/(L3*L4) * sp.cos(beta2) * sp.tan(alpha2))
f8 = v2 * (sp.tan(alpha2)/L3 - sp.sin(beta2)/L4 + M2/(L3*L4) * sp.cos(beta2) * sp.tan(alpha2))

# Calculate the Jacobians
J1 = sp.Matrix([
    [sp.diff(f1, var) for var in [x1, y1, theta1, beta1, x2, y2, theta2, beta2]],
    [sp.diff(f2, var) for var in [x1, y1, theta1, beta1, x2, y2, theta2, beta2]],
    [sp.diff(f3, var) for var in [x1, y1, theta1, beta1, x2, y2, theta2, beta2]],
    [sp.diff(f4, var) for var in [x1, y1, theta1, beta1, x2, y2, theta2, beta2]],
    [sp.diff(f5, var) for var in [x1, y1, theta1, beta1, x2, y2, theta2, beta2]],
    [sp.diff(f6, var) for var in [x1, y1, theta1, beta1, x2, y2, theta2, beta2]],
    [sp.diff(f7, var) for var in [x1, y1, theta1, beta1, x2, y2, theta2, beta2]],
    [sp.diff(f8, var) for var in [x1, y1, theta1, beta1, x2, y2, theta2, beta2]],
])

J2 = sp.Matrix([
    [sp.diff(f1, var) for var in [alpha1, v1, alpha2, v2]],
    [sp.diff(f2, var) for var in [alpha1, v1, alpha2, v2]],
    [sp.diff(f3, var) for var in [alpha1, v1, alpha2, v2]],
    [sp.diff(f4, var) for var in [alpha1, v1, alpha2, v2]],
    [sp.diff(f5, var) for var in [alpha1, v1, alpha2, v2]],
    [sp.diff(f6, var) for var in [alpha1, v1, alpha2, v2]],
    [sp.diff(f7, var) for var in [alpha1, v1, alpha2, v2]],
    [sp.diff(f8, var) for var in [alpha1, v1, alpha2, v2]],
])

# sp.pprint(J1[0, 0])
# sp.pprint(J2)

import casadi as ca
import numpy as np
import matplotlib.pyplot as plt
# test01 trajectory optimization
# test02 feedback

T = 0.5 
N = int(36.0 / T) 

# x, y, theta (trailer), beta (truck)
x1 = ca.SX.sym('x1'); y1 = ca.SX.sym('y1')
theta1 = ca.SX.sym('theta1'); beta1 = ca.SX.sym('beta1')
states = ca.vertcat(x1, y1, theta1, beta1)
n_states = states.size()[0]

# alpha (truck), v (truck)
alpha1 = ca.SX.sym('alpha1'); v1 = ca.SX.sym('v1')
controls = ca.vertcat(alpha1, v1)
n_controls = controls.size()[0]

# xdot = f(x, u)

M1 = 2; L1 = 8; W1 = 4; L2 = 12; W2= 3

rhs = ca.vertcat(v1*ca.cos(beta1)*(1 + M1/L1*ca.tan(beta1)*ca.tan(alpha1))*ca.cos(theta1), 
                 v1*ca.cos(beta1)*(1 + M1/L1*ca.tan(beta1)*ca.tan(alpha1))*ca.sin(theta1),
                 v1*(ca.sin(beta1)/L2 - M1/(L1*L2)*ca.cos(beta1)*ca.tan(alpha1)),
                 v1*(ca.tan(alpha1)/L1 - ca.sin(beta1)/L2 + M1/(L1*L2)*ca.cos(beta1)*ca.tan(alpha1)))
f = ca.Function('f', [states, controls], [rhs])

# dynamics ---> optimization
# decision/optimization variable
U = ca.SX.sym('U', n_controls, N); X = ca.SX.sym('X', n_states, N+1)
P = ca.SX.sym('P', n_states + n_states)

# quadratic
# Q = np.eye(4, 4)*5
Q = np.diag([10, 10, 20, 20])
R = np.eye(2, 2)
# print("R: ", R)

obj = 0 
g = []
g.append(X[:, 0] - P[:n_states])
g.append(X[:, N] - P[n_states:])

for i in range(N-1):

    obj += ca.mtimes([(X[:, i] - P[n_states:]).T, Q, (X[:, i] - P[n_states:])]) + ca.mtimes([U[:, i].T, R, U[:, i]])
    obj += ca.mtimes([U[:, i].T, R, U[:, i]])

Q_terminal = Q * 10
obj += ca.mtimes([(X[:, N] - P[n_states:]).T, Q_terminal, (X[:, N] - P[n_states:])])

for i in range(N-1):

    x_next = f(X[:, i], U[:, i])*T + X[:, i]
    g.append(X[:, i+1] - x_next) # dynamics

# OCP
opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': ca.vertcat(*g)}

opts_setting = {'ipopt.max_iter': 2000, 'ipopt.print_level': 5, 'print_time': 0,
                'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

# inequality constraint ---> function
lbg = [0.0] * (N * n_states + n_states)
ubg = [0.0] * (N * n_states + n_states)

lbx = []
ubx = []

alpha1_max = np.pi/4; v1_max = 10

for _ in range(N):

    lbx.append(-alpha1_max) # control inputs 
    lbx.append(-v1_max)     # alpha, v
         
    ubx.append(alpha1_max)
    ubx.append(v1_max)

for _ in range(N+1):

    lbx.append(-np.inf) # states x, y, theta, beta
    lbx.append(-np.inf)
    lbx.append(-np.pi)
    lbx.append(-np.pi/3.6)

    ubx.append(np.inf)
    ubx.append(np.inf)
    ubx.append(np.pi)
    ubx.append(np.pi/3.6)

# !!
x0 = np.array([-55, 30, 0, 0]); xs = np.array([24, -20, np.pi/2, 0])

# Initial guess
c_p = np.concatenate((x0, xs)) 

def initial_guess(initial_p, target_p, u0, p):
    # Generate initial guess for decision variables

    # if y0 > -5 (truck above the obstacle), one way-point is used
    if initial_p[1] >= -5 and initial_p[1] <= 5:
        p1 = round(p / 2)
        p2 = p - p1 + 1
        xMiddle = 0
        yMiddle = 10
        thetaMiddle = np.pi / 4

        xGuess = np.concatenate((np.linspace(initial_p[0], xMiddle, p1),
                                 np.linspace(xMiddle, target_p[0], p2)))

        yGuess = np.concatenate((np.linspace(initial_p[1], yMiddle, p1),
                                 np.linspace(yMiddle, target_p[1], p2)))

        thetaGuess = np.concatenate((np.linspace(initial_p[2], thetaMiddle, p1),
                                     np.linspace(thetaMiddle, target_p[2], p2)))
    else:
        # if y0 < -10 (truck below the obstacle), two way-points are used
        p1 = round(p / 3)
        p2 = round(p / 3)
        p3 = p - p1 - p2 + 1
        x1 = 0
        y1 = 0
        theta1 = np.pi / 6
        x2 = 24
        y2 = 0
        theta2 = np.pi / 3

        xGuess = np.concatenate((np.linspace(initial_p[0], x1, p1),
                                 np.linspace(x1, x2, p2),
                                 np.linspace(x2, target_p[0], p3)))

        yGuess = np.concatenate((np.linspace(initial_p[1], y1, p1),
                                 np.linspace(y1, y2, p2),
                                 np.linspace(y2, target_p[1], p3)))

        thetaGuess = np.concatenate((np.linspace(initial_p[2], theta1, p1),
                                     np.linspace(theta1, theta2, p2),
                                     np.linspace(theta2, target_p[2], p3)))

    betaGuess = np.zeros(p + 1)

    # Combine the guesses into the state guess matrix
    stateGuess = np.vstack((xGuess, yGuess, thetaGuess, betaGuess))

    z0 = []
    for ct in range(p):
        z0.append(np.concatenate((stateGuess[:, ct], u0)))
    z0 = np.concatenate((np.array(z0).flatten(), stateGuess[:, -1]))

    XY0 = stateGuess[:2, :].T
    return z0, XY0

# u0 = np.zeros(2)
u0 = np.array([0.0, -1.0])
p = 72
z0, XY0 = initial_guess(x0, xs, u0, p)

init_control = np.concatenate((np.zeros((N, n_controls)).reshape(-1, 1),
                               np.zeros((N+1, n_states)).reshape(-1, 1)))

res = solver(x0=z0, p=c_p, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
estimated_opt = res['x'].full()

u_opt = estimated_opt[:N*n_controls].reshape(N, n_controls)
x_opt = estimated_opt[N*n_controls:].reshape(N+1, n_states)

t_c = np.linspace(0, T*N, N+1)

# trajectory
plt.figure()
plt.plot(x_opt[:, 0], x_opt[:, 1])
plt.grid(True)
plt.title('Trajectory')
plt.xlabel('x position')
plt.ylabel('y position')
plt.savefig('C:/Users/14404/OneDrive/Desktop/PythonMathematics/MPC/Casadi/trajectoryoptimization.png')
plt.show()

# states
plt.figure()

plt.subplot(2, 2, 1)
plt.plot(t_c, x_opt[:, 0]); plt.grid(True)
plt.ylabel('x position')

plt.subplot(2, 2, 2)
plt.plot(t_c, x_opt[:, 1]); plt.grid(True)
plt.ylabel('y position')

plt.subplot(2, 2, 3)
plt.plot(t_c, x_opt[:, 2]); plt.grid(True)
plt.ylabel('theta (trailer)')
plt.xlabel('Time (s)')

plt.subplot(2, 2, 4)
plt.plot(t_c, x_opt[:, 3]); plt.grid(True)
plt.ylabel('beta (truck)')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()

# control inputs
plt.figure()

t_c_u = np.linspace(0, T*(N-1), N)

plt.subplot(2, 1, 1)
plt.plot(t_c_u, u_opt[:, 0]); plt.grid(True)
plt.title('Control Inputs over Time')
plt.ylabel('Linear Velocity (v)')

plt.subplot(2, 1, 2)
plt.plot(t_c_u, u_opt[:, 1]); plt.grid(True)
plt.ylabel('Angular Velocity (omega)')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()

# animation

from Truck_trailer_plot import *

params = {
    'M': 2, 'W1': 4.0, 'L1': 8.0, 'W2': 3.0, 'L2': 12.0, 'Wwheel': 0.4, 'Lwheel': 1
}

fig, ax = plt.subplots()
ax.set_xlim(-60, 60)
ax.set_ylim(-50, 50)
ax.set_xlabel('x')
ax.set_ylabel('y')

animate_tractor_trailer(ax, t_c, x_opt, u_opt[:, 0], params)
