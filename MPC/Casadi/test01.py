import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# test01 trajectory optimization
# test02 feedback

T = 0.2 
N = int(20.0 / T) 
v_max = 0.6
omega_max = np.pi/4.0

# x
x = ca.SX.sym('x'); y = ca.SX.sym('y'); theta = ca.SX.sym('theta')
states = ca.vertcat(x, y, theta)
n_states = states.size()[0]

# u
v = ca.SX.sym('v'); omega = ca.SX.sym('omega')
controls = ca.vertcat(v, omega)
n_controls = controls.size()[0]

# xdot = f(x, u)
rhs = ca.vertcat(v*ca.cos(theta), v*ca.sin(theta), omega)
f = ca.Function('f', [states, controls], [rhs])

# dynamics ---> optimization
# decision/optimization variable
U = ca.SX.sym('U', n_controls, N); X = ca.SX.sym('X', n_states, N+1)
P = ca.SX.sym('P', n_states + n_states)

# quadratic
Q = np.array([[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, .1]])
R = np.array([[0.5, 0.0], [0.0, 0.05]])

obj = 0 
g = []
g.append(X[:, 0] - P[:n_states])

for i in range(N):
    obj += ca.mtimes([(X[:, i] - P[n_states:]).T, Q, (X[:, i] - P[n_states:])]) + ca.mtimes([U[:, i].T, R, U[:, i]])
    x_next = f(X[:, i], U[:, i])*T + X[:, i]
    g.append(X[:, i+1] - x_next) # dynamics

# OCP
opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': ca.vertcat(*g)}

opts_setting = {'ipopt.max_iter': 1500, 'ipopt.print_level': 5, 'print_time': 0,
                'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}
solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

# inequality constraint ---> function
lbg = [0.0] * (N * n_states + n_states)
ubg = [0.0] * (N * n_states + n_states)

lbx = []
ubx = []

for _ in range(N):
    lbx.append(-v_max)
    lbx.append(-omega_max)
    ubx.append(v_max)
    ubx.append(omega_max)

for _ in range(N+1):
    lbx.append(-2.0)
    lbx.append(-2.0)
    lbx.append(-np.inf)
    ubx.append(2.5)
    ubx.append(2.1)
    ubx.append(np.inf)

# !!
x0 = np.array([0.0, 0.0, 0.0]); xs = np.array([2.5, 2.1, 0.0])

# Initial guess
c_p = np.concatenate((x0, xs)) 
init_control = np.concatenate((np.zeros((N, n_controls)).reshape(-1, 1),
                               np.zeros((N+1, n_states)).reshape(-1, 1)))

res = solver(x0=init_control, p=c_p, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
estimated_opt = res['x'].full()

u_opt = estimated_opt[:N*n_controls].reshape(N, n_controls)
x_opt = estimated_opt[N*n_controls:].reshape(N+1, n_states)

t_c = np.linspace(0, T*N, N+1)

# states
plt.figure()

plt.subplot(3, 1, 1)
plt.plot(t_c, x_opt[:, 0]); plt.grid(True)
plt.title('States over Time')
plt.ylabel('x position')

plt.subplot(3, 1, 2)
plt.plot(t_c, x_opt[:, 1]); plt.grid(True)
plt.ylabel('y position')

plt.subplot(3, 1, 3)
plt.plot(t_c, x_opt[:, 2]); plt.grid(True)
plt.ylabel('theta (orientation)')
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

# trajectory
plt.figure()
plt.plot(x_opt[:, 0], x_opt[:, 1])
plt.grid(True)
plt.title('Trajectory')
plt.xlabel('x position')
plt.ylabel('y position')
plt.savefig('C:/Users/14404/OneDrive/Desktop/PythonMathematics/MPC/Casadi/trajectoryoptimization.png')
plt.show()