import casadi as ca
import numpy as np
import time
import matplotlib.pyplot as plt

# test02 feedback (shift_movement)
# test01 trajectory optimization

def shift_movement(T, t0, x0, u, x_f, f):
    f_value = f(x0, u[0, :])
    st = x0 + T*f_value.full()
    t = t0 + T
    # print(u[:,0])
    # u_end = np.concatenate((u[:, 1:], u[:, -1:]), axis=1)
    u_end = np.concatenate((u[1:], u[-1:]))
    # x_f = np.concatenate((x_f[:, 1:], x_f[:, -1:]), axis=1)
    x_f = np.concatenate((x_f[1:], x_f[-1:]), axis=0)

    return t, st, u_end, x_f

if __name__ == '__main__':
    
    T = 0.5; N = 100 
    # rob_diam = 0.3  # [m]
    alpha_max = np.pi/4; v_max = 10.0

    # x, y, theta (trailer), beta (truck)
    x1 = ca.SX.sym('x1'); y1 = ca.SX.sym('y1')
    theta1 = ca.SX.sym('theta1'); beta1 = ca.SX.sym('beta1')
    states = ca.vertcat(x1, y1, theta1, beta1)
    n_states = states.size()[0]

    # alpha (truck), v (truck)
    alpha1 = ca.SX.sym('alpha1'); v1 = ca.SX.sym('v1')
    controls = ca.vertcat(alpha1, v1)
    n_controls = controls.size()[0]

    # rhs
    # xdot = f(x, u)
    M1 = 2; L1 = 8; W1 = 4; L2 = 12; W2= 3
    rhs = ca.vertcat(v1*ca.cos(beta1)*(1 + M1/L1*ca.tan(beta1)*ca.tan(alpha1))*ca.cos(theta1), 
                 v1*ca.cos(beta1)*(1 + M1/L1*ca.tan(beta1)*ca.tan(alpha1))*ca.sin(theta1),
                 v1*(ca.sin(beta1)/L2 - M1/(L1*L2)*ca.cos(beta1)*ca.tan(alpha1)),
                 v1*(ca.tan(alpha1)/L1 - ca.sin(beta1)/L2 + M1/(L1*L2)*ca.cos(beta1)*ca.tan(alpha1)))
    # f = ca.Function('f', [states, controls], [rhs])

    # function
    f = ca.Function('f', [states, controls], [rhs], [
                    'input_state', 'control_input'], ['rhs'])
    
    # dynamics ---> optimization OCP
    U = ca.SX.sym('U', n_controls, N)
    X = ca.SX.sym('X', n_states, N+1)
    P = ca.SX.sym('P', n_states+n_states)

    # quadratic
    Q = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, .0], 
                  [0.0, 0.0, 2.0, .0], [0.0, 0.0, 0.0, 2.0]])
    R = np.array([[0.5, 0.0], [0.0, 0.05]])

    obj = 0 
    g = []  # equality constraint
    g.append(X[:, 0]-P[:4])
    for i in range(N):
        obj = obj + ca.mtimes([(X[:, i]-P[4:]).T, Q, X[:, i]-P[4:]]
                              ) + ca.mtimes([U[:, i].T, R, U[:, i]])
        x_next_ = f(X[:, i], U[:, i])*T + X[:, i]
        g.append(X[:, i+1]-x_next_)

    opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

    nlp_prob = {'f': obj, 'x': opt_variables, 'p': P, 'g': ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter': 1000, 'ipopt.print_level': 5, 'print_time': 0,
                    'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}
    
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    lbg = 0.0
    ubg = 0.0
    lbx = []
    ubx = []

    # opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
    for _ in range(N):
        lbx.append(-alpha_max)
        lbx.append(-v_max)
        ubx.append(alpha_max)
        ubx.append(v_max)

    for _ in range(N+1):  # note that this is different with the method using structure
        lbx.append(-np.inf)
        lbx.append(-np.inf)
        lbx.append(-np.inf)
        lbx.append(-np.pi/3.0)

        ubx.append(np.inf)
        ubx.append(np.inf)
        ubx.append(np.inf)
        ubx.append(np.pi/3.0)

# Simulation
# Iteration ---> feedback
    t0 = 0.0
    x0 = np.array([-55, 30, 0, 0]).reshape(-1, 1)  # initial state
    x0_ = x0.copy()
    x_m = np.zeros((n_states, N+1))
    next_states = x_m.copy().T

    xs = np.array([24, -20, np.pi/2, 0]).reshape(-1, 1)  # final state
    u0 = np.array([1, 2]*N).reshape(-1, 2).T  # np.ones((N, 2)) # controls

    x_c = [] # x history
    u_c = [] # x history
    t_c = [] # time
    xx = []
    sim_time = 20.0

    # start !!
    mpciter = 0
    start_time = time.time()
    index_t = []
    # loop
    while(np.linalg.norm(x0-xs) > 1e-2 and mpciter-sim_time/T < 0.0):
        # parameter
        c_p = np.concatenate((x0, xs))
        init_control = np.concatenate((u0.reshape(-1, 1), next_states.reshape(-1, 1)))
        t_ = time.time()
        res = solver(x0=init_control, p=c_p, lbg=lbg,
                     lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time() - t_)

        estimated_opt = res['x'].full()
        u0 = estimated_opt[:200].reshape(N, n_controls)  # (N, n_controls)
        x_m = estimated_opt[200:].reshape(N+1, n_states)  # (N+1, n_states)
        
        x_c.append(x_m.T)
        u_c.append(u0[0, :])
        t_c.append(t0)
        
        # shift_movement ---> iteration ---> feedback
        t0, x0, u0, next_states = shift_movement(T, t0, x0, u0, x_m, f)
        x0 = ca.reshape(x0, -1, 1)
        x0 = x0.full()
        xx.append(x0)
        # print(u0[0])
        mpciter = mpciter + 1

    t_v = np.array(index_t)
    print("time mean: ",t_v.mean())
    print("time/mpciter: ", (time.time() - start_time)/(mpciter))

    x_c = np.array(x_c)
    u_c = np.array(u_c)
    t_c = np.array(t_c)

    # states
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(t_c, x_c[:, 0, 0])
    plt.grid(True)
    plt.title('States over Time')
    plt.ylabel('x position')

    plt.subplot(3, 1, 2)
    plt.plot(t_c, x_c[:, 1, 0])
    plt.grid(True)
    plt.ylabel('y position')

    plt.subplot(3, 1, 3)
    plt.plot(t_c, x_c[:, 2, 0])
    plt.grid(True)
    plt.ylabel('theta (orientation)')
    plt.xlabel('Time (s)')

    plt.tight_layout()
    plt.show()

    # control inputs
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(t_c, u_c[:, 0])
    plt.grid(True)
    plt.title('Control Inputs over Time')
    plt.ylabel('Linear Velocity (v)')

    plt.subplot(2, 1, 2)
    plt.plot(t_c, u_c[:, 1])
    plt.grid(True)
    plt.ylabel('Angular Velocity (omega)')
    plt.xlabel('Time (s)')

    plt.tight_layout()
    plt.show()

    # trajectory
    plt.figure()
    plt.plot(x_c[:, 0, 0], x_c[:, 1, 0])
    plt.grid(True)
    plt.title('Trajectory')
    plt.xlabel('x position')
    plt.ylabel('y position')
    plt.savefig('C:/Users/14404/OneDrive/Desktop/PythonMathematics/MPC/Casadi/trajectoryfeedback.png')
    plt.show()
