# k = pyairy.dlqri(ad, bd, q=q, r=r,
#                      ni=ni, dt=dt, bryson=True)[0]

import control as ct
import numpy as np

import casadi as ca

def mpc(a, b, q, r, n=None, ni=0, dofs_i_local=[], dt=1):

    T = dt*0.01
    N = int(dt / T)

    # linear, discrete, MPC
    # x, y, z, Psi (Psi), theta (theta), Phi (Phi)
    x = ca.SX.sym('x'); y = ca.SX.sym('y'); z = ca.SX.sym('z')
    Psi = ca.SX.sym('Psi'); theta = ca.SX.sym('theta'); Phi = ca.SX.sym('Phi')
    # u, v, w (speed)
    # p, q, r (angular speed)
    u = ca.SX.sym('u'); v = ca.SX.sym('v'); w = ca.SX.sym('w')
    p = ca.SX.sym('p'); q = ca.SX.sym('q'); r = ca.SX.sym('r')
    states = ca.vertcat(x, y, z, Psi, theta, Phi, u, v, w, p, q, r)
    n_states = states.size()[0]

    # F force
    Fx = ca.SX.sym('Fx'); Fy = ca.SX.sym('Fy'); Fz = ca.SX.sym('Fz')
    # M moments
    Mx = ca.SX.sym('Mx'); My = ca.SX.sym('My'); Mz = ca.SX.sym('Mz')
    controls = ca.vertcat(Fx, Fy, Fz, Mx, My, Mz)
    n_controls = controls.size()[0]

    # Digital Control of Dynamic Systems, Gene F. Franklin, page 324, expression 8.84:
    c = np.eye(a.shape[0])
    if dofs_i_local is None:
        dofs_i_local = np.arange(ni)
    else:
        ni = len(dofs_i_local)
        
    ci = c[dofs_i_local, :] 

    # integration 12*12 ---> 14*14
    aa = np.block([[np.eye(ni), ci],
                   [np.zeros((a.shape[0], ni)), a]])
    # integration: xdot = A*x + B*u
    ba = np.block([[np.zeros((ni, b.shape[1]))], [b]])

    qd, rd, nd = discretize_cost(aa, ba, q, r, n, dt=dt)

    A_ca = ca.SX(aa)
    B_ca = ca.SX(ba)

    # xdot = f(x, u)
    rhs = ca.mtimes(A_ca, states) + ca.mtimes(B_ca, controls)
    f = ca.Function('f', [states, controls], [rhs])

    # dynamics ---> optimization
    # decision/optimization variable
    U = ca.SX.sym('U', n_controls, N); X = ca.SX.sym('X', n_states, N+1)
    P = ca.SX.sym('P', n_states + n_states)

    # a = []; b = [] # xdot = A*x + B*u
    q = [] # quadratics
    r = [] # cost

    Q = np.array(q); R = np.array(r)

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

    opts_setting = {'ipopt.max_iter': 2000, 'ipopt.print_level': 5, 'print_time': 0,
                    'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}
    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    # inequality constraint ---> function
    lbg = [0.0] * (N * n_states + n_states)
    ubg = [0.0] * (N * n_states + n_states)

    lbx = []; ubx = []

    u1_max = 10; u2_max = 10; u3_max = 10
    u4_max = 5; u5_max = 5; u6_max = 5

    for _ in range(N):

        # control inputs ---> lower bound
        lbx.append(-u1_max)
        lbx.append(-u2_max)
        lbx.append(-u3_max)
        lbx.append(-u4_max)
        lbx.append(-u5_max)
        lbx.append(-u6_max)

        # upper bound
        ubx.append(u1_max)
        ubx.append(u2_max)
        ubx.append(u3_max)
        ubx.append(u4_max)
        ubx.append(u5_max)
        ubx.append(u6_max)


    for _ in range(N+1): # x, y, z, yaw, pitch, roll, u, v, w, p, q, r
        
        # position + angle ---> lower bound
        lbx.append(-10.0); lbx.append(-10.0); lbx.append(-2.0)
        lbx.append(-np.pi); lbx.append(-np.pi); lbx.append(-np.pi)

        # speed + angular speed ---> lower bound
        lbx.append(0.0); lbx.append(0.0); lbx.append(-2.0)
        lbx.append(-np.pi); lbx.append(-np.pi); lbx.append(-np.pi)

        # position + angle ---> upper bound
        ubx.append(10.0); ubx.append(10.0); ubx.append(2.0)
        ubx.append(np.pi); ubx.append(np.pi); ubx.append(np.pi)

        # speed + angular speed ---> upper bound 
        ubx.append(10.0); ubx.append(10.0); ubx.append(2.0)
        ubx.append(np.pi); ubx.append(np.pi); ubx.append(np.pi)

    # !! states x_k
    x0 = np.array([])
    # setpoint x_ref
    xs = np.array([])

    # Initial guess
    c_p = np.concatenate((x0, xs)) 
    init_control = np.concatenate((np.zeros((N, n_controls)).reshape(-1, 1),
                                np.zeros((N+1, n_states)).reshape(-1, 1)))

    res = solver(x0=init_control, p=c_p, lbg=lbg, ubg=ubg, lbx=lbx, ubx=ubx)
    estimated_opt = res['x'].full()

    u_opt = estimated_opt[:N*n_controls].reshape(N, n_controls)
    x_opt = estimated_opt[N*n_controls:].reshape(N+1, n_states)



def dlqri(a, b, q, r, n=None, ni=0, dofs_i_local=[], dt=1, bryson=False):
    """! Computes the optimal linear quadratic state feedback controller for a given linear system.
    The discretization of the optimization problem is managed internally
    @param a: Discrete state space representation dynamics  matrix
    @param b: Discrete state space representation input matrix
    @param q: State weight matrix
    @param r: Input weight matrix
    @param ni: Number of integrators
    @param dofs_i_local: Indexes of the states that will be integrated
    @param dt: Sample time
    @param bryson: Select whether to use Bryson's rule for weight matrices or not
    """

    if n is None:
        n = np.zeros((len(q), len(r)))
    if bryson:
        q = np.diag(1 / np.array(q) ** 2)
        r = np.diag(1 / np.array(r) ** 2)
        n = np.array([[1/el if el != 0 else 0 for el in row] for row in n]) # Avoid division by zero
        
    # Digital Control of Dynamic Systems, Gene F. Franklin, page 324, expression 8.84:
    c = np.eye(a.shape[0])
    if dofs_i_local is None:
        dofs_i_local = np.arange(ni)
    else:
        ni = len(dofs_i_local)
        
    ci = c[dofs_i_local, :] # Modified to allow for any dof to be integrated

    aa = np.block([[np.eye(ni), ci],
                   [np.zeros((a.shape[0], ni)), a]])
    ba = np.block([[np.zeros((ni, b.shape[1]))], [b]])

    qd, rd, nd = discretize_cost(aa, ba, q, r, n, dt=dt)

    # print("Qd: ", qd)
    # print("Rd: ", rd)
    # print("Nd: ", nd) 
    
    
    def print_matlab_matrix(matrix, label="matrix"):
      rows, cols = matrix.shape
      print(f"{label} = [")
      for i in range(rows):
          row_str = ", ".join([f"{elem:.15e}" for elem in matrix[i]])
          print(f"{row_str};" if i < rows - 1 else f"{row_str}")
      print("];")
        
    try:
        (k, s, e) = ct.dlqr(aa, ba, qd, rd, nd)
        # (k, s, e) = ct.matlab.dlqr(aa, ba, qd, rd, nd)
        # print(f"Worked:")
        # print_matlab_matrix(aa, "aa")
        # print_matlab_matrix(ba, "ba")
        # print_matlab_matrix(qd, "qd")
        # print_matlab_matrix(rd, "rd")
        # print_matlab_matrix(nd, "nd")
        # print_matlab_matrix(k, "k")

    except Exception as e:
        print("Error in dlqri: ", e)
        print_matlab_matrix(aa, "aa")
        print_matlab_matrix(ba, "ba")
        print_matlab_matrix(qd, "qd")
        print_matlab_matrix(rd, "rd")
        print_matlab_matrix(nd, "nd")
        # print(f"Det(A) = {np.linalg.det(aa)}")
        # print(f"Det(Q) = {np.linalg.det(qd)}")
        # print(f"Det(R) = {np.linalg.det(rd)}")
        raise e
    return (k, s, e)
