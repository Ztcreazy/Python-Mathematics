import casadi as ca
import casadi.tools as ca_tools

import numpy as np
import time

import sys
sys.path.append('C:/Users/14404/OneDrive/Desktop/PythonMathematics/MPC/Casadi')
from SingleShooting.draw import Draw_MPC_Obstacle

def shift_movement(T, t0, x0, u, x_f, f):
    f_value = f( x0, u[:, 0] )
    st = x0 + T *f_value.full()
    t = t0 + T
    u_end = np.concatenate( (u[:, 1:], u[:, -1:]), axis=1)
    x_f = np.concatenate( (x_f[:, 1:], x_f[:, -1:]), axis=1)

    return t, st, u_end, x_f

if __name__ == '__main__':
    # T = 0.2 # sampling time [s]
    # N = 100 # prediction horizon
    # rob_diam = 0.3 # [m]
    # v_max = 0.6
    # omega_max = np.pi/4.0

    T = 0.2
    N = 100

    del_min = -2 *np.pi
    del_max =  2* np.pi
    vel_min = 0.0
    vel_max = 50.8
    psi_min = -2 *np.pi
    psi_max =  2 *np.pi
    
    vel_delta_min = 0.0
    vel_delta_max = 10.0
    
    a_long_min = -11.5
    a_long_max =  11.5

    # x = ca.SX.sym('x')
    # y = ca.SX.sym('y')
    # theta = ca.SX.sym('theta')
    # states = ca.vertcat(x, y)
    # states = ca.vertcat(states, theta)

    sx    = ca.SX.sym('sx')
    sy    = ca.SX.sym('sy')
    delta = ca.SX.sym('delta')
    vel   = ca.SX.sym('vel')
    psi   = ca.SX.sym('psi')
    states = ca.vertcat(sx, sy, delta, vel, psi)
    n_states = states.size()[0]

    # v = ca.SX.sym('v')
    # omega = ca.SX.sym('omega')
    # controls = ca.vertcat(v, omega)
    # n_controls = controls.size()[0]

    v_delta = ca.SX.sym('v_delta')
    a_long = ca.SX.sym('a_long')
    controls = ca.vertcat(v_delta, a_long)
    n_controls = controls.size()[0]

    ## rhs
    l_wb = 2.5891
    rhs = ca.vertcat(vel*ca.cos(psi), vel*ca.sin(psi), v_delta, a_long, vel / l_wb * np.tan(delta) )
    

    ## function
    f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

    ## for MPC
    U = ca.SX.sym('U', n_controls, N)
    X = ca.SX.sym('X', n_states, N+1)
    P = ca.SX.sym('P', n_states+n_states)

    ### define
    # Q = np.array( [[1.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, .1]] )

    Q = np.array( [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0, 0.0], 
                   [0.0, 0.0, 0.1, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0],
                   [0.0, 0.0, 0.0, 0.0, 1.0]] )

    R = np.array([[0.5, 0.0], [0.0, 0.05]])
    #### cost function
    obj = 0 #### cost
    g = [] # equal constrains



    g.append(X[:, 0]-P[:5])



    for i in range(N):
        obj = obj + ca.mtimes([(X[:, i]-P[5:]).T, Q, X[:, i]-P[5:]]) + ca.mtimes([U[:, i].T, R, U[:, i]])
        x_next_ = f(X[:, i], U[:, i])*T + X[:, i]
        g.append(X[:, i+1]-x_next_)

    #### constraints
    obs_x = 0.5
    obs_y = 0.5
    obs_diam = 0.3



    for i in range(N+1):
        g.append(ca.sqrt((X[0, i]-obs_x)**2+(X[1, i]-obs_y)**2)) # should be smaller als 0.0



    opt_variables = ca.vertcat( ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

    nlp_prob = {'f': obj, 'x': opt_variables, 'p':P, 'g':ca.vertcat(*g)}
    opts_setting = {'ipopt.max_iter':100, 'ipopt.print_level':5, 'print_time':0, 'ipopt.acceptable_tol':1e-8, 'ipopt.acceptable_obj_change_tol':1e-6}

    solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

    lbg = []
    ubg = []
    lbx = []
    ubx = []
    for _ in range(N+1): # g.append(X[:, 0]-P[:5]) ----> 3, 5
        lbg.append(0.0)  # for i in range(N): g.append(X[:, i+1]-x_next_)
        lbg.append(0.0)
        lbg.append(0.0)
        lbg.append(0.0)
        lbg.append(0.0)

        ubg.append(0.0)
        ubg.append(0.0)
        ubg.append(0.0)
        ubg.append(0.0)
        ubg.append(0.0)
    
    for _ in range(N+1): # for i in range(N+1): g.append(ca.sqrt((X[0, i]-obs_x)**2+(X[1, i]-obs_y)**2))
        lbg.append(0.3)
        ubg.append(5.0) # np.inf

    # opt_variables = ca.vertcat( ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))
    # ca.reshape(U, -1, 1)
    for _ in range(N): # controls
        lbx.append(vel_delta_min)
        lbx.append(a_long_min)
        
        ubx.append(vel_delta_max)
        ubx.append(a_long_max)

    for _ in range(N+1): # note that this is different with the method using structure
        lbx.append(-2.0) # states
        lbx.append(-2.0)
        lbx.append(del_min)
        lbx.append(vel_min)
        lbx.append(psi_min)
        
        ubx.append(5.0)
        ubx.append(5.0)
        ubx.append(del_max)
        ubx.append(vel_max)
        ubx.append(psi_max)

    # Simulation
    t0 = 0.0
    # x0 = np.array( [0.0, 0.0, 0.0] ).reshape(-1, 1) # initial state
    x0 = np.array( [0.0, 0.0, 0.0, 0.0, 0.0] ).reshape(-1, 1)
    x0_ = x0.copy()
    x_m = np.zeros( (n_states, N+1) )
    next_states = x_m.copy()

    # xs = np.array( [1.5, 1.5, 0.0] ).reshape(-1, 1) # final state
    xs = np.array( [2.0, 1.5, 0.0, 0.0, 0.0] ).reshape(-1, 1)

    u0 = np.array( [1, 2]*N ).reshape(-1, 2).T # np.ones((N, 2)) # controls
    x_c = [] # contains for the history of the state
    u_c = []
    t_c = [t0] # for the time
    xx = []
    sim_time = 20.0

    ## start MPC
    mpciter = 0
    start_time = time.time()
    index_t = []
    while(np.linalg.norm(x0-xs)>1e-2 and mpciter-sim_time/T<0.0 and mpciter<50):
        ## set parameter
        c_p = np.concatenate((x0, xs))
        # print('{0}'.format(next_states))
        # print('{0}'.format(next_states.T.reshape(-1, 1)[:6]))
        init_control = np.concatenate((u0.T.reshape(-1, 1), next_states.T.reshape(-1, 1)))
        t_ = time.time()
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time()- t_)

        estimated_opt = res['x'].full() # the feedback is in the series [u0, x0, u1, x1, ...]
        u0 = estimated_opt[:200].reshape(N, n_controls).T # (n_controls, N)
        x_m = estimated_opt[200:].reshape(N+1, n_states).T# [n_states, N]
        x_c.append(x_m.T)
        u_c.append(u0[:, 0])
        t_c.append(t0)

        t0, x0, u0, next_states = shift_movement(T, t0, x0, u0, x_m, f)
        x0 = ca.reshape(x0, -1, 1)
        x0 = x0.full()
        xx.append(x0)
        mpciter = mpciter + 1

    print(mpciter)
    t_v = np.array(index_t)
    print(t_v.mean())
    print((time.time() - start_time)/(mpciter))
    draw_result = Draw_MPC_Obstacle(rob_diam=0.3, init_state=x0_, target_state=xs, robot_states=xx, obstacle=np.array([obs_x, obs_y, obs_diam/2.]), export_fig=True)
