import casadi as ca
import numpy as np
import matplotlib.pyplot as plt


Ts = 0.5
horizon = 100

sx = ca.MX.sym('x')
sy = ca.MX.sym('y')
sz = ca.MX.sym('z')
yaw = ca.MX.sym('yaw')
pitch = ca.MX.sym('pitch')
roll = ca.MX.sym('roll')

states = ca.vertcat(*[sx, sy, sz, yaw, pitch, roll])
num_states = states.size()[0]

u = ca.MX.sym('u')
v = ca.MX.sym('v')
w = ca.MX.sym('w')
p = ca.MX.sym('p')
q = ca.MX.sym('q')
r = ca.MX.sym('r')

controls = ca.vertcat(u, v, w, p, q, r)
num_controls = controls.size()[0]



# Define the kinematic model
def kinematics(states, controls):
    yaw = states[3]
    pitch = states[4]
    roll = states[5]
    u = controls[0]
    v = controls[1]
    w = controls[2]
    p = controls[3]
    q = controls[4]
    r = controls[5]
    
    x_dot = ca.MX.zeros(6, 1)
    
    x_dot[0] = ca.cos(pitch) * ca.cos(yaw) * u + (ca.sin(roll) * ca.sin(pitch) * ca.cos(yaw) - ca.cos(roll) * ca.sin(yaw)) * v + \
               (ca.cos(roll) * ca.sin(pitch) * ca.cos(yaw) + ca.sin(roll) * ca.sin(yaw)) * w
               
    x_dot[1] = ca.cos(pitch) * ca.sin(yaw) * u + (ca.sin(roll) * ca.sin(pitch) * ca.sin(yaw) + ca.cos(roll) * ca.cos(yaw)) * v + \
               (ca.cos(roll) * ca.sin(pitch) * ca.sin(yaw) - ca.sin(roll) * ca.cos(yaw)) * w
               
    x_dot[2] = -ca.sin(pitch) * u + ca.sin(roll) * ca.cos(pitch) * v + ca.cos(roll) * ca.cos(pitch) * w
    
    x_dot[3] = (ca.sin(roll) * q + ca.cos(roll) * r) / ca.cos(pitch)
    x_dot[4] = ca.cos(roll) * q - ca.sin(roll) * r
    x_dot[5] = p + (ca.sin(roll) * q + ca.cos(roll) * r) * ca.tan(pitch)
    
    return states + x_dot * Ts

rhs = kinematics(states, controls)
f = ca.Function('f', [states, controls], [rhs], ['input_state', 'control_input'], ['rhs'])

def cost(states, controls, target_pose):
    Q = np.eye(6) * 10 
    R = np.eye(6) * 1 
    state_error = states - target_pose
    return ca.mtimes([state_error.T, Q, state_error]) + ca.mtimes([controls.T, R, controls])



U = ca.SX.sym('U', num_controls, horizon)
X = ca.SX.sym('X', num_states, horizon + 1)

J = 0
g = []

target_pose = np.array([6, -6, 0, -np.pi/2, 0, 0])

for k in range(p):
    J += cost(X[:, k], U[:, k], target_pose)
    x_next = f(X[:, k], U[:, k])
    g.append(X[:, k + 1] - x_next)

J += cost(X[:, -1], ca.MX.zeros(6), target_pose)
g.append(X[:, -1] - target_pose)

nlp = {'x': ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1)),
       'f': J,
       'g': ca.vertcat(*g)}

opts = {'ipopt.print_level': 0, 'print_time': 0}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

initial_pose = np.array([-8, 8, 0, np.pi/2, 0, 0])
x0 = np.tile(initial_pose, (p + 1, 1)).flatten()
u0 = np.zeros(6 * p)

lbx = []
ubx = []

u1_min = -1
u1_max = 1
u2_min = -2
u2_max = 2
u3_min = -1
u3_max = 1

u4_min = -np.pi
u4_max = np.pi
u5_min = -np.pi
u5_max = np.pi
u6_min = -np.pi
u6_max = np.pi


for _ in range(horizon):
            
        lbx.append(u1_min)
        ubx.append(u1_max)
            
        lbx.append(u2_min)
        ubx.append(u2_max)

        lbx.append(u3_min)
        ubx.append(u3_max)
            
        lbx.append(u4_min)
        ubx.append(u4_max)

        lbx.append(u5_min)
        ubx.append(u5_max)
            
        lbx.append(u6_min)
        ubx.append(u6_max)

sx_min = -20
sx_max = 20

sy_min = -20
sy_max = 20

sz_min = -2
sz_max = 2

yaw_min = -np.pi/2.0
yaw_max = np.pi/2.0
pitch_min = -np.pi
pitch_max = np.pi
roll_min = np.pi
roll_max = -np.pi

for _ in range(horizon+1):  # note that this is different with the method using structure
            
        lbx.append(sx_min)
        lbx.append(sy_min)
        lbx.append(sz_min)
        lbx.append(yaw_min)
        lbx.append(pitch_min)
        lbx.append(roll_min)
            
        ubx.append(sx_max)
        ubx.append(sy_max)
        ubx.append(sz_max)
        ubx.append(yaw_max)
        ubx.append(pitch_max)
        ubx.append(roll_max)


lbg = []
ubg = []

for k in range(p+1):
    
    lbg.append(0.0)
    lbg.append(0.0)
    lbg.append(0.0)
    lbg.append(0.0)
    lbg.append(0.0)
    lbg.append(0.0)

    ubg.append(0.0)
    ubg.append(0.0)
    ubg.append(0.0)
    ubg.append(0.0)
    ubg.append(0.0)
    ubg.append(0.0)

sol = solver(x0=np.concatenate((x0, u0)), lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

x_current = initial_pose

for k in range(p):
    u_current = u_opt[k]
    x_current = f(x_current, u_current).full().flatten()
    print(f"State: {x_current}, Control: {u_current}")

# Plotting the results for better visualization (optional)
states_history = np.array([x_opt[k] for k in range(p + 1)])
plt.figure(figsize=(10, 6))
plt.plot(states_history[:, 0], states_history[:, 1], 'b-o', label='Trajectory')
plt.plot(initial_pose[0], initial_pose[1], 'go', label='Start')
plt.plot(target_pose[0], target_pose[1], 'ro', label='Target')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.grid()
plt.title('Trajectory from Initial to Target Position')
plt.show()
