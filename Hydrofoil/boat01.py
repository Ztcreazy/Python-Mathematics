import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

T = 0.02 
N = int(10.0 / T)

# x, y, z, Psi (Psi), theta (theta), Phi (Phi)
x = ca.SX.sym('x'); y = ca.SX.sym('y'); z = ca.SX.sym('z')
Psi = ca.SX.sym('Psi'); theta = ca.SX.sym('theta'); Phi = ca.SX.sym('Phi')
states = ca.vertcat(x, y, z, Psi, theta, Phi)
n_states = states.size()[0]

# u, v, w speed
u = ca.SX.sym('u'); v = ca.SX.sym('v'); w = ca.SX.sym('w')
# angular speed
p = ca.SX.sym('p'); q = ca.SX.sym('q'); r = ca.SX.sym('r')
controls = ca.vertcat(u, v, w, p, q, r)
n_controls = controls.size()[0]

# xdot = f(x, u)
rhs = ca.vertcat(ca.cos(theta) * ca.cos(Psi) * u + (ca.sin(Phi) * ca.sin(theta) * ca.cos(Psi) - ca.cos(Phi) * ca.sin(Psi)) * v + \
                (ca.cos(Phi) * ca.sin(theta) * ca.cos(Psi) + ca.sin(Phi) * ca.sin(Psi)) * w,
                ca.cos(theta) * ca.sin(Psi) * u + (ca.sin(Phi) * ca.sin(theta) * ca.sin(Psi) + ca.cos(Phi) * ca.cos(Psi)) * v + \
                (ca.cos(Phi) * ca.sin(theta) * ca.sin(Psi) - ca.sin(Phi) * ca.cos(Psi)) * w,
                -ca.sin(theta) * u + ca.sin(Phi) * ca.cos(theta) * v + ca.cos(Phi) * ca.cos(theta) * w,
                (ca.sin(Phi) * q + ca.cos(Phi) * r) / ca.cos(theta),
                ca.cos(Phi) * q - ca.sin(Phi) * r, 
                p + (ca.sin(Phi) * q + ca.cos(Phi) * r) * ca.tan(theta))
f = ca.Function('f', [states, controls], [rhs])

# dynamics ---> optimization
# decision/optimization variable
U = ca.SX.sym('U', n_controls, N); X = ca.SX.sym('X', n_states, N+1)
P = ca.SX.sym('P', n_states + n_states)

# quadratic
Q = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 5.0, 0.0, 0.0, 0.0, 0.0], 
              [0.0, 0.0, .1, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 2.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])

R = np.array([[0.5, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.5, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.5, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.5]])

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

lbx = []
ubx = []

u1_max = 10; u2_max = 10; u3_max = 10

u4_max = 5; u5_max = 5; u6_max = 5

for _ in range(N):
    lbx.append(-u1_max)
    lbx.append(-u2_max)
    lbx.append(-u3_max)
    lbx.append(-u4_max)
    lbx.append(-u5_max)
    lbx.append(-u6_max)

    ubx.append(u1_max)
    ubx.append(u2_max)
    ubx.append(u3_max)
    ubx.append(u4_max)
    ubx.append(u5_max)
    ubx.append(u6_max)


for _ in range(N+1): # x, y, z, yaw, pitch, roll
    lbx.append(-10.0)
    lbx.append(-10.0)
    lbx.append(-2.0)
    lbx.append(-np.pi)
    lbx.append(-np.pi)
    lbx.append(-np.pi)

    ubx.append(10.0)
    ubx.append(10.0)
    ubx.append(2.0)
    ubx.append(np.pi)
    ubx.append(np.pi)
    ubx.append(np.pi)

# !! x, y, z, yaw, pitch, roll
x0 = np.array([-4.0, -2.5, -0.2, np.pi/2.0, 0.0, 0.0])
xs = np.array([4.0, 2.5, 0.0, 0.0, 0.0, .0])

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

plt.subplot(3, 2, 1)
plt.plot(t_c, x_opt[:, 0]); plt.grid(True)
plt.title('States over Time')
plt.ylabel('x position')

plt.subplot(3, 2, 2)
plt.plot(t_c, x_opt[:, 1]); plt.grid(True)
plt.ylabel('y position')

plt.subplot(3, 2, 3)
plt.plot(t_c, x_opt[:, 2]); plt.grid(True)
plt.ylabel('z position')
plt.xlabel('Time (s)')

plt.subplot(3, 2, 4)
plt.plot(t_c, x_opt[:, 3]); plt.grid(True)
plt.title('States over Time')
plt.ylabel('Psi yaw')

plt.subplot(3, 2, 5)
plt.plot(t_c, x_opt[:, 4]); plt.grid(True)
plt.ylabel('theta pitch')

plt.subplot(3, 2, 6)
plt.plot(t_c, x_opt[:, 5]); plt.grid(True)
plt.ylabel('Phi roll')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()



# control inputs
t_c_u = np.linspace(0, T*(N-1), N)
plt.figure()

plt.subplot(3, 2, 1)
plt.plot(t_c_u, u_opt[:, 0]); plt.grid(True)
plt.title('Control inputs over Time')
plt.ylabel('u speed')

plt.subplot(3, 2, 2)
plt.plot(t_c_u, u_opt[:, 1]); plt.grid(True)
plt.ylabel('v speed')

plt.subplot(3, 2, 3)
plt.plot(t_c_u, u_opt[:, 2]); plt.grid(True)
plt.ylabel('w speed')
plt.xlabel('Time (s)')

plt.subplot(3, 2, 4)
plt.plot(t_c_u, u_opt[:, 3]); plt.grid(True)
plt.title('Control inputs over Time')
plt.ylabel('p angular speed')

plt.subplot(3, 2, 5)
plt.plot(t_c_u, u_opt[:, 4]); plt.grid(True)
plt.ylabel('q angular speed')

plt.subplot(3, 2, 6)
plt.plot(t_c_u, u_opt[:, 5]); plt.grid(True)
plt.ylabel('r angular speed')
plt.xlabel('Time (s)')

plt.tight_layout()
plt.show()


# !! animation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter

hull_vertices = np.array([
    [-1, -0.5, -0.5], [1, -0.5, -0.5], [1, 0.5, -0.5], [-1, 0.5, -0.5],  # Bottom face
    [-1, -0.5, 0.5], [1, -0.5, 0.5], [1, 0.5, 0.5], [-1, 0.5, 0.5]   # Top face
])

# bow_vertices = np.array([
#     [-1, -0.5, -0.5], [-1.5, -0.5, -0.5], [-1.5, 0, -0.75], [-1, 0, -0.75],  # Bow
#     [-1, -0.5, 0.5], [-1.5, -0.5, 0.5], [-1.5, 0, 0.75], [-1, 0, 0.75]
# ])
bow_vertices = np.array([
    [-1, -0.5, -0.5], [-1.5, -0.5, -0.5], [-1.5, -0.75, 0], [-1, -0.75, 0],  # Bow flat face down
    [-1, 0.5, -0.5], [-1.5, 0.5, -0.5], [-1.5, 0.75, 0], [-1, 0.75, 0]      # Bow flat face up
])

# Define the 6 faces of the hull and bow
hull_faces = [
    [hull_vertices[0], hull_vertices[1], hull_vertices[2], hull_vertices[3]],  # Bottom
    [hull_vertices[4], hull_vertices[5], hull_vertices[6], hull_vertices[7]],  # Top
    [hull_vertices[0], hull_vertices[1], hull_vertices[5], hull_vertices[4]],  # Front
    [hull_vertices[2], hull_vertices[3], hull_vertices[7], hull_vertices[6]],  # Back
    [hull_vertices[0], hull_vertices[3], hull_vertices[7], hull_vertices[4]],  # Left
    [hull_vertices[1], hull_vertices[2], hull_vertices[6], hull_vertices[5]]   # Right
]

# bow_faces = [
#     [bow_vertices[0], bow_vertices[1], bow_vertices[2], bow_vertices[3]],  # Front face
#     [bow_vertices[4], bow_vertices[5], bow_vertices[6], bow_vertices[7]],  # Back face
#     [bow_vertices[0], bow_vertices[1], bow_vertices[5], bow_vertices[4]],  # Bottom face
#     [bow_vertices[3], bow_vertices[2], bow_vertices[6], bow_vertices[7]],  # Top face
#     [bow_vertices[0], bow_vertices[3], bow_vertices[7], bow_vertices[4]],  # Left face
#     [bow_vertices[1], bow_vertices[2], bow_vertices[6], bow_vertices[5]]   # Right face
# ]

bow_faces = [
    [bow_vertices[0], bow_vertices[1], bow_vertices[2], bow_vertices[3]],  # Bottom flat face
    [bow_vertices[4], bow_vertices[5], bow_vertices[6], bow_vertices[7]],  # Top flat face
    [bow_vertices[0], bow_vertices[1], bow_vertices[5], bow_vertices[4]],  # Front face
    [bow_vertices[3], bow_vertices[2], bow_vertices[6], bow_vertices[7]],  # Back face
    [bow_vertices[0], bow_vertices[3], bow_vertices[7], bow_vertices[4]],  # Left face
    [bow_vertices[1], bow_vertices[2], bow_vertices[6], bow_vertices[5]]   # Right face
]

# Set up figure and 3D axis with a larger figure size
fig = plt.figure(figsize=(12, 10))  # Increase figure size here
ax = fig.add_subplot(111, projection='3d')

# Set the limits for the axes to increase the viewing area
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_zlim(-2.5, 2.5)
ax.set_box_aspect([5, 5, 2.5])

# Create Poly3DCollection for the hull and bow without color specification
hull = Poly3DCollection(hull_faces, linewidths=1, edgecolors='k', alpha=.6)
bow = Poly3DCollection(bow_faces, linewidths=1, edgecolors='k', alpha=.8)
ax.add_collection3d(hull)
ax.add_collection3d(bow)

# Add a point on the boat's position from x_opt data
point, = ax.plot([], [], [], 'ro')  # 'ro' for a red point

trajectory_line, = ax.plot([], [], [], 'r-', lw=2)

# Initialize the text object to display the states
state_text = ax.text2D(0.05, 0.95, "", transform=ax.transAxes, fontsize=12)

# Lists to store the trajectory points
x_traj, y_traj, z_traj = [], [], []

def update(num, x, y, z, yaw, pitch, roll):
    # Compute rotation matrices for yaw, pitch, and roll
    R_yaw = np.array([[np.cos(yaw[num]), -np.sin(yaw[num]), 0],
                      [np.sin(yaw[num]), np.cos(yaw[num]), 0],
                      [0, 0, 1]])

    R_pitch = np.array([[np.cos(pitch[num]), 0, np.sin(pitch[num])],
                        [0, 1, 0],
                        [-np.sin(pitch[num]), 0, np.cos(pitch[num])]])

    R_roll = np.array([[1, 0, 0],
                       [0, np.cos(roll[num]), -np.sin(roll[num])],
                       [0, np.sin(roll[num]), np.cos(roll[num])]])

    R = R_yaw @ R_pitch @ R_roll

    # Apply rotation and translation to hull and bow faces
    transformed_hull_faces = []
    for face in hull_faces:
        face = np.array(face)
        rotated_face = (R @ face.T).T
        translated_face = rotated_face + np.array([x[num], y[num], z[num]])
        transformed_hull_faces.append(translated_face)
    
    transformed_bow_faces = []
    for face in bow_faces:
        face = np.array(face)
        rotated_face = (R @ face.T).T
        translated_face = rotated_face + np.array([x[num], y[num], z[num]])
        transformed_bow_faces.append(translated_face)

    # Update the Poly3DCollection objects
    hull.set_verts(transformed_hull_faces)
    bow.set_verts(transformed_bow_faces)

    # Update the point's position based on the x_opt data
    point.set_data(x[num], y[num])
    point.set_3d_properties(z[num])

    # Add current point to the trajectory
    x_traj.append(x[num])
    y_traj.append(y[num])
    z_traj.append(z[num])

    # Update the trajectory line
    trajectory_line.set_data(x_traj, y_traj)
    trajectory_line.set_3d_properties(z_traj)

    # Update the state text
    state_text.set_text(f"Position: ({x[num]:.2f}, {y[num]:.2f}, {z[num]:.2f})\n"
                        f"Yaw: {np.degrees(yaw[num]):.2f}°\n"
                        f"Pitch: {np.degrees(pitch[num]):.2f}°\n"
                        f"Roll: {np.degrees(roll[num]):.2f}°")

    return hull, bow, point, state_text, trajectory_line

# start_frame = 0
# update(start_frame, x_opt[start_frame:,0], x_opt[start_frame:,1], x_opt[start_frame:,2], 
#        x_opt[start_frame:,3], x_opt[start_frame:,4], x_opt[start_frame:,5])

ani = FuncAnimation(fig, update, frames=len(x_opt[1:, 0]), fargs=(x_opt[1:, 0], x_opt[1:, 1], x_opt[1:, 2], 
                                                       x_opt[1:, 3], x_opt[1:, 4], x_opt[1:, 5]), 
                                                       interval=100, blit=False)
ax.view_init(elev=30, azim=60)
# ax.view_init(elev=90, azim=-90)
plt.show()

gif_writer = PillowWriter(fps=5)
ani.save("C:/Users/14404/OneDrive/Desktop/PythonMathematics/Hydrofoil/animation.gif", writer=gif_writer)


