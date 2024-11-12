import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

T = 0.02 
N = int(10.0 / T)

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

# !!
# Moment of inertia

# xdot = f(x, u)
rhs = ca.vertcat(ca.cos(theta) * ca.cos(Psi) * u + (ca.sin(Phi) * ca.sin(theta) * ca.cos(Psi) - ca.cos(Phi) * ca.sin(Psi)) * v + \
                (ca.cos(Phi) * ca.sin(theta) * ca.cos(Psi) + ca.sin(Phi) * ca.sin(Psi)) * w,
                ca.cos(theta) * ca.sin(Psi) * u + (ca.sin(Phi) * ca.sin(theta) * ca.sin(Psi) + ca.cos(Phi) * ca.cos(Psi)) * v + \
                (ca.cos(Phi) * ca.sin(theta) * ca.sin(Psi) - ca.sin(Phi) * ca.cos(Psi)) * w,
                -ca.sin(theta) * u + ca.sin(Phi) * ca.cos(theta) * v + ca.cos(Phi) * ca.cos(theta) * w,
                (ca.sin(Phi) * q + ca.cos(Phi) * r) / ca.cos(theta), # Psi
                ca.cos(Phi) * q - ca.sin(Phi) * r, # theta
                p + (ca.sin(Phi) * q + ca.cos(Phi) * r) * ca.tan(theta), # Phi
                )
f = ca.Function('f', [states, controls], [rhs])
