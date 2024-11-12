import numpy as np

def Kinematics(x, U):
    yaw = x[3]
    pitch = x[4]
    roll = x[5]
    u = U[0]
    v = U[1]
    w = U[2]
    p = U[3]
    q = U[4]
    r = U[5]

    xdot = np.zeros((6, 1))

    xdot[0, 0] = np.cos(pitch) * np.cos(yaw) * u + (np.sin(roll) * np.sin(pitch) * np.cos(yaw) - np.cos(roll) * np.sin(yaw)) * v + \
                 (np.cos(roll) * np.sin(pitch) * np.cos(yaw) + np.sin(roll) * np.sin(yaw)) * w

    xdot[1, 0] = np.cos(pitch) * np.sin(yaw) * u + (np.sin(roll) * np.sin(pitch) * np.sin(yaw) + np.cos(roll) * np.cos(yaw)) * v + \
                 (np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.sin(roll) * np.cos(yaw)) * w

    xdot[2, 0] = -np.sin(pitch) * u + np.sin(roll) * np.cos(pitch) * v + np.cos(roll) * np.cos(pitch) * w

    xdot[3, 0] = (np.sin(roll) * q + np.cos(roll) * r) / np.cos(pitch)

    xdot[4, 0] = np.cos(roll) * q - np.sin(roll) * r

    xdot[5, 0] = p + (np.sin(roll) * q + np.cos(roll) * r) * np.tan(pitch)

    return xdot.flatten().tolist()