import numpy as np

def Jacobian(x, U):
    yaw = x[3]
    pitch = x[4]
    roll = x[5]
    u = U[0]
    v = U[1]
    w = U[2]
    p = U[3]
    q = U[4]
    r = U[5]

    A = np.zeros((6, 6))

    A[0, 3] = w * (np.cos(yaw) * np.sin(roll) - np.cos(roll) * np.sin(pitch) * np.sin(yaw)) - \
              v * (np.cos(roll) * np.cos(yaw) + np.sin(pitch) * np.sin(roll) * np.sin(yaw)) - \
              u * np.cos(pitch) * np.sin(yaw)
    A[0, 4] = w * np.cos(pitch) * np.cos(roll) * np.cos(yaw) - u * np.cos(yaw) * np.sin(pitch) + \
              v * np.cos(pitch) * np.cos(yaw) * np.sin(roll)
    A[0, 5] = v * (np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.cos(yaw) * np.sin(pitch)) + \
              w * (np.cos(roll) * np.sin(yaw) - np.cos(yaw) * np.sin(pitch) * np.sin(roll))

    A[1, 3] = w * (np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.cos(yaw) * np.sin(pitch)) - \
              v * (np.cos(roll) * np.sin(yaw) - np.cos(yaw) * np.sin(pitch) * np.sin(roll)) + \
              u * np.cos(pitch) * np.cos(yaw)
    A[1, 4] = w * np.cos(pitch) * np.cos(roll) * np.sin(yaw) - u * np.sin(pitch) * np.sin(yaw) + \
              v * np.cos(pitch) * np.sin(roll) * np.sin(yaw)
    A[1, 5] = - v * (np.cos(yaw) * np.sin(roll) - np.cos(roll) * np.sin(pitch) * np.sin(yaw)) - \
              w * (np.cos(roll) * np.cos(yaw) + np.sin(pitch) * np.sin(roll) * np.sin(yaw))

    A[2, 4] = - u * np.cos(pitch) - w * np.cos(roll) * np.sin(pitch) - v * np.sin(pitch) * np.sin(roll)
    A[2, 5] = v * np.cos(pitch) * np.cos(roll) - w * np.cos(pitch) * np.sin(roll)

    A[3, 4] = (np.sin(pitch) * (r * np.cos(roll) + q * np.sin(roll))) / np.cos(pitch)**2
    A[3, 5] = (q * np.cos(roll) - r * np.sin(roll)) / np.cos(pitch)

    A[4, 5] = - r * np.cos(roll) - q * np.sin(roll)

    A[5, 4] = (np.tan(pitch)**2 + 1) * (r * np.cos(roll) + q * np.sin(roll))
    A[5, 5] = np.cos(roll) * np.tan(pitch) * (q * np.cos(roll) - r * np.sin(roll))

    B = np.zeros((6, 6))

    B[0, 0] = np.cos(pitch) * np.cos(yaw)
    B[0, 1] = np.cos(yaw) * np.sin(pitch) * np.sin(roll) - np.cos(roll) * np.sin(yaw)
    B[0, 2] = np.sin(roll) * np.sin(yaw) + np.cos(roll) * np.cos(yaw) * np.sin(pitch)

    B[1, 0] = np.cos(pitch) * np.sin(yaw)
    B[1, 1] = np.cos(roll) * np.cos(yaw) + np.sin(pitch) * np.sin(roll) * np.sin(yaw)
    B[1, 2] = np.cos(roll) * np.sin(pitch) * np.sin(yaw) - np.cos(yaw) * np.sin(roll)

    B[2, 0] = -np.sin(pitch)
    B[2, 1] = np.cos(pitch) * np.sin(roll)
    B[2, 2] = np.cos(pitch) * np.cos(roll)

    B[3, 4] = np.sin(roll) / np.cos(pitch)
    B[3, 5] = np.cos(roll) / np.cos(pitch)

    B[4, 4] = np.cos(roll)
    B[4, 5] = -np.sin(roll)

    B[5, 3] = 1
    B[5, 4] = np.tan(pitch) * np.sin(roll)
    B[5, 5] = np.cos(roll) * np.tan(pitch)

    return A, B