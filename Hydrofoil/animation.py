import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def Animation(states):
    length = 3
    width = 1.5
    height = 1.5

    V = Cuboid(length, width, height)

    faces = [
        [V[0], V[1], V[5], V[4]],
        [V[1], V[2], V[6], V[5]],
        [V[2], V[3], V[7], V[6]],
        [V[3], V[0], V[4], V[7]],
        [V[0], V[1], V[2], V[3]],
        [V[4], V[5], V[6], V[7]]
    ]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_zlim(-10, 2)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(-30, 30)

    # Plot initial cuboid
    poly = Poly3DCollection(faces, facecolors='black', edgecolors='black', alpha=0.7)
    ax.add_collection3d(poly)

    for k in range(states.shape[0]):
        state = states[k, :]
        Vt = transformCuboid(V, state)

        # Update cuboid vertices
        for i in range(len(faces)):
            for j in range(len(faces[i])):
                faces[i][j] = Vt[faces[i][j] - 1]

        # Update cuboid in the plot
        poly.set_verts(faces)

        plt.pause(0.1)

    plt.subplot(3, 2, 2)
    plt.plot(states[:, 0], 'g', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('X')
    plt.grid(True)

    plt.subplot(3, 2, 4)
    plt.plot(states[:, 1], 'r', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Y')
    plt.grid(True)

    plt.subplot(3, 2, 6)
    plt.plot(states[:, 2], 'b', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Z')
    plt.grid(True)

    plt.subplot(3, 2, 1)
    plt.plot(states[:, 3], 'c', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Yaw')
    plt.grid(True)

    plt.subplot(3, 2, 3)
    plt.plot(states[:, 4], 'm', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Pitch')
    plt.grid(True)

    plt.subplot(3, 2, 5)
    plt.plot(states[:, 5], 'y', linewidth=2)
    plt.xlabel('Time')
    plt.ylabel('Roll')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def Cuboid(length, width, height):
    V = np.array([
        [-length/2, -width/2, -height/2],
        [length/2, -width/2, -height/2],
        [length/2, width/2, -height/2],
        [-length/2, width/2, -height/2],
        [-length/2, -width/2, height/2],
        [length/2, -width/2, height/2],
        [length/2, width/2, height/2],
        [-length/2, width/2, height/2]
    ])
    return V


def transformCuboid(V, state):
    x, y, z, yaw, pitch, roll = state

    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])

    R_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])

    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])

    R = R_yaw.dot(R_pitch).dot(R_roll)

    Vt = (R.dot(V.T)).T + np.array([x, y, z])
    return Vt