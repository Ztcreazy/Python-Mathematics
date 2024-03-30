import numpy as np
import matplotlib.pyplot as plt

def z_function(x,y):
    return np.sin(5*x) * np.cos(5*y)/5

def z_gradient(x,y):
    return np.cos(5*x) * np.cos(5*y), np.sin(5*x) * -(np.sin(5*y))

from scipy.optimize import minimize
def f(x):
    return np.sin(5*x[0]) * np.cos(5*x[1]) / 5
def df(x):
    return np.array([np.cos(5*x[0]) * np.cos(5*x[1]), np.sin(5*x[0]) * -np.sin(5*x[1])])
result = minimize(
    f, np.zeros(2), method='trust-constr', jac=df)

print("min_x, min_y: ", result.x)
print("min_z: ", z_function(result.x[0], result.x[1]))

x = np.arange(-1, 1, 0.02)
y = np.arange(-1, 1, 0.02)

X,Y = np.meshgrid(x, y)

Z = z_function(X, Y)
print("z function min: ", np.min(Z))

fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(projection = "3d")
ax.plot_surface(X,Y,Z,cmap = "viridis")
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$f(x,y)$')
ax.view_init(45, 30)

plt.show()

x0 = 0.7
y0 = 0.3
current_position = (x0, y0, z_function(x0, y0))

learning_rate = 0.08

fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(projection = "3d", computed_zorder = False)

for _ in range(100):
    
    X_d, Y_d = z_gradient(current_position[0], current_position[1])
    X_new, Y_new = current_position[0] - learning_rate * X_d, current_position[1] - learning_rate * Y_d
    current_position = (X_new, Y_new, z_function(X_new, Y_new))

    ax.plot_surface(X,Y,Z,cmap = "viridis", zorder = 0)
    ax.scatter(current_position[0], current_position[1], current_position[2], color = 'red', linewidths=5, zorder = 1)
    ax.set_xlabel('$x$')
    ax.set_ylabel('$y$')
    ax.set_zlabel('$f(x,y)$')
    ax.view_init(45, 30)
    plt.pause(0.01)
    ax.clear()

    if _ % 10 == 0:
        print("current position: ", X_new, Y_new, z_function(X_new, Y_new))
    