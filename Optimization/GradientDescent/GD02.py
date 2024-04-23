import numpy as np
import matplotlib.pyplot as plt

def y_function(x):
    return np.tanh(x)
    # np.tan(x) 
    # np.sin(x) 
    # x**2

def y_derivative(x):
    return 1 - (np.tanh(x))**2
    # (1/(np.cos(x)))**2 
    # np.cos(x) 
    # 2*x

x0 = 1.2
initial_position = [x0, y_function(x0)]

x = np.arange(-5, 5, .01)
y = y_function(x)

# plt.plot(x,y)
# plt.scatter(initial_position[0], initial_position[1], color = 'red', linewidths=3)
# plt.grid(True)
# plt.show()

learning_rate = 0.08
tolerance = 1e-3

for _ in range(2000):
    new_x = initial_position[0] - learning_rate * y_derivative(initial_position[0])
    error = abs(new_x - initial_position[0])
    new_y = y_function(new_x)
    initial_position = [new_x, new_y]

    if error <= tolerance: # if new_x <= tolerance:
        print("new x: ", new_x)
        print("iteration number: ", _, "error: ", error)
        break

    plt.plot(x,y)
    plt.scatter(initial_position[0], initial_position[1], color = 'red', linewidths=3)
    plt.grid(True)
    plt.pause(0.01)
    plt.clf()