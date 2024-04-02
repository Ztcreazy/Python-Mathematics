import matplotlib.pyplot as plt
import numpy as np

def circumcircle(rectangle):

    center = np.mean(rectangle, axis=0)

    diagonal_length = np.linalg.norm(rectangle[2] - rectangle[0])
    radius = diagonal_length / 2
    
    return center, radius

def plot_rectangle_with_circumcircle(rectangle):

    plt.axis('equal')

    plt.plot([rectangle[0][0], rectangle[1][0], rectangle[2][0], rectangle[3][0], rectangle[0][0]], 
             [rectangle[0][1], rectangle[1][1], rectangle[2][1], rectangle[3][1], rectangle[0][1]], 'bo-')

    circumcircle_center, circumcircle_radius = circumcircle(rectangle)
    circle = plt.Circle(circumcircle_center, circumcircle_radius, color='r', fill=False)
    plt.gca().add_patch(circle)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Circumcircle of Rectangle')
    plt.grid(True)
    plt.show()

# np.array([[0, 0], [6, 0], [6, 4], [0, 4]])
# verticle
rectangle = np.array([[0, 0], [10, 1], [9, 11], [-1, 10]])

plot_rectangle_with_circumcircle(rectangle)