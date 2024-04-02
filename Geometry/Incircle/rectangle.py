import matplotlib.pyplot as plt
import numpy as np

def inscribed_circle(rectangle):

    incenter = np.mean(rectangle, axis=0)

    width = np.linalg.norm(rectangle[1] - rectangle[0])
    height = np.linalg.norm(rectangle[2] - rectangle[1])
    inradius = min(width, height) / 2
    
    return incenter, inradius

def plot_rectangle_with_incircle(rectangle):

    plt.axis('equal')

    plt.plot([rectangle[0][0], rectangle[1][0], rectangle[2][0], rectangle[3][0], rectangle[0][0]], 
             [rectangle[0][1], rectangle[1][1], rectangle[2][1], rectangle[3][1], rectangle[0][1]], 'bo-')

    incenter, inradius = inscribed_circle(rectangle)
    circle = plt.Circle(incenter, inradius, color='r', fill=False)
    plt.gca().add_patch(circle)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Inscribed Circle of Rectangle')
    plt.grid(True)
    plt.show()


rectangle = np.array([[0, 0], [6, 0], [6, 4], [0, 4]])

plot_rectangle_with_incircle(rectangle)