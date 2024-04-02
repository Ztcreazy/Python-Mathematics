import matplotlib.pyplot as plt
import numpy as np

def inscribed_circle(triangle):

    A, B, C = triangle
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)
    
    perimeter = a + b + c
    incenter = (a * A + b * B + c * C) / perimeter
    
    s = perimeter / 2
    inradius = np.sqrt((s - a) * (s - b) * (s - c) / s)
    
    return incenter, inradius

def plot_triangle(triangle):

    plt.axis('equal')

    plt.plot([triangle[0][0], triangle[1][0], triangle[2][0], triangle[0][0]], 
             [triangle[0][1], triangle[1][1], triangle[2][1], triangle[0][1]], 'bo-')

    incenter, inradius = inscribed_circle(triangle)
    circle = plt.Circle(incenter, inradius, color='r', fill=False)
    plt.gca().add_patch(circle)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Inscribed Circle of Triangle')
    plt.grid(True)
    plt.show()


triangle = np.array([[0, 0], [4, 0], [4, 6]])

plot_triangle(triangle)