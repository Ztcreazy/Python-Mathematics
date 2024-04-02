import matplotlib.pyplot as plt
import numpy as np

def circumcircle(triangle):

    A, B, C = triangle
    a = np.linalg.norm(B - C)
    b = np.linalg.norm(A - C)
    c = np.linalg.norm(A - B)
    
    cos_A = (b**2 + c**2 - a**2) / (2 * b * c)
    cos_B = (a**2 + c**2 - b**2) / (2 * a * c)
    cos_C = (a**2 + b**2 - c**2) / (2 * a * b)
    
    circumcenter = (A * np.sin(2 * np.arccos(cos_A)) + B * np.sin(2 * np.arccos(cos_B)) + C * np.sin(2 * np.arccos(cos_C))) / (np.sin(2 * np.arccos(cos_A)) + np.sin(2 * np.arccos(cos_B)) + np.sin(2 * np.arccos(cos_C)))
    
    circumradius = np.linalg.norm(circumcenter - A)
    
    return circumcenter, circumradius

def plot_triangle(triangle):

    plt.axis('equal')
    
    plt.plot([triangle[0][0], triangle[1][0], triangle[2][0], triangle[0][0]], 
             [triangle[0][1], triangle[1][1], triangle[2][1], triangle[0][1]], 'bo-')
    
    circumcenter, circumradius = circumcircle(triangle)
    circle = plt.Circle(circumcenter, circumradius, color='r', fill=False)
    plt.gca().add_patch(circle)
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Circumcircle of Triangle')
    plt.grid(True)
    plt.show()


triangle = np.array([[0, 0], [4, 2], [2, 6]])

plot_triangle(triangle)