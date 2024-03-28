# https://realpython.com/gradient-descent-algorithm-python/#basic-gradient-descent-algorithm
"""
Stochastic gradient descent is an optimization algorithm 
often used in machine learning applications to find the model parameters 
that correspond to the best fit between predicted and actual outputs. 
It's an inexact but powerful technique.
"""
"""
Cost Function: The goal of Optimization
The cost function, or loss function, is the function to be minimized (or maximized) 
by varying the decision variables. Many machine learning methods solve optimization problems 
under the surface. They tend to minimize the difference between actual 
and predicted outputs by adjusting the model parameters 
(like weights and biases for neural networks, decision rules for random forest 
or gradient boosting, and so on).
"""
"""
Gradient of a Function: Calculus Refresher
In calculus, the derivative of a function shows you how much a value changes 
when you modify its argument (or arguments). Derivatives are important for optimization 
because the zero derivatives might indicate a minimum, maximum, or saddle point.
The gradient of a function C of several independent variables 𝑣₁, …, 𝑣ᵣ 
is denoted with ∇𝐶(𝑣₁, …, 𝑣ᵣ) and defined as the vector function of the partial derivatives 
of C with respect to each independent variable: ∇𝐶 = (∂𝐶/∂𝑣₁, …, ∂𝐶/𝑣ᵣ). 
The symbol ∇ is called nabla.
"""
# Small learning rates can result in very slow convergence. 
# If the number of iterations is limited, then the algorithm may return 
# before the minimum is found. Otherwise, 
# the whole process might take an unacceptably large amount of time.
"""
A lower learning rate prevents the vector from making large jumps, and in this case, 
the vector remains closer to the global optimum.
Adjusting the learning rate is tricky. You can't know the best value in advance. 
There are many techniques and heuristics that try to help with this. 
In addition, machine learning practitioners often tune the learning rate 
during model selection and evaluation.
Besides the learning rate, the starting point can affect the solution significantly, 
especially with nonconvex functions.
"""
import numpy as np

def gradient_descent(
    gradient, start, learn_rate, n_iter=50, tolerance=1e-06
):
    vector = start
    for _ in range(n_iter):
        diff = -learn_rate * gradient(vector)
        if np.all(np.abs(diff) <= tolerance):
            print(f"n_iter: {_} ----> diff: {diff} <= {tolerance}")
            break
        vector += diff
    return vector

def gradient_function():
    gradient=lambda v: np.array([2 * v[0], 4 * v[1]**3])
    return gradient

def ssr_gradient(x, y, b):
    res = b[0] + b[1] * x - y
    return res.mean(), (res * x).mean()  # .mean() is a method of np.ndarray

print(gradient_descent(
    gradient = gradient_function(), 
    start=np.array([1.0, 1.0]), 
    learn_rate=0.2
    ))