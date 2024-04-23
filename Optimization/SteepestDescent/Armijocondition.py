import numpy as np
import matplotlib.pyplot as plt

def f(x):
    """
    objective function
    """
    return 0.5*(x[0] - 4.5)**2 + 2.5*(x[1] - 2.3)**2

def df(x):
    """
    derivative
    """
    return np.array([x[0] - 4.5, 5*(x[1] - 2.3)])

from scipy.optimize import minimize

result = minimize(
    f, np.zeros(2), method='trust-constr', jac=df)

result.x

# Prepare the objective function between -10 and 10
X, Y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
Z = f(np.array([X, Y]))

# Minimizer
min_x0, min_x1 = np.meshgrid(result.x[0], result.x[1])   
min_z = f(np.stack([min_x0, min_x1]))


# Plot
fig = plt.figure(figsize=(12, 6))

# First subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.contour3D(X, Y, Z, 60, cmap='viridis')
ax.scatter(min_x0, min_x1, min_z, marker='o', color='red', linewidth=10)
ax.set_xlabel('$x_{0}$')
ax.set_ylabel('$x_{1}$')
ax.set_zlabel('$f(x)$')
ax.view_init(40, 20)

# Second subplot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.contour3D(X, Y, Z, 60, cmap='viridis')
ax.scatter(min_x0, min_x1, min_z, marker='o', color='red', linewidth=10)
ax.set_xlabel('$x_{0}$')
ax.set_ylabel('$x_{1}$')
ax.set_zlabel('$f(x)$')
ax.axes.zaxis.set_ticklabels([])
ax.view_init(90, -90)

plt.show()

"""
x_k+1 = x_k + alpha_k * p_k
x_k+1 = x_k - alpha_k * delta_f(x_k) gradient, derivative

alpha: g(alpha) 
min_alpha ----> g(alpha) = f(x_k + alpha_k * p_k)
min_x ----> f(x)

Wolfe conditions: https://en.wikipedia.org/wiki/Wolfe_conditions
(1) f(x_k + alpha_k * p_k) <= f(x_k) + c1 * alpha * delta_f(x_k).T * p_k
(2) -p_k.T * delta_f(x_k + alpha_k * p_k) <= -c2 * p_k.T * delta_f(x_k)
0 < c1 < c2 < 1
c1: 10^-4
c2 = 0.9 (Newton or quasi-Newton methods) 
c2 = 0.1 (nonlinear conjugate gradient method)
"""



def line_search(step, x, gradient_x, c = 1e-4, tol = 1e-8):
    '''
    Inexact line search where the step length is updated through the Armijo condition:
    $ f (x_k + alpha * p_k ) ≤ f ( x_k ) + c * alpha * ∇ f_k^T * p_k $

    Args:
      - step: starting alpha value
      - x: current point
      - gradient_x: gradient of the current point
      - c: constant value (default: 1e-4)
      - tol: tolerance value (default: 1e-6)
    Out:
      - New value of step: the first value found respecting the Armijo condition
    '''
    f_x = f(x)
    gradient_square_norm = np.linalg.norm(gradient_x)**2
    
    # Until the sufficient decrease condition is met 
    while f(x - step * gradient_x) >= (f_x - c * step * gradient_square_norm):
        
        # Update the stepsize (backtracking)
        step /= 2
        
        # If the step size falls below a certain tolerance, exit the loop
        if step < tol:
            break
    
    return step

def steepest_descent(gradient, x0 = np.zeros(2), max_iter = 10000, tolerance = 1e-10): 
    '''
    Steepest descent with alpha updated through line search (Armijo condition).
    
    Args:
      - gradient: gradient of the objective function
      - x0: initial guess for x_0 and x_1 (default values: zero) <numpy.ndarray>
      - max_iter: maximum number of iterations (default: 10000)
      - tolerance: minimum gradient magnitude at which the algorithm stops (default: 1e-10)
    
    Out:
      - results: <numpy.ndarray> with x_0 and x_1 values at each iteration
      - number of steps: <int>
    '''
    
    # Prepare list to store results at each iteration 
    results = np.array([])
    
    # Evaluate the gradient at the starting point 
    gradient_x = gradient(x0)
    
    # Initialize the steps counter 
    steps_count = 0
    
    # Set the initial point 
    x = x0 
    results = np.append(results, x, axis=0)
   
    # Iterate until the gradient is below the tolerance or maximum number of iterations is reached
    # Stopping criterion: inf norm of the gradient (max abs)
    while any(abs(gradient_x) > tolerance) and steps_count < max_iter:

        # Update the step size through the Armijo condition
        # Note: the first value of alpha is commonly set to 1
        alpha = line_search(1, x, gradient_x)
        
        # Update the current point by moving in the direction of the negative gradient 
        x = x - alpha * gradient_x
        
        # Store the result
        results = np.append(results, x, axis=0)
        
        # Evaluate the gradient at the new point 
        gradient_x = gradient(x) 
                
        # Increment the iteration counter 
        steps_count += 1 
        
    # Return the steps taken and the number of steps
    return results.reshape(-1, 2), steps_count



# Steepest descent
points, iters = steepest_descent(
  df, x0 = np.array([-9, -9]))

# Found minimizer
minimizer = points[-1].round(1)

# Print results
print('- Final results: {}'.format(minimizer))
print('- N° steps: {}'.format(iters))

# Steepest descent steps
X_estimate, Y_estimate = points[:, 0], points[:, 1] 
Z_estimate = f(np.array([X_estimate, Y_estimate]))

# Plot
fig = plt.figure(figsize=(12, 6))

# First subplot
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.contour3D(X, Y, Z, 60, cmap='viridis')
ax.plot(X_estimate, Y_estimate, Z_estimate, color='red', linewidth=3)
ax.scatter(min_x0, min_x1, min_z, marker='o', color='red', linewidth=10)
ax.set_xlabel('$x_{0}$')
ax.set_ylabel('$x_{1}$')
ax.set_zlabel('$f(x)$')
ax.view_init(20, 20)

# Second subplot
ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.contour3D(X, Y, Z, 60, cmap='viridis')
ax.plot(X_estimate, Y_estimate, Z_estimate, color='red', linewidth=3)
ax.scatter(min_x0, min_x1, min_z, marker='o', color='red', linewidth=10)
ax.set_xlabel('$x_{0}$')
ax.set_ylabel('$x_{1}$')
ax.set_zlabel('$f(x)$')
ax.axes.zaxis.set_ticklabels([])
ax.view_init(90, -90)

plt.show()