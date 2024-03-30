# %%
import numpy as np

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

print(result.x) 

import matplotlib.pyplot as plt

# -10 ----> 10
X,Y = np.meshgrid(np.linspace(-15,15,100), np.linspace(-15,15,100))
Z = f(np.array([X, Y]))

min_x0, min_x1 = np.meshgrid(result.x[0], result.x[1])   
min_z = f(np.stack([min_x0, min_x1]))

fig = plt.figure(figsize=(12, 6))
# plt.show()

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.contour3D(X, Y, Z, 60, cmap='viridis')
ax.scatter(min_x0, min_x1, min_z, marker='o', color='red', linewidth=5)
ax.set_xlabel('$x_{0}$')
ax.set_ylabel('$x_{1}$')
ax.set_zlabel('$f(x)$')
ax.view_init(45, 30)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.contour3D(X, Y, Z, 60, cmap='viridis')
ax.scatter(min_x0, min_x1, min_z, marker='o', color='red', linewidth=5)
ax.set_xlabel('$x_{0}$')
ax.set_ylabel('$x_{1}$')
ax.set_zlabel('$f(x)$')
ax.axes.zaxis.set_ticklabels([])
ax.view_init(90, -90)

plt.show()

# %%
"""
x_k+1 = x_k + alpha_k * p_k
x_k+1 = x_k - alpha_k * delta_f(x_k) gradient, derivative

alpha: g(alpha) 
min_alpha ----> g(alpha) = f(x_k + alpha_k * p_k)
min_x ----> f(x)

Wolfe conditions: https://en.wikipedia.org/wiki/Wolfe_conditions
(1) f(x_k + alpha_k * p_k) <= f(x_k) + c * alpha * delta_f(x_k).T * p_k
(2) -p_k.T * delta_f(x_k + alpha_k * p_k) <= -c2 * p_k.T * delta_f(x_k)
0 < c1 < c2 < 1
c1: 10^-4
c2 = 0.9 (Newton or quasi-Newton methods) 
c2 = 0.1 (nonlinear conjugate gradient method)
"""

def steepest_descent(gradient, x0 = np.zeros(2), alpha = 0.001, max_iter = 10000, tolerance = 1e-6):
    '''
    Steepest descent with constant step size alpha.
    
    Args:
      - gradient: gradient of the objective function
      - alpha: line search parameter (default: 0.01)
      - x0: initial guess for x_0 and x_1 (default values: zero) <numpy.ndarray>
      - max_iter: maximum number of iterations (default: 10000)
      - tolerance: minimum gradient magnitude at which the algorithm stops (default: 1e-10)
    
    Out:
      - results: <numpy.ndarray> of size (n_iter, 2) with x_0 and x_1 values at each iteration
      - number of steps: <int>
    '''

    results = np.array([])

    gradient_x = gradient(x0)

    steps_count = 0

    x = x0 
    results = np.append(results, x, axis=0)

    while any(abs(gradient_x) > tolerance) and steps_count < max_iter:
        x = x - alpha * gradient_x
        results = np.append(results, x, axis=0)
        gradient_x = gradient(x)
        steps_count += 1
        
    return results.reshape(-1, 2), steps_count

points, iters = steepest_descent(
  df, x0 = np.array([-9, -9]), alpha=0.30)

minimizer = points[-1].round(1)

print('- Final results: {}'.format(minimizer))
print('- N° steps: {}'.format(iters))

X_estimate, Y_estimate = points[:, 0], points[:, 1] 
Z_estimate = f(np.array([X_estimate, Y_estimate]))

fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.contour3D(X, Y, Z, 60, cmap='viridis')
ax.plot(X_estimate, Y_estimate, Z_estimate, color='red', linewidth=3)
ax.scatter(min_x0, min_x1, min_z, marker='o', color='red', linewidth=5)
ax.set_xlabel('$x_{0}$')
ax.set_ylabel('$x_{1}$')
ax.set_zlabel('$f(x)$')
ax.view_init(45, 30)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.contour3D(X, Y, Z, 60, cmap='viridis')
ax.plot(X_estimate, Y_estimate, Z_estimate, color='red', linewidth=3)
ax.scatter(min_x0, min_x1, min_z, marker='o', color='red', linewidth=5)
ax.set_xlabel('$x_{0}$')
ax.set_ylabel('$x_{1}$')
ax.set_zlabel('$f(x)$')
ax.axes.zaxis.set_ticklabels([])
ax.view_init(90, -90)

plt.show()

alphas = [0.01, 0.25, 0.3, 0.35, 0.4, 1.2]

X_estimates, Y_estimates, Z_estimates = [], [], []

fig, ax = plt.subplots(len(alphas), figsize=(8, 6))
fig.suptitle('$f(x)$ at each iteration for different $α$')

for i, alpha in enumerate(alphas):

    estimate, iters = steepest_descent(
      df, x0 = np.array([-5, -5]), alpha=alpha, max_iter=3000)
    
    print('Input alpha: {}'.format(alpha))
    print('\t- Final results: {}'.format(estimate[-1].round(1)))
    print('\t- N° steps: {}'.format(iters))

    X_estimates.append(estimate[:, 0])
    Y_estimates.append(estimate[:, 1])  
    Z_estimates.append(f(np.array([estimate[:, 0], estimate[:, 1]])))

    ax[i].plot([f(var) for var in estimate], label='alpha: '+str(alpha))
    ax[i].axhline(y=0, color='r', alpha=0.7, linestyle='dashed')
    ax[i].set_xlabel('Number of iterations')
    ax[i].set_ylabel('$f(x)$')
    ax[i].set_ylim([-10, 200])
    ax[i].legend(loc='upper right')

fig = plt.figure(figsize=(12, 60))

for i in range(0, len(alphas)):

    ax = fig.add_subplot(len(alphas), 2, (i*2)+1, projection='3d')
    ax.contour3D(X, Y, Z, 60, cmap='viridis')
    ax.plot(X_estimates[i], Y_estimates[i], Z_estimates[i], color='red', label='alpha: '+str(alphas[i]) , linewidth=3)
    ax.scatter(min_x0, min_x1, min_z, marker='o', color='red', linewidth=5)
    ax.set_xlabel('$x_{0}$')
    ax.set_ylabel('$x_{1}$')
    ax.set_zlabel('$f(x)$')
    ax.view_init(45, 30)
    plt.legend(prop={'size': 15})

    ax = fig.add_subplot(len(alphas), 2, (i*2)+2, projection='3d')
    ax.contour3D(X, Y, Z, 60, cmap='viridis')
    ax.plot(X_estimates[i], Y_estimates[i], Z_estimates[i], color='red', label='alpha: '+str(alphas[i]) , linewidth=3)
    ax.scatter(min_x0, min_x1, min_z, marker='o', color='red', linewidth=10)
    ax.set_xlabel('$x_{0}$')
    ax.set_ylabel('$x_{1}$')
    ax.set_zlabel('$f(x)$')
    ax.axes.zaxis.set_ticklabels([])
    ax.view_init(90, -90)
    plt.legend(prop={'size': 15})

plt.show()

# %%
def line_search(step, x, gradient_x, c = 1e-4, tol = 1e-8):
    '''
    Inexact line search where the step length is updated through the Armijo condition:
    $ f (x_k + α * p_k ) ≤ f ( x_k ) + c * α * ∇ f_k^T * p_k $

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
    
    while f(x - step * gradient_x) >= (f_x - c * step * gradient_square_norm):
        
        step /= 2
        
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
    
    results = np.array([])
    
    gradient_x = gradient(x0)
    
    steps_count = 0
    
    x = x0 
    results = np.append(results, x, axis=0)
   
    while any(abs(gradient_x) > tolerance) and steps_count < max_iter:

        alpha = line_search(1.2, x, gradient_x)
        
        x = x - alpha * gradient_x
        
        results = np.append(results, x, axis=0)
        
        gradient_x = gradient(x) 
                
        steps_count += 1 
        
    return results.reshape(-1, 2), steps_count

points, iters = steepest_descent(
  df, x0 = np.array([-9, -9]))

minimizer = points[-1].round(1)

print('- Final results: {}'.format(minimizer))
print('- N° steps: {}'.format(iters))

X_estimate, Y_estimate = points[:, 0], points[:, 1] 
Z_estimate = f(np.array([X_estimate, Y_estimate]))

fig = plt.figure(figsize=(12, 6))

ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.contour3D(X, Y, Z, 60, cmap='viridis')
ax.plot(X_estimate, Y_estimate, Z_estimate, color='red', linewidth=3)
ax.scatter(min_x0, min_x1, min_z, marker='o', color='red', linewidth=5)
ax.set_xlabel('$x_{0}$')
ax.set_ylabel('$x_{1}$')
ax.set_zlabel('$f(x)$')
ax.view_init(45, 30)

ax = fig.add_subplot(1, 2, 2, projection='3d')
ax.contour3D(X, Y, Z, 60, cmap='viridis')
ax.plot(X_estimate, Y_estimate, Z_estimate, color='red', linewidth=3)
ax.scatter(min_x0, min_x1, min_z, marker='o', color='red', linewidth=5)
ax.set_xlabel('$x_{0}$')
ax.set_ylabel('$x_{1}$')
ax.set_zlabel('$f(x)$')
ax.axes.zaxis.set_ticklabels([])
ax.view_init(90, -90)

plt.show()
