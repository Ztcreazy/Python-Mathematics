import numpy as np
# https://towardsdatascience.com/implementing-the-steepest-descent-algorithm-in-python-from-scratch-d32da2906fe2#e046

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



def steepest_descent(gradient, x0 = np.zeros(2), alpha = 0.01, max_iter = 10000, tolerance = 1e-10): 
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
    # print("alpha: ", alpha)
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
        #alpha = line_search(1, x, gradient_x)
        
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



# convergence is not guaranteed for any value of the step size
alphas = [0.01, 0.25, 0.3, 0.35, 0.4] # , 1.2

X_estimates, Y_estimates, Z_estimates = [], [], []

fig, ax = plt.subplots(len(alphas), figsize=(8, 6))
fig.suptitle('$f(x)$ at each iteration for different $alpha$')

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

fig = plt.figure(figsize=(25, 60))

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