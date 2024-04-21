import numpy as np
from scipy.optimize import minimize

def f1(x):
    return (x[0]-2)**2 + (x[1]-1)**2

def f2(x):
    return (x[2]-1)**2 + (x[3]-2)**2

def objective_function(x):
    return f1(x) + f2(x)

def callback_function(xk):
    print("Current x:", xk)
    print("Current objective function value:", objective_function(xk))

x0 = np.array([4.0, 5.0, 6.0, 7.0])

# optimization_result = minimize(objective_function, x0, method='SLSQP')
# optimization_result = minimize(objective_function, x0, method='SLSQP', options={'disp': True})
optimization_result = minimize(objective_function, x0, method='SLSQP', callback=callback_function)

x_optimized = optimization_result.x
minimum_value = optimization_result.fun

print("Optimization result: ")
print("Message: ", optimization_result.message)
print("Number of iterations: ", optimization_result.nit)
print("Optimized solution: ")
print("x: ", x_optimized)
print("Minimum value of the objective function: ", minimum_value)
