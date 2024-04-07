from math import *
import matplotlib.pyplot as plt

import numpy as np

# Gaussian
def f(mu, sigma2, x):
    """
    f takes in a mean and squared variance, and an input x and returns the gaussian value.
    """
    coefficient = 1.0 / sqrt(2.0 * sigma2)
    exponential = exp(-0.5 * (x-mu) **2 / sigma2)
    return coefficient * exponential

def update(mean1, var1, mean2, var2):
    """
    This function takes in two means and two squared variance terms, 
    and returns updated Gaussian parameters.
    """
    # Calculate the new parameters
    # mu' = (r **2 * u + sigma **2 * v) / (r **2 + sigma **2)
    # sigma **2' = 1 / ( 1 / r **2 + 1 / sigma **2)
    new_mean = (var2 * mean1 + var1 * mean2) / (var2 + var1)
    new_var = 1 / (1 / var2 + 1 / var1)
    return [new_mean, new_var]

# the motion update/predict function
def predict(mean1, var1, mean2, var2):
    ''' This function takes in two means and two squared variance terms,
        and returns updated gaussian parameters, after motion.'''
    # Calculate the new parameters
    new_mean = mean1 + mean2
    new_var = var1 + var2
    
    return [new_mean, new_var]

print("update: ", update(20, 9, 30, 3))

# measurements for mu and motions, U
measurements = [5., 6., 7., 9., 10.]
motions = [1., 1., 2., 1., 1.]

# initial parameters
measurement_sig = 4.
motion_sig = 2.
mu = 0.
sig = 10000.

## TODO: Loop through all measurements/motions
# this code assumes measurements and motions have the same length
# so their updates can be performed in pairs
for n in range(len(measurements)):
    # measurement update, with uncertainty
    mu, sig = update(mu, sig, measurements[n], measurement_sig)
    print('Update: [{}, {}]'.format(mu, sig))
    # motion update, with uncertainty
    mu, sig = predict(mu, sig, motions[n], motion_sig)
    print('Predict: [{}, {}]'.format(mu, sig))

# print the final, resultant mu, sig
print('\n')
print('Final result: [{}, {}]'.format(mu, sig))

## Print out and display the final, resulting Gaussian 
# set the parameters equal to the output of the Kalman filter result
mu = mu
sigma2 = sig

# define a range of x values
x_axis = np.arange(-20, 20, 0.1)

# create a corresponding list of gaussian values
g = []
for x in x_axis:
    g.append(f(mu, sigma2, x))

# plot the result 
plt.plot(x_axis, g)
plt.grid(True)
plt.show()

# display the *initial* gaussian over a range of x values
# define the parameters
mu = 0
sigma2 = 10000

# define a range of x values
x_axis = np.arange(-20, 20, 0.1)

# create a corresponding list of gaussian values
g = []
for x in x_axis:
    g.append(f(mu, sigma2, x))

# plot the result 
plt.plot(x_axis, g)
plt.grid(True)
plt.show()
