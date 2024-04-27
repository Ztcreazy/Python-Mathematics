import numpy as np
# https://realpython.com/numpy-random-normal/

rng = np.random.default_rng()

print("rng: ", rng)
print("rng normal: ", rng.normal())

numbers = rng.normal(size=10_000)
print("numbers mean: ", numbers.mean())
print("numbers std: ", numbers.std())

# multidimensional arrays of numbers
print("rng normal (2, 4)", rng.normal(size=(2, 4)))

import matplotlib.pyplot as plt
plt.hist(numbers) # histogram
plt.grid(True)
plt.show()

plt.hist(numbers, bins=100)
plt.grid(True)
plt.show()

bins = 100
bin_width = (numbers.max() - numbers.min()) / bins
hist_area = len(numbers) * bin_width
print("histogram area: ", hist_area)

# The SciPy library contains several functions named pdf(). 
# These are probability density functions. You can use these 
# to plot theoretical probability distributions:
import scipy.stats
x = np.linspace(-4, 4, 101)
plt.plot(x, scipy.stats.norm.pdf(x))
plt.grid(True)
plt.show()


x = np.linspace(numbers.min(), numbers.max(), 101)
plt.hist(numbers, bins=100)
plt.plot(x, scipy.stats.norm.pdf(x) * hist_area)
plt.grid(True)
plt.show()

# !!!!
weights = rng.normal(1023, 19, size=5_000) # mu = 1023 sigma = 19 
print("weights [:3]: ", weights[:3])
print("weights mean: ", weights.mean())
print("weights std: ", weights.std())
