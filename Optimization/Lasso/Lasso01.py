"""
The Lasso is a linear model that estimates sparse coefficients. 
It is useful in some contexts due to its tendency to prefer solutions 
with fewer non-zero coefficients, effectively reducing the number of features 
upon which the given solution is dependent. For this reason, 
Lasso and its variants are fundamental to the field of compressed sensing. 
Under certain conditions, it can recover the exact set of non-zero coefficients.
"""
"""
Mathematically, it consists of a linear model with an added regularization term. 
The objective function to minimize is:
min_w 1 / (2 * n_samples) * ||X*w - y||_2 **2 + alpha * ||w||_1
"""
"""
The lasso estimate thus solves the minimization of the least-squares penalty 
with alpha * ||w||_1 added, where alphais a constant 
and ||w||_1 is the l1-norm of the coefficient vector.
"""

import numpy as np

rng = np.random.RandomState(0)
n_samples, n_features, n_informative = 50, 100, 10
time_step = np.linspace(-2, 2, n_samples)
freqs = 2 * np.pi * np.sort(rng.rand(n_features)) / 0.01
# print(freqs)
X = np.zeros((n_samples, n_features))

for i in range(n_features):
    X[:, i] = np.sin(freqs[i] * time_step)

idx = np.arange(n_features)
true_coef = (-1) ** idx * np.exp(-idx / 10)
true_coef[n_informative:] = 0  # sparsify coef
y = np.dot(X, true_coef)

for i in range(n_features):
    X[:, i] = np.sin(freqs[i] * time_step + 2 * (rng.random_sample() - 0.5))
    X[:, i] += 0.2 * rng.normal(0, 1, n_samples)

y += 0.2 * rng.normal(0, 1, n_samples)

import matplotlib.pyplot as plt

plt.plot(time_step, y)
plt.ylabel("target signal")
plt.xlabel("time")
plt.grid(True)
_ = plt.title("Superposition of sinusoidal signals")

plt.show()