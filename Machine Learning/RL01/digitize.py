import numpy as np

x = np.array([0.2, 6.4, 3.0, 1.6])

bins = np.array([0.0, 1.0, 2.5, 4.0, 10.0])

inds = np.digitize(x, bins)

print(inds)