import numpy as np
from scipy.integrate import odeint
from scipy.linalg import eig
import matplotlib.pyplot as plt

def model(X, t):

    x1, x2, x3 = X
    L3 = np.array([[1, -1, 0],
                  [0, 1.5, -1.5],
                  [-2, 0, 2]])
    
    # D, W = np.linalg.eig(L3)
    D, W_left = eig(L3, left=True, right=False)
    # print("W left: ", W_left)

    w = W_left[:, 0]
    w /= np.sum(w)

    L3_balanced = np.diag(w) @ L3

    dXdt1 = -np.dot(L3, X)
    dXdt2 = -np.dot(L3_balanced, X)

    # return dXdt1
    return dXdt2.astype(np.float64)

X0 = [0.2, 0.4, 0.6]

t = np.linspace(0, 10, 100)

X = odeint(model, X0, t)

plt.plot(t, X[:, 0], label='x1Sol', linewidth=2)
plt.plot(t, X[:, 1], label='x2Sol', linewidth=2)
plt.plot(t, X[:, 2], label='x3Sol', linewidth=2)
plt.xlabel('Time')
plt.ylabel('Sol')
plt.grid(True)
plt.legend()
plt.show()
