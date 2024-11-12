import numpy as np

def Cost(u):
    w = np.eye(6)  # Define weight matrix (identity matrix for simplicity)
    cost = np.dot(u.T, np.dot(w, u))  # Compute cost as u^T * w * u
    return cost