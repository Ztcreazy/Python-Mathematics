from disropt.functions import Variable
n = 2 # dimension of the variable
x = Variable(n)
print("x input shape: ", x.input_shape) # -> (2, 1)
print("x output shape: ", x.output_shape) # -> (2, 1)

import numpy as np
a = 1
A = np.array([[1,2], [2,4]])
b = np.array([[1], [1]])
f = A @ x - b

# or, alternatively
from disropt.functions import AffineForm
f = AffineForm(x, A, b)

from disropt.functions import QuadraticForm
Q = np.random.rand(2,2)
g = QuadraticForm(f, Q) # or: g = f @ (Q.tranpose() @ f)
print("g input shape: ", g.input_shape) # -> (2, 1)
print("g output shape: ", g.output_shape) # -> (1, 1)

# Function properties and methods
print("g is differentiable: ", g.is_differentiable) # -> True
# g.is_affine # -> False
# g.is_quadratic # -> True
# f.is_affine # -> True

# g.output_shape # -> (1,1)
# g.input_shape # -> (2,1)

# pt = np.random.rand(2,1)
# # the value of g computed at pt is obtained as
# g.eval(pt)
# # the value of the jacobian of g computed at pt is
# g.jacobian(pt)
# # the value of a (sub)gradient of g is available only if the output shape of g is (1,1)
# g.subgradient(pt)
# # otherwise it will result in an error
# f.subgradient(pt) # -> Error
# # the value of the hessian of g computed at pt is
# g.hessian(pt)

f = A @ x + b
print("f get parameters: ", f.get_parameters()) # -> A, b



# !!!!
# Defining constraints from functions
constraint = g == 0 # g(x) = 0
constraint = g >= 0 # g(x) >= 0
constraint = g <= 0 # g(x) <= 0

c = np.random.rand(2,1)
constr = f <= c

# Constraints can be evaluated at any point by using the eval method 
# which returns a boolean value if the constraint is satisfied. 
pt = np.random.rand(2,1)
constr.eval(pt) # -> True if f(pt) <= c
constr.function.eval(pt) # -> value of f - c at pt