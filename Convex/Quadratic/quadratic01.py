# Qiskit optimization introduces the QuadraticProgram class to make a model of an optimization problem.
# https://qiskit-community.github.io/qiskit-optimization/tutorials/01_quadratic_program.html
"""
minimize      x.T *Q0 *x + c.T *x
subject to    A *c <= b
              x.T *Qi *x + ai.T *x <= ri,   i = 1, ..., i, ..., q
              li <= xi <= ui    1, ..., i, ..., n  

              Qi:   n *n matrices
              A:    m *n matrix
              x, c: n-dimensional vectors
              b:    m-dimensional vector
              x:    binary, integer, or continuous variable  
"""
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.translators import from_docplex_mp



#### Loading a QuadraticProgram from a docplex model
from docplex.mp.model import Model

mdl = Model("docplex model")
x = mdl.binary_var("x")
y = mdl.integer_var(lb=-1, ub=5, name="y")
mdl.minimize(x + 2 * y)
mdl.add_constraint(x - y == 3)
mdl.add_constraint((x + y) * (x - y) <= 1)
print(mdl.export_as_lp_string())

## load from a Docplex model
mod = from_docplex_mp(mdl)
print(type(mod))
print()
print(mod.prettyprint())



#### Directly constructing a QuadraticProgram
# make an empty problem
mod = QuadraticProgram( "my problem" )
print(mod.prettyprint())

# Add variables
mod.binary_var(name = "x")
mod.integer_var(name = "y", lowerbound=-1, upperbound=5)
mod.continuous_var(name = "z", lowerbound=-1, upperbound=5)
print(mod.prettyprint())

"""
You can set the objective function by invoking QuadraticProgram.minimize 
or QuadraticProgram.maximize. You can add a constant term as well as 
linear and quadratic objective function by specifying linear and quadratic terms 
with either list, matrix or dictionary.
"""
# Add objective function using dictionaries
mod.minimize(constant=3, linear={"x": 1}, quadratic={("x","y"): 2, ("z","z"): -1})
print(mod.prettyprint())

"""
Another way to specify the quadratic program is using arrays. For the linear term, 
the array corresponds to the vector c in the mathematical formulation. For the quadratic term, 
the array corresponds to the matrix Q.
"""
# Add objective function using lists/arrays
mod.minimize(constant=3, linear=[1, 0, 0], quadratic=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
print(mod.prettyprint())

print("constant:\t\t\t", mod.objective.constant)
print("linear dict:\t\t\t", mod.objective.linear.to_dict())
print("linear array:\t\t\t", mod.objective.linear.to_array())
print("linear array as sparse matrix:\n", mod.objective.linear.coefficients, "\n")
print("quadratic dict w/ index:\t", mod.objective.quadratic.to_dict())
print("quadratic dict w/ name:\t\t", mod.objective.quadratic.to_dict(use_name=True))
print(
    "symmetric quadratic dict w/ name:\t",
    mod.objective.quadratic.to_dict(use_name=True, symmetric=True),
)
print("quadratic matrix:\n", mod.objective.quadratic.to_array(), "\n")
print("symmetric quadratic matrix:\n", mod.objective.quadratic.to_array(symmetric=True), "\n")
print("quadratic matrix as sparse matrix:\n", mod.objective.quadratic.coefficients)

# Add linear constraints
mod.linear_constraint(linear={"x": 1, "y": 2}, sense="==", rhs=3, name="lin_eq")
mod.linear_constraint(linear={"x": 1, "y": 2}, sense="<=", rhs=3, name="lin_leq")
mod.linear_constraint(linear={"x": 1, "y": 2}, sense=">=", rhs=3, name="lin_geq")
print(mod.prettyprint())

# Add quadratic constraints
mod.quadratic_constraint(
    linear={"x": 1, "y": 1},
    quadratic={("x", "x"): 1, ("y", "z"): -1},
    sense="==",
    rhs=1,
    name="quad_eq",
)
mod.quadratic_constraint(
    linear={"x": 1, "y": 1},
    quadratic={("x", "x"): 1, ("y", "z"): -1},
    sense="<=",
    rhs=1,
    name="quad_leq",
)
mod.quadratic_constraint(
    linear={"x": 1, "y": 1},
    quadratic={("x", "x"): 1, ("y", "z"): -1},
    sense=">=",
    rhs=1,
    name="quad_geq",
)
print(mod.prettyprint())

lin_geq = mod.get_linear_constraint("lin_geq")
print("lin_geq:", lin_geq.linear.to_dict(use_name=True), lin_geq.sense, lin_geq.rhs)
quad_geq = mod.get_quadratic_constraint("quad_geq")
print(
    "quad_geq:",
    quad_geq.linear.to_dict(use_name=True),
    quad_geq.quadratic.to_dict(use_name=True),
    quad_geq.sense,
    lin_geq.rhs,
)

# Remove constraints
mod.remove_linear_constraint("lin_eq")
mod.remove_quadratic_constraint("quad_leq")
print(mod.prettyprint())

#### Substituting Variables
sub = mod.substitute_variables(constants={"x": 0}, variables={"y": ("z", -1)})
print(sub.prettyprint())

# sub = mod.substitute_variables(constants={"x": -1})
# print(sub.status)

from qiskit_optimization import QiskitOptimizationError

try:
    sub = mod.substitute_variables(constants={"x": -1}, variables={"y": ("x", 1)})
except QiskitOptimizationError as e:
    print("Error: {}".format(e))

mod = QuadraticProgram()
mod.binary_var(name="e")
mod.binary_var(name="f")
mod.continuous_var(name="g")
mod.minimize(linear=[1, 2, 3])
print(mod.export_as_lp_string())