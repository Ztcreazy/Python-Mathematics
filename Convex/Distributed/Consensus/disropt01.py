"""
disropt is a Python package for distributed optimization over peer-to-peer networks 
of computing units called agents. The main idea of distributed optimization is to 
solve an optimization problem (enjoying a given structure) over a (possibly unstructured) 
network of processors. Each agent can perform local computation and can exchange information
with only its neighbors in the network. A distributed algorithm consists of 
an iterative procedure in which each agent maintains a local estimate of the problem solution 
which is properly updated in order to converge towards the solution.
"""
# min f(x)
#  x
# subject to x -> X
# The optimization problem is assumed to be feasible and has finite optimal cost. 
# Thus, it admits at least an optimal solution that is usually denoted as x*. 
# The optimal solution is a vector that satisfies all the constraints 
# and attains the optimal cost.

# Distributed optimization set-ups
# Cost-coupled set-up
"""In this optimization set-up, the cost function is expressed as the sum of cost functions 
f_i and all of them depend on a common optimization variable x. Formally, the set-up is

min_x sum_from i to N f_i(x)
subject to x -> X
x -> R_d
X ----> R_d

The global constraint set X is common to all agents,
while f_i: from R_d to R is assumed to be known by agent i only,

The goal for distributed algorithms for the cost-coupled set-up is that 
all agent estimates are eventually consensual to an optimal solution x* of the problem
"""
#
"""
Common cost set-ups
In this optimization set-up, there is a unique cost function f that 
depends on a common optimization variable x, 
and the optimization variable must further satisfy local constraints. 
Formally, the set-up is

min_x f(x)
subject to x -> from i = 1 to N Intersection X_i    # Union
x -> R_d
X_i ----> R_d

The cost function f is assumed to be known by all the agents,
while each set 
X_i is assumed to be known by agent i only,

The goal for distributed algorithms for the common-cost set-up is that 
all agent estimates are eventually consensual to an optimal solution 
x* of the problem.
"""
#
"""
Constraint-coupled set-up
In this optimization set-up, the cost function is expressed as the sum of local cost functions 
f_i that depend on a local optimization variable x_i. 
The variables must satisfy local constraints (involving only each optimization variable x_i) 
and global coupling constraints (involving all the optimization variables). 
Formally, the set-up is

min_(x_1, x_2,...,x_N) sum from i = 1 to N: f_i(x_i)
subject to x_i -> X_i
sum from i = 1 to N: g_i(x_i) <= 0

Here the symbol <= is also used to denote component-wise inequality for vectors.
Therefore, the optimization variable consists of the stack of all x_i, namely the vector 
(x_1, x_2, ..., x_N). 
All the quantities with the index i are assumed to be known by agent i only, 
the function g_i, with values in R_S, is used to express the i-th contribution to 
S coupling constraints among all the variables

The goal for distributed algorithms for the constraint-coupled set-up is that 
each agent estimate is asymptotically equal to its portion x_i* -> X_i of an optimal solution 
(x_1*, x_2*, ..., x_N*) of the problem.
"""