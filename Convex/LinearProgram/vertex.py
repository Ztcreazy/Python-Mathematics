# c_1*x_1 + c_2*x_2 + c_3*x_3 + ...... + c_n*x_n
# a_11*x_1 + a_12*x_2 + a_13*x_3 + ...... <= b1
# a_21*x_1 + a_22*x_2 + a_23*x_3 + ...... <= b2
# ......
# a_m1*x_1 + a_m2*x_2 + a_m3*x_3 + ...... <= bm

# example1
# maximize 4*x_1 + 6*x_2
# -x_1 + x_2 <= 11    x_1 + x_2 <= 27
# 2*x_1 + 5*x_2 <= 90    x_1, x_2>=0
# slack variable -x_1 + x_2 + s_1 = 11 ......
# x_1 + x_2 + s_2 = 27    2*x_1 + 5*x_2 + s_3 = 90
# 
#
#
#      4      6      0      0      0
# 1    x_1    x_2    s_1    s_2    s_3    RHS    
# 2    -1     1      1      0      0      11   
# 3    1      1      0      1      0      27
# 4    2      5      0      0      1      90
# 5    -4     -6     0      0      0      0 



# x_1 = x_2 = 0    s_1 = 11    s_2 = 27    s_3 = 90
# entering: -4, -6 ----> -6 ----> x_2 ----> 11/1=11 27/1=27 90/5=18 ----> 11: exiting
# Gaussian Elimination
#      4      6      0      0      0
# 1    x_1    x_2    s_1    s_2    s_3    RHS
#    
# 2    -1     1      1      0      0      11   
# 3    2      0      -1     1      0      16    RHS3 - RHS2
# 4    7      0      -5     0      1      35    RHS4 - 5*RHS2
# 5    -10    0      6      0      0      66     RHS5 + 6*RHS2
#
#
#
# -10 ----> ...... x_1: a x_2: b x_3: c
import numpy as np
from scipy.optimize import linprog

obj = [-4, -6]
lhs = [[-1,1], [1,1], [2,5]]
rhs = [11, 27, 90]

#lhs_eq = [[x1,x2],[y1,y2]]
#rhs_eq = [[a1,a2]]

bnd = [(0,float('inf')),(0,float('inf'))]

#optimization = linprog(c = obj,
#                       A_ub = lhs,
#                       b_ub = rhs,
#                       bounds = bnd,
#                       A_eq = lhs_eq,
#                       b_eq = rhs_eq
#                       method = 'simplex')

optimization = linprog(c = obj,
                       A_ub = lhs,
                       b_ub = rhs,
                       bounds = bnd,
                       method = 'highs') # 'simplex'

print("optimization x: ", optimization.x)
print("optimization function: ", optimization.fun)
print("optimization status: ", optimization.status)



obj = [-5, 3, 4, -7]
lhs = [[1,1,1,1],
       [1,0,1,0],
       [2,1,1,0]]
rhs = [14, 7, 13]
bnd = [(0,float('inf')),(0,float('inf')),(0,float('inf')),(0,float('inf'))] #There are 4 bounds because 4 variables

optimize = linprog(c = obj,
                   A_ub = lhs,
                   b_ub = rhs,
                   bounds = bnd,
                   method = 'highs')

print("optimization x: ", optimization.x)
print("optimization function: ", optimization.fun)
print("optimization status: ", optimization.status)