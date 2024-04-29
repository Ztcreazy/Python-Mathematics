# https://www.geeksforgeeks.org/min-cost-path-dp-6/
"""
Given a cost matrix cost[][] and a position (M, N) in cost[][], write a function that 
returns cost of minimum cost path to reach (M, N) from (0, 0). Each cell of the matrix 
represents a cost to traverse through that cell. The total cost of a path to reach (M, N) is 
the sum of all the costs on that path (including both source and destination). 
You can only traverse down, right and diagonally lower cells from a given cell, i.e., 
from a given cell (i, j), cells (i+1, j), (i, j+1), and (i+1, j+1) can be traversed.
"""
"""
This problem has the optimal substructure property. The path to reach (m, n) must be through 
one of the 3 cells: (m-1, n-1) or (m-1, n) or (m, n-1). So minimum cost to reach (m, n) can be 
written as “minimum of the 3 cells plus cost[m][n]”.

minCost(m, n) = min (minCost(m-1, n-1), minCost(m-1, n), minCost(m, n-1)) + cost[m][n] 
"""
# A Naive recursive implementation of MCP(Minimum Cost Path) problem
import sys
R = 3
C = 3
 
# Returns cost of minimum cost path from (0,0) to (m, n) in mat[R][C]
 
 
def minCost_recursive(cost, m, n):
    if (n < 0 or m < 0):
        return sys.maxsize
    elif (m == 0 and n == 0):
        return cost[m][n]
    else:
        return cost[m][n] + min(minCost_recursive(cost, m-1, n-1),
                                minCost_recursive(cost, m-1, n),
                                minCost_recursive(cost, m, n-1))
 
# A utility function that returns minimum of 3 integers */
 
 
def min(x, y, z):
    if (x < y):
        return x if (x < z) else z
    else:
        return y if (y < z) else z
  


# Returns cost of minimum cost path
# from (0,0) to (m, n) in mat[R][C]
def min_cost_memoized(cost, m, n, memo):
    if n < 0 or m < 0:
        return float('inf')
    elif m == 0 and n == 0:
        return cost[m][n]
 
    if memo[m][n] != -1:
        return memo[m][n]
 
    memo[m][n] = cost[m][n] + min(
        min_cost_memoized(cost, m - 1, n - 1, memo),
        min_cost_memoized(cost, m - 1, n, memo),
        min_cost_memoized(cost, m, n - 1, memo)
    )
 
    return memo[m][n]
 
# Returns cost of minimum cost path
# from (0,0) to (m, n) in mat[R][C]
def min_cost(cost, m, n):
    memo = [[-1] * C for _ in range(R)]  # Initialize memo table with -1
 
    return min_cost_memoized(cost, m, n, memo)
 
# Driver code
cost = [
    [1, 2, 3],
    [4, 8, 2],
    [1, 5, 3]
]

print("min cost path recursive: ", minCost_recursive(cost, 2, 2))
print("min cost path memoization: ", min_cost(cost, 2, 2))
