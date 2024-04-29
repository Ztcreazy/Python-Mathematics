# Dynamic Programming Python implementation of Min Cost Path
# problem
R = 3
C = 3
 
 
def minCost_bottom_up(cost, m, n):
 
    # Instead of following line, we can use int tc[m+1][n+1] or
    # dynamically allocate memoery to save space. The following
    # line is used to keep te program simple and make it working
    # on all compilers.
    tc = [[0 for x in range(C)] for x in range(R)]
 
    tc[0][0] = cost[0][0]
 
    # Initialize first column of total cost(tc) array
    for i in range(1, m+1):
        tc[i][0] = tc[i-1][0] + cost[i][0]
 
    # Initialize first row of tc array
    for j in range(1, n+1):
        tc[0][j] = tc[0][j-1] + cost[0][j]
 
    # Construct rest of the tc array
    for i in range(1, m+1):
        for j in range(1, n+1):
            tc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j]
 
    return tc[m][n]
  
# Driver code
cost = [[1, 2, 3],
        [4, 8, 2],
        [1, 5, 3]]

print("min Cost bottom up: ",minCost_bottom_up(cost, 2, 2))



def minCost_space_optimized(cost, row, col):
 
    # For 1st column
    for i in range(1, row):
        cost[i][0] += cost[i - 1][0]
 
    # For 1st row
    for j in range(1, col):
        cost[0][j] += cost[0][j - 1]
 
    # For rest of the 2d matrix
    for i in range(1, row):
        for j in range(1, col):
            cost[i][j] += (min(cost[i - 1][j - 1],
                               min(cost[i - 1][j],
                                   cost[i][j - 1])))
 
    # Returning the value in
    # last cell
    return cost[row - 1][col - 1]
 
 
# Driver code
if __name__ == '__main__':
 
    row = 3
    col = 3
 
    cost = [[1, 2, 3],
            [4, 8, 2],
            [1, 5, 3]]
 
    print("min Cost space optimized: ", minCost_space_optimized(cost, row, col))
