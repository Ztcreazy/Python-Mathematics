# https://www.geeksforgeeks.org/min-cost-path-dp-6/
"""
Follow the below steps to solve the problem:

Create a 2-D dp array to store answer for each cell
Declare a priority queue to perform dijkstra’s algorithm
Return dp[M][N]
"""
# Minimum Cost Path using Dijkstra’s shortest path
#  algorithm with Min Heap by dinglizeng
# Python3
 
# Define the number of rows and the number of columns
R = 4
C = 5
 
# 8 possible moves 
dx = [ 1, -1, 0, 0, 1, 1, -1, -1 ] 
dy = [ 0, 0, 1, -1, 1, -1, 1, -1 ] 
 
# The data structure to store the coordinates of 
#  the unit square and the cost of path from the top 
#  left. 
class Cell(): 
    def __init__(self, x, y, z): 
        self.x = x 
        self.y = y 
        self.cost = z 
 
# To verify whether a move is within the boundary. 
def isSafe(x, y): 
    return (x >= 0 and x < R and
            y >= 0 and y < C) 
 
# This solution is based on Dijkstra’s shortest
#  path algorithm 
# For each unit square being visited, we examine all
#  possible next moves in 8 directions, 
# calculate the accumulated cost of path for each
#  next move, adjust the cost of path of the adjacent 
#  units to the minimum as needed. 
# then add the valid next moves into a Min Heap. 
# The Min Heap pops out the next move with the minimum 
# accumulated cost of path. 
# Once the iteration reaches the last unit at the lower 
# right corner, the minimum cost path will be returned. 
def minCost(cost, m, n): 
 
    # the array to store the accumulated cost
    # of path from top left corner 
    dp = [[0 for x in range(C)] for x in range(R)] 
 
    # the array to record whether a unit
    # square has been visited 
    visited = [[False for x in range(C)] 
                for x in range(R)] 
 
    # Initialize these two arrays, set path cost 
    # to maximum integer value, each unit as 
    # not visited 
    for i in range(R): 
        for j in range(C): 
            dp[i][j] = float("Inf") 
            visited[i][j] = False
 
    # Define a reverse priority queue. 
    # Priority queue is a heap based implementation. 
    # The default behavior of a priority queue is
    # to have the maximum element at the top. 
    # The compare class is used in the definition of
    # the Min Heap. 
    pq = [] 
 
    # initialize the starting top left unit with the 
    # cost and add it to the queue as the first move. 
    dp[0][0] = cost[0][0] 
    pq.append(Cell(0, 0, cost[0][0])) 
 
    while(len(pq)): 
     
        # pop a move from the queue, ignore the units 
        # already visited 
        cell = pq[0] 
        pq.pop(0) 
        x = cell.x 
        y = cell.y 
        if(visited[x][y]): 
            continue
 
        # mark the current unit as visited 
        visited[x][y] = True
 
        # examine all non-visited adjacent units in 8
        # directions 
        # calculate the accumulated cost of path for
        # each next move from this unit, 
        # adjust the cost of path for each next
        # adjacent units to the minimum if possible. 
        for i in range(8): 
            next_x = x + dx[i] 
            next_y = y + dy[i] 
            if(isSafe(next_x, next_y) and
                not visited[next_x][next_y]): 
                dp[next_x][next_y] = min(dp[next_x][next_y],
                                        dp[x][y] + cost[next_x][next_y]) 
                pq.append(Cell(next_x, next_y, 
                                dp[next_x][next_y])) 
 
    # return the minimum cost path at the lower 
    # right corner 
    return dp[m][n] 
 
# Driver code 
cost = [[1, 8, 8, 1, 5], 
        [4, 1, 1, 8, 1], 
        [4, 2, 8, 8, 1], 
        [1, 5, 8, 8, 1]]
 
print("min Cost Dijkstra: ", minCost(cost, 3, 4))
