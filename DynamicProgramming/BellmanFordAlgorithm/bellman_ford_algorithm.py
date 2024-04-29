"""
Imagine you have a map with different cities connected by roads, each road having 
a certain distance. The Bellman–Ford algorithm is like a guide that helps you find 
the shortest path from one city to all other cities, even if some roads have negative lengths. 
It’s like a GPS for computers, useful for figuring out the quickest way to get 
from one point to another in a network
"""
"""
Bellman-Ford is a single source shortest path algorithm that determines the shortest path 
between a given source vertex and every other vertex in a graph. This algorithm can be used 
on both weighted and unweighted graphs.
"""
"""
A Bellman-Ford algorithm is also guaranteed to find the shortest path in a graph, 
similar to Dijkstra’s algorithm. Although Bellman-Ford is slower than Dijkstra’s algorithm, 
it is capable of handling graphs with negative edge weights, which makes it more versatile. 
The shortest path cannot be found if there exists a negative cycle in the graph. 
If we continue to go around the negative cycle an infinite number of times, 
then the cost of the path will continue to decrease (even though the length of the path 
is increasing). As a result, Bellman-Ford is also capable of detecting negative cycles, 
which is an important feature.
"""
"""
The Bellman-Ford algorithm’s primary principle is that it starts with a single source 
and calculates the distance to each node. The distance is initially unknown and assumed to be 
infinite, but as time goes on, the algorithm relaxes those paths by identifying 
a few shorter paths. Hence it is said that Bellman-Ford is based on “Principle of Relaxation“.
"""
# https://www.geeksforgeeks.org/bellman-ford-algorithm-dp-23/
# Python3 program for Bellman-Ford's single source
# shortest path algorithm.

# Class to represent a graph
class Graph:

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []

    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    # utility function used to print the solution
    def printArr(self, dist):
        print("Vertex Distance from Source")
        for i in range(self.V):
            print("{0}\t\t{1}".format(i, dist[i]))

    # The main function that finds shortest distances from src to
    # all other vertices using Bellman-Ford algorithm. The function
    # also detects negative weight cycle
    def BellmanFord(self, src):

        # Step 1: Initialize distances from src to all other vertices
        # as INFINITE
        dist = [float("Inf")] * self.V
        dist[src] = 0

        # Step 2: Relax all edges |V| - 1 times. A simple shortest
        # path from src to any other vertex can have at-most |V| - 1
        # edges
        for _ in range(self.V - 1):
            # Update dist value and parent index of the adjacent vertices of
            # the picked vertex. Consider only those vertices which are still in
            # queue
            for u, v, w in self.graph:
                if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                    dist[v] = dist[u] + w

        # Step 3: check for negative-weight cycles. The above step
        # guarantees shortest distances if graph doesn't contain
        # negative weight cycle. If we get a shorter path, then there
        # is a cycle.

        for u, v, w in self.graph:
            if dist[u] != float("Inf") and dist[u] + w < dist[v]:
                print("Graph contains negative weight cycle")
                return

        # print all distance
        self.printArr(dist)


# Driver's code
if __name__ == '__main__':
    g = Graph(5)
    g.addEdge(0, 1, -1)
    g.addEdge(0, 2, 4)
    g.addEdge(1, 2, 3)
    g.addEdge(1, 3, 2)
    g.addEdge(1, 4, 2)
    g.addEdge(3, 2, 5)
    g.addEdge(3, 1, 1)
    g.addEdge(4, 3, -3)

    # function call
    g.BellmanFord(0)
