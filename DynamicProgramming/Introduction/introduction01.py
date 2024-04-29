# https://www.geeksforgeeks.org/introduction-to-dynamic-programming-data-structures-and-algorithm-tutorials/
"""
Steps to solve a Dynamic programming problem:
1. Identify if it is a Dynamic programming problem.
2. Decide a state expression with the Least parameters.
3. Formulate state and transition relationships.
4. Do tabulation (or memorization).
"""
"""
1) How to classify a problem as a Dynamic Programming algorithm Problem?
Typically, all the problems that require maximizing or minimizing certain quantities 
or counting problems that say to count the arrangements under certain conditions 
or certain probability problems can be solved by using Dynamic Programming.

All dynamic programming problems satisfy the overlapping subproblems property 
and most of the classic Dynamic programming problems also satisfy 
the optimal substructure property. Once we observe these properties in a given problem 
be sure that it can be solved using Dynamic Programming.

2) Deciding the state:
Problems with dynamic programming are mostly concerned with the state and its transition. 
The most fundamental phase must be carried out with extreme care because the state transition 
depends on the state definition you select.

A state is a collection of characteristics that can be used to specifically describe 
a given position or standing in a given challenge. To minimise state space, 
this set of parameters has to be as compact as feasible.

3) Formulating a relation among the states:
The hardest part of a Dynamic Programming challenge is this step, which calls for 
a lot of intuition, observation, and training.

Example:

Given 3 numbers {1, 3, 5}, the task is to tell the total number of ways we can form 
a number N using the sum of the given three numbers. 
(allowing repetitions and different arrangements).
"""
# Function to find nth fibonacci number 
def fib_recursive(n): 
    if (n <= 1): 
        return n 
    x = fib_recursive(n - 1) 
    y = fib_recursive(n - 2) 
  
    return x + y 
  


# Helper Function 
def fibo_helper(n, ans): 
  # Base case 
  if (n <= 1): 
    return n 
  
  # To check if output already exists 
  if (ans[n] != -1): # is not
    return ans[n] 
  
  # Calculate output 
  x = fibo_helper(n - 1, ans) 
  y = fibo_helper(n - 2, ans) 
  
  # Saving the output for future use 
  ans[n] = x + y 
  
  # Returning the final output 
  return ans[n] 
  
def fibo_memoization(n): 
  ans = [-1]*(n+1) 
  # print("ans: ", ans)

  # Initializing with -1 
  #for (i = 0; i <= n; i++) { 
  for i in range(0,n+1): 
    ans[i] = -1
  # print("ans: ", ans)

  return fibo_helper(n, ans)



# Function for calculating the nth 
# Fibonacci number 
def fibo_bottom_up(n): 
  ans = [None] * (n + 1) 
  
  # Storing the independent values in the 
  # answer array 
  ans[0] = 0
  ans[1] = 1
  
  # Using the bottom-up approach 
  for i in range(2,n+1): 
    ans[i] = ans[i - 1] + ans[i - 2] 
  
  # Returning the final index 
  return ans[n] 



def fibo_observation(n): 
    prevPrev, prev, curr = 0, 1, 1
    # Using the bottom-up approach 
    for i in range(2, n+1): 
        curr = prev + prevPrev 
        prevPrev = prev 
        prev = curr 
    # Returning the final answer 
    return curr 

# n = 15
n = 48 

# Function Call 
# print(f"recursive fib_recursive({n}): {fib_recursive(n)}")
print(f"memoization fibo_memoization({n}): {fibo_memoization(n)}")
print(f"bottom up fibo_bottom_up({n}): {fibo_bottom_up(n)}")
print(f"bottom up fibo_observation({n}): {fibo_observation(n)}")
