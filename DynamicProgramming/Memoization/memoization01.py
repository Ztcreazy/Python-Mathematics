# https://www.geeksforgeeks.org/what-is-memoization-a-complete-tutorial/
# https://www.geeksforgeeks.org/tabulation-vs-memoization/
"""
In computing, memoization is used to speed up computer programs by 
eliminating the repetitive computation of results, and by avoiding repeated calls 
to functions that process the same input.
"""
"""
Memoization is a top-down approach where we cache the results of function calls 
and return the cached result if the function is called again with the same inputs. 
It is used when we can divide the problem into subproblems 
and the subproblems have overlapping subproblems. Memoization is typically implemented 
using recursion and is well-suited for problems that have a relatively small set of inputs.

Tabulation is a bottom-up approach where we store the results of the subproblems 
in a table and use these results to solve larger subproblems 
until we solve the entire problem. It is used when we can define the problem 
as a sequence of subproblems and the subproblems do not overlap. 
Tabulation is typically implemented using iteration and is well-suited for problems 
that have a large set of inputs.
"""
import sys
sys.set_int_max_str_digits(0) # ValueError: maxdigits must be 0 or larger than 640

# Memoization implementation:
def fibonacci_memoization(n, cache={}): # cache memo
    if n in cache:
        return cache[n]
    if n == 0:
        result = 0
    elif n == 1:
        result = 1
    else:
        result = fibonacci_memoization(n-1) + fibonacci_memoization(n-2)
    
    print("memoization cache: ", cache)
    cache[n] = result

    # print("memoization cache: ", cache)
    return result



# Tabulation implementation:
def fibonacci_tabulation(n):
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        table = [0] * (n + 1)
        table[0] = 0
        table[1] = 1
        for i in range(2, n+1):
            table[i] = table[i-1] + table[i-2]
        
        print("tabulation table: ", table)
        return table[n]

n = 500
print(f"memoization fibo_memoization({n}): {fibonacci_memoization(n)}")
print(f"tabulation fibonacci_tabulation({n}): {fibonacci_tabulation(n)}")
