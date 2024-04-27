import time
# Fibonacci Series using Dynamic Programming
def fibonacci(n):
 
    # Taking 1st two fibonacci numbers as 0 and 1
    f = [0, 1]
 
    for i in range(2, n+1):
        f.append(f[i-1] + f[i-2])
    return f[n]
 
start_time = time.time() 
print("fibonacci 48: ", fibonacci(48))
end_time = time.time()
elapsed_time = (end_time - start_time) *1000
print("elapsed time: ", elapsed_time, "seconds")
