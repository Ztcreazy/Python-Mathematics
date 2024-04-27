import time

def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
 
 
if __name__ == "__main__":
    
    start_time = time.time()

    n = 28
    print(n, "th Fibonacci Number: ")
    print(fibonacci(n))

    end_time = time.time()
    elapsed_time = (end_time - start_time) *1000
    print("elapsed time: ", elapsed_time, "milliseconds")
