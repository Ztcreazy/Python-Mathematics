#include <stdio.h>
#include <time.h>

long long int fib(int n)
{
    if (n <= 1)
        return n;
    return fib(n - 1) + fib(n - 2);
}
 
int main()
{
    clock_t start = clock();

    int n = 48;
    printf("%dth Fibonacci Number: %lld\n", n, fib(n));

    clock_t end = clock();
    double elapsed_time = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("elapsed time: %f milliseconds\n", elapsed_time);

    return 0;
}
