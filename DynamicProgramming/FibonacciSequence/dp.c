// Fibonacci Series using Dynamic Programming
#include <stdio.h>
#include <time.h>
 
long long int fib(int n)
{
    /* Declare an array to store Fibonacci numbers. */
    long long int f[n+2]; // 1 extra to handle case, n = 0
    int i;
 
    /* 0th and 1st number of the series are 0 and 1*/
    f[0] = 0;
    f[1] = 1;
 
    for (i = 2; i <= n; i++) {
        /* Add the previous 2 numbers in the series
           and store it */
        f[i] = f[i - 1] + f[i - 2];
    }
 
    return f[n];
}
 
int main()
{
    clock_t start = clock();

    int n = 48;
    printf("%lld\n", fib(n));

    clock_t end = clock();
    double elapsed_time = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("elapsed time: %f milliseconds\n", elapsed_time);

    // getchar();
    return 0;
}