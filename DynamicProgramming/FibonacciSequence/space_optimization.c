#include <stdio.h>
#include <time.h>

long long int fib(int n)
{
    long long int a = 0, b = 1, c, i;
    if (n == 0)
        return a;
    for (i = 2; i <= n; i++) {
        c = a + b;
        a = b;
        b = c;
    }
    return b;
}
 
int main()
{
    clock_t start = clock();

    int n = 48;
    printf("fib(n): %lld\n", fib(n));

    clock_t end = clock();
    double elapsed_time = ((double) (end - start)) / CLOCKS_PER_SEC * 1000.0;
    printf("elapsed time: %f milliseconds\n", elapsed_time);
    
    // getchar();
    return 0;
}
