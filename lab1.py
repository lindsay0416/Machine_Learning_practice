#import numpy as np

def fib(n):
    a, b = 0, 1
    while a < n:
        print(a, end=' ')
        if 0:
            a, b = b, a+b
        elif 0:
            T = a
            a = b
            b = T + b
        else:
            b = a + b
            a = b - a

    print()

fib(10)