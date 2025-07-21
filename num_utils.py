import math

def is_even(v):
    return v % 2 == 0

def is_odd(v):
    return v % 2 == 1

def quotient(a, b):
    r = a % b
    return (a - r) // b

def factors(num):
    ret = []
    for n in range(1, num):
        if num % n == 0:
            ret.append(n)
    return ret

def sieve(n):
    numbers = [True] * n
    primes = []
    n_sqrt = pow(n, 0.5)
    i1 = 2

    while i1 < n_sqrt:
        i1 += 1
        j = i1 * i1
        # print(j)
        while j < n:
            # print(j)
            numbers[j] = False
            j += i1

    for i in range(2, n):
        if numbers[i]:
            primes.append(i)

    return primes