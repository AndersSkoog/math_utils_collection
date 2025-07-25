import math

cubling = lambda v: v * pow(v, 2)
pyth_area_tri = lambda n: int(((3 * n) * (4 * n)) / 2)
pyth_area_rect = lambda n: (3 * n) * (4 * n)
pyth_hypot = lambda n: int(math.hypot(3 * n, 4 * n))
pyth_prod = lambda n: math.prod([3 * n, 4 * n, pyth_hypot(n)])
pyth_tripple = lambda n: [3 * n, 4 * n, int(math.hypot(3 * n, 4 * n))]
pyth_trip_pow = lambda n: [pow(v, 2) for v in pyth_tripple(1)]
pyth_side_a = lambda n: 3 * n
pyth_side_b = lambda n: 4 * n
nth_root = lambda x,n: x ** 1/n
roots_of_unity = lambda n: math.log(pow(math.e,math.tau)) / n
normalize = lambda x,maximum: x / (maximum * x)
is_num = lambda x: isinstance(x,(int,float))
is_num_between = lambda x,_min,_max: is_num(x) and _min <= x <= _max
cot = lambda x: 1 / math.tan(x)
quotient = lambda a,b: (a - (a % b)) // b

def linlin(v,inmin,inmax,outmin,outmax): return (v - inmin)/(inmax - inmin) * (outmax - outmin) + outmin

def binomial(n,k): return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))



def amt(x):  return math.tau / math.log(x)

def pyth_third(n):
  whole = ((3*n) * (4*n)) / 3
  third_of_whole = whole / 3
  return [whole,third_of_whole]

# divide like diophantus
def divide_with_difference(whole, difference):
    """
    To divide a proposed number into two numbers having a given difference.
    Now, let the given number be 100, and the difference be 40 units.
    To find the numbers.
    Let the smaller be assigned to be 1 Number. Therefore, the greater will be 1
    Number, 40 units. The two together, therefore, become 2 Numbers, 40 units. But
    they were given to be 100 units. Thus, 100 units are equal to 2 Numbers, 40 units.
    And likes from likes From the 100 (units) I subtract 40 units, and from the
    2 Numbers and the 40 units, similarly, 40 units.  The remaining 2 Numbers are
    equal to 60 units; therefore, each one becomes 30 units.
    To the hypostases: the smaller will be 30 units, the greater 70 units, and the
    proof is obvious.
    """
    sml = (whole - difference) / 2
    grt = whole - sml
    return [grt, sml]

def divide_with_ratio(whole, ratio):
    """
    It is required to divide a proposed number into two numbers having a given ratio.
    Now, let it be proposed to divide 60 into two numbers having a triple ratio.
    Let the smaller be assigned to be 1 Number. Therefore, the greater will be 3 Numbers,
    so the greater is the triple of the smaller. Then the two must be equal to 60
    units. But the two, when added together, are 4 Numbers. Thus, 4 Numbers are equal to 60 units;
    therefore, the Number is 15 units. Thus, the smaller will be 15 units, and the greater 45 units.
    """
    sml = whole / (ratio + 1)
    grt = whole - sml
    return [grt, sml]

def divide_with_diff_and_ratio(whole, diff, ratio):
    """
    To divide a proposed number into two numbers having a given ratio and difference
    Now, let it be proposed to divide 80 into two numbers so that the greater is the
    triple of the smaller and, moreover, exceeds (it) by (an additional) 4 units.
    Let the smaller be assigned to be 1 Number. Therefore, the greater will be 3
    Numbers and 4 units; so the greater, being the triple of the smaller, exceeds more-
    over by (an additional) four units. Then, I want the two to be equal to 80 units. But
    the two, when added together, are 4 Numbers and 4 units. Thus, 4 Numbers and 4
    units are equal to 80 units.
    And I subtract likes from likes. Then, there remain 76 units equal to 4 Numbers,
    and the Number becomes 19 units
    """
    wsub = whole - diff
    sml = wsub / (ratio + 1)
    grt = (sml * ratio) + diff
    return [grt, sml]

def find_dividends(diff, ratio):
    """
    Now, let it be proposed that the greater is the quintuple of the smaller,
    and their difference makes 20 units.
    Let the smaller be assigned to be 1 Number. Therefore, the greater will be 5
    Numbers. Then, I want 5 Numbers to exceed 1 Number by 20 units. But their dif-
    ference is 4 Numbers. These are equal to 20 units.
    The smaller number will be 5 units, and the greater 25 units. And it is fulfilled,
    since the greater is the quintuple of the smaller, and the difference is 20 units.
    """
    part = diff / (ratio - 1)
    sml = part
    grt = part * ratio
    return [grt, sml]

def divide_by_ratios(whole, ratio1, ratio2):
    """
    Now, let it be proposed to divide 100 into two numbers such that the third of the
    first and the fifth of the second, if added together, make 30 units.
    I assign the fifth of the second to be 1 Number; (the second) itself, therefore,
    will be 5 Numbers. Then, the third of the first will be 30 units lacking 1 Number;
    (the first) itself, therefore, will be 90 units lacking 3 Numbers. Then, I want the
    two, if added together to make 100 units. But the two, when added together, make
    2 Numbers and 90 units. These are equal to 100 units.
    And likes from likes. Then, there remain 10 units equal to 2 Numbers;
    therefore, the Number will be 5 units.
    To the hypostases: I assigned the fifth of the second to be 1 Number;
    it will be 5 units; (the second) itself, therefore, will be 25 units.
    And (I assigned) the third of the first, to be 30 units lacking 1 Number; it will be | 25 units;
    (the first) itself, therefore, will be 75 units. And it is fulfilled,
    since the third of the first and the fifth of the second is 30 units,
    and these, when added together, make the proposed number
    """
    r1 = max(ratio1, ratio2)
    r2 = min(ratio1, ratio2)
    part1 = whole / (r1 + 1)
    part2 = part1 / r2
    grt = part1 * r1
    sml = part2 * r2
    return [grt, sml, grt + sml]

def continued_fraction(x, limit=10):
    terms = []
    while limit > 0:
        frc = x - int(x)
        rec = 1 / frc
        t = int(rec)
        if frc == 0:
            break
        terms.append(t)
        x = rec
        limit -= 1
    return terms

def continued_fraction_reconst(rep):
    result = rep[-1]
    # Iterate backward through the list
    for i in range(len(rep) - 2, -1, -1):
        result = rep[i] + 1 / result
    return result

def fraction_between(x, y):
    """
    Simplest fraction strictly between fractions x and y.
    """
    if x == y:
        raise ValueError("no fractions between x and y")

    # Reduce to case 0 <= x < y
    x, y = min(x, y), max(x, y)
    if y <= 0:
        return -fraction_between(-y, -x)
    elif x < 0:
        return 1

    s, t, u, v = x.numerator, x.denominator, y.numerator, y.denominator
    a, b, c, d = 1, 0, 0, 1
    while True:
        q = s // t
        s, t, u, v = v, u - q * v, t, s - q * t
        a, b, c, d = b + q * a, a, d + q * c, c
        if t > s:
            return (a + b) / (c + d)

def gcd(a, b):
  _a = a if a > 0 else -a
  _b = b if b > 0 else -b
  while True:
    if _b == 0:
        return _a
    _a %= _b
    if a == 0:
        return _b
    _b %= _a

def lcm(a, b): return a * b / gcd(a, b)

def is_square(n): return math.sqrt(n) % 1 == 0

def harm_mean(a, b):
  nom = 2 * (a*b)
  dnom = a + b
  return nom / dnom

def arithmetic_mean(A, B): return (A + B) / 2

def gnomonic_number(n): return 1 if n == 1 else pow(n, 2) - pow(n - 1, 2)

def polygon_num(s, n):
    t1 = (n * (n - 1)) / 2
    t2 = s - 2
    res = t2 * t1 + n
    return 1 if n <= 1 else int(res)

def centered_polygon_num(s,n):
    return 1 if n <= 1 else int((s * pow(n,2) - s * n + 2) / 2)

def tetrahedral_numbers(n):
    return 1 if n <= 1 else int((n * ((n + 1) * (n + 2))) / 6)

def cubic_number(n):
    return 1 if n <= 1 else pow(n, 3)

def octahedral_number(n):
    return 1 if n <= 1 else int((n * (pow(2 * n, 2) + 1)) / 3)

def icosahedral_number(n):
     n2 = pow(n, 2)
     n5 = 5 * n
     return 1 if n <= 1 else int((n * (5*n2 - n5 + 2)) / 2)

def dodecahedral_number(n):
    nom = n * ((3*n) - 1) * ((3*n) - 2)
    return 1 if n <= 1 else int(nom / 2)

def stella_octangula_number(n):
    return 1 if n <= 1 else octahedral_number(n) + (8 * tetrahedral_numbers(n - 1))

def rhombic_dodecahedron_number(n):
    return 1 if n <= 1 else centered_cube_number(n) + (6 * m_gon_pyramid(4, n - 1))

def centered_cube_number(n):
    return 1 if n <= 1 else pow(n, 3) + pow(n - 1, 3)

def m_gon_pyramid(m, n):
    nom = n * (n + 1) * ((m - 2) * n - m + 5)
    dnom = 6
    return 1 if n <= 1 else int(nom / dnom)

def centered_mgon_pyramid(m, n):
    n2 = 2 * n - 1
    m_1 = m - 1
    nn = pow(n,2)
    nom = (n2 * m_1 * nn) - (m_1 * n + 6)
    return 1 if n <= 1 else int(nom/2)
    #return 1 if n <= 1 else int(((2*n - 1) * (s - 1) * pow(n,2) - (s - 1) * n + 6) / 2)

def seq(s_start,s_stop,init,fn):
    ret = init
    for i in range(s_start,s_stop):
        nv = fn(ret,i)
        ret.append(nv)
    return ret

def get_perfect(doubles_seq):
    s = sum(doubles_seq)
    last = doubles_seq[len(doubles_seq) - 1]
    return s * last

def polygon_diagonals(sides):
    return sides * (sides - 3) / 2

def algo_seq(n_start, n_stop, fn, init):
    ret = init
    for i in range(n_start, n_stop):
        v = fn(ret, i)
        if v:
         ret.append(v)
    return ret

def solid_numbers(n_iter):
    laterals_a = []
    laterals_b = []
    diagonals_a = []
    diagonals_b = []
    prev_s = 1
    prev_d = 1
    for i in range(1,n_iter):
      s = prev_s + prev_d
      d = prev_d + (prev_s * 2)
      s2 = s ** 2
      d2 = d ** 2
      prev_s = s
      prev_d = d
      laterals_a.append(s)
      laterals_b.append(s2)
      diagonals_a.append(d)
      diagonals_b.append(d2)
    return [laterals_a,laterals_b,diagonals_a,diagonals_b]

def is_even(v):return v % 2 == 0

def is_odd(v):return v % 2 == 1

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

def is_number(x):return isinstance(x,(int,float))

def in_range(value,low,high):
    """Ensures value is in the range [0, π]."""
    if value < low or value > high: raise ValueError(f"Value {value} out of range")
    return value
