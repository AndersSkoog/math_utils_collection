from argparse import ArgumentTypeError
import numpy as np
import math
import itertools
from numbers import Number
from fractions import Fraction
import cmath
from Types import Number






number_decimals = 8
tau = math.tau
circ_angle = lambda div,n: math.sin((math.pi/div) * n)
cubling = lambda v: v * pow(v, 2)
pyth_area_tri = lambda n: int(((3 * n) * (4 * n)) / 2)
pyth_area_rect = lambda n: (3 * n) * (4 * n)
pyth_hypot = lambda n: int(math.hypot(3 * n, 4 * n))
pyth_prod = lambda n: math.prod([3 * n, 4 * n, pyth_hypot(n)])
pyth_tripple = lambda n: [3 * n, 4 * n, int(math.hypot(3 * n, 4 * n))]
pyth_trip_pow = lambda n: [pow(v, 2) for v in pyth_tripple(1)]
pyth_side_a = lambda n: 3 * n
pyth_side_b = lambda n: 4 * n
wrap = lambda x,bound: round(bound * ((x/bound) - int(x/bound)),number_decimals) if x > bound else x
cyc_wrap = lambda x,bound: wrap(x,bound) if int(x/bound) % 2 == 0 else round(bound - (wrap(x,bound)),number_decimals)
cyc_direction = lambda x,bound: 1 if int(cyc_wrap(x,bound)) % 2 == 0 else -1
prop = lambda w, a, b: w / (a / b)
frac_series = lambda w, q, n: [int(w * (1 / q)) * i for i in range(1, n + 1)]
prop_golden_ratio = lambda w: prop(w,1+(5**1/2),2)
quotient = lambda a,b: (a - (a % b)) // b
cot = lambda x: 1 / math.tan(x)
nth_root = lambda x,n: x ** 1/n
phi = (1 + (5 ** 1/2)) / 2
ang_degrees = np.linspace(0,tau,360)
theta = lambda n: math.sin((tau/360) * n % 90)
circum = lambda r: tau*r
roots_of_unity = lambda n: math.log(math.e ** tau) / n
sphere_vol = lambda r: tau * pow(r,3)
normalize = lambda x,maximum: x / (maximum * x)
arclength = lambda frac,radius: circum(radius) / frac
tau_dec = tau - int(tau)
e_dec = math.e - int(math.e)
ang_val = lambda n: ang_degrees[n%360]
quantize = lambda x,n: (0.1 * n) * tau
arc_len = lambda div,n: circ_angle(int(div),int(n))
#is_bounded_index = lambda ind,arr: isinstance(ind,int) and ind >= 0 and ind <= (len(arr) -1)
right_angle = math.radians(90)
is_number_list = lambda arr: isinstance(arr,list) and all(isinstance(x,(int,float)) for x in arr)
t_of_k = lambda k: math.log(pow(math.e,k)) / k
sinc = lambda x: math.sin(x) / x


def spatial_unit(q,aspect_ratio):
  u = q / sum(aspect_ratio)
  l = len(aspect_ratio)
  return [u*aspect_ratio[axis] for axis in range(0,l)]





#calculate volume and surface area of a sphere in any dimension given side length for a square area
def ball(side_length, dim):
     l = side_length
     r = l / 2
     fourth_of_tau = math.tau / 4
     if dim > 1:
       vol = (dim * fourth_of_tau) * pow(r,dim)
       area = (2 * (dim * fourth_of_tau)) * pow(r,dim - 1)
       return {"vol": vol, "area": area}
     elif dim == 1:
       vol = 2*r
       area = 2
       return {"vol":vol,"area":area}
     else: raise ValueError("dim must be positive integer <= 1")

def ball2(l:Number,dim:int):
    r = l / 2
    nomval = pow(math.pi,int(dim/2))
    gamval = math.gamma(int(dim/2) + 1)
    # Volume formula for an n-ball
    vol = (nomval / gamval) * pow(r, dim)
    # Surface area formula for an (n-1)-sphere
    area = ((2 * nomval) / gamval) * pow(r,dim-1)
    return {"vol":vol,"area":area}

def cart_to_sphere(p):
  u,v = p[0],p[1]
  dnom = 1 + pow(u,2) + pow(v,2)
  x,y,z = (2*u) / dnom, (2*v) / dnom, (-1 + pow(u,2) + pow(v,2)) / dnom
  return [x,y,z]

class ProjectiveSpace:

    def __init__(self,L:Number):
      self.sidelength = L
      self.sqr_area = L ** 2
      self.circ_diam = L
      self.circ_radius = L / 2
      self.circum = math.tau / self.circ_radius
      self.circ_area = math.tau * self.circ_radius

    def volume_ball(self,dimension:int):
        return self.circum * pow(math.pi,int(dimension/2)) * pow(self.circ_radius,dimension)



def pyth_third(n):
  whole = ((3*n) * (4*n)) / 3
  third_of_whole = whole / 3

  return [whole,third_of_whole]

#print("pyth_third",pyth_third(1))




def linlin(v,inmin,inmax,outmin,outmax): return (v - inmin)/(inmax - inmin) * (outmax - outmin) + outmin

def in_S2(tup):
    x1,x2,x3 = tup[0],tup[1],tup[2]
    return True if pow(x1,2) + pow(x2,2) + pow(x3,2) == 1 else False


def unary_operator(fn,dim):
    if isinstance(dim,int) and dim > 0:
        if dim == 1: return fn
        else:
          return lambda a: [fn(a[i]) for i in range(dim)]
    else:
        raise ArgumentTypeError("dim must be int > 0")

def binary_operator(fn,dim):
    if isinstance(dim,int) and dim > 0:
        if dim == 1: return fn
        else:
          return lambda a,b: [fn(a[i],b[i]) for i in range(dim)]
    else:
        raise ArgumentTypeError("dim must be int > 0")

def mobius_trans(z,coef):
  a,b,c,d = coef[0],coef[1],coef[2],coef[3]
  return ((a*z) + b) /((c*z) + d)

def mobius_transl(z,b):
  return mobius_trans(z,[1,b,0,1])

def mobius_scale(z,s):
    return mobius_trans(z,[s,0,0,1])

def mobius_inv(z):
    return mobius_trans(z,[0,1,1,0])

def mobius_rot(z,ang):
    return mobius_trans(z, [cmath.exp(1j * ang), 0, 0, 1])

def wedge(v1,v2):
  e = dict()
  for i in range(len(v1)):
      for j in range(i + 1, len(v1)):
        k = "e"+str(i)+str(j)
        if i != 0:
          e[k] = (v1[i] * v2[j] - v1[j] * v2[i])
  return e, sum(e.values())
















"""

class composite_number:

    def __init__(self,real,imag_parts):
      if is_number_list(imag_parts) and isinstance(real,(int,float)):
        self.real = real
        self.imag_parts = imag_parts
        self.highest_dim = len(imag_parts) - 1
        self.imag_sum = sum(self.imag_parts)
        self.angles = [math.radians(normalize(p,360)) for p in self.imag_parts]
        self.wrap = lambda x, bound: round(bound * ((x / bound) - int(x / bound)),8) if x > bound else round(x,8)
        self.cyc_wrap = lambda x, bound: wrap(x, bound) if int(x / bound) % 2 == 0 else round(bound - (wrap(x, bound)),8)
        #self.direction = 1 if int(num / math.pi) % 2 == 0 else -1
        #self.angle = (tau/4) + cyc_wrap(num,right_angle)
        #self.value = math.sin(self.angle)

    def get_direction(self,part_index):
       if is_bounded_index(self.imag_parts,part_index):
           return int(self.imag_parts[part_index]/right_angle) % 2 == 0

    def get_angle(self,part_index,sector=None):
        if is_bounded_index(self.imag_parts,part_index):
            imag_part = self.imag_parts[part_index]
            normal_angle = self.cyc_wrap(imag_part,right_angle)
            if sector and isinstance(sector, int) and 0 <= sector <= 3:
                return normal_angle + (right_angle * sector)
            else:
                return normal_angle + (right_angle * sector)
        else: raise ValueError("index out of range")

    def get_normal(self,part_index):
        return round(math.sin(self.get_angle(part_index)[0]),8)

    def get_number_of_turns(self,part_index):
        if is_bounded_index(self.imag_parts,part_index):
            return int(self.imag_parts[part_index]/math.tau)
        else: raise ValueError("index out of range")

    def nsphere_coordinate(self,rad_ratio,dim):
        cos_c = lambda i: math.cos(self.angles[i])
        sin_c = lambda i: math.sin(self.angles[i])
        cos_d,sin_d,cos_l,sin_l = cos_c(-1),sin_c(-1),cos_c(-2),sin_c(-2)
        br = self.real ** dim
        rad = normalize(rad_ratio,br) * br
        comp_first = cos_d
        comp_mid = lambda j: math.prod([sin_d] + [sin_c(i) if i < (j-1) else cos_c(i) for i in range(0,j)])
        comp_last = lambda j: math.prod([sin_d] + [sin_c(i) for i in range(0,j)])
        if dim == 1:
            return [rad,comp_first]
        elif dim == 2:
            return [rad,comp_first,comp_mid(2)]
        else:
            mids = [comp_mid(k) for k in range(dim)]
            return [rad,comp_first] + mids + comp_last(dim)

"""





def amt(x):  return tau / math.log(x)

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

def rational_div(nom, dnom):
    if nom // dnom - nom / dnom == 0:
        return nom // dnom
    else:
        return Fraction(nom, dnom)

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

def lies_on_line(p,l):
    """
    The point A ≡ [x, y] lies on the line l ≡ <a : b : c>,
    or equivalently the line l passes through the point A,
    precisely when ax + by + c = 0.
    """
    a, b, c, x, y = l[0], l[1], l[2], p[0], p[1]
    return (a * x) + (b * y) + c == 0

def is_central_line(l: [Number]):
    return l[2] == 0

def pass_through_point(l: [Number], p: [Number]):
    a, b, c, x, y = l[0], l[1], l[2], p[0], p[1]
    return (a * x) + (b * y) + c == 0

def abs_diff(vec):
    return abs(vec[0] - vec[1])

def is_perp(l1, l2):
  a1, b1, a2, b2 = l1[0], l1[1], l2[0], l2[1]
  return (a1 * a2) + (b1 * b2) == 0

def lines_concurrent(lines):
  if len(lines) >= 3:
    p = intersection_point(lines[0], lines[1])
    return all(map(lambda l: pass_through_point(l, p), lines))

def Alt_to_line(p: [Number], l: [Number]):
  """
  For any point A ≡ [x, y] and any line l ≡ <ha : b : c>
  there is a unique line n, called the altitude from A to l,
  which passes through A and is perpendicular to l, namely:
        n = <−b : a : bx − ay>
  """
  na = l[0]
  nb = l[0]
  nc = (l[1] * p[0]) - (l[0] * p[0])
  return [na, nb, nc]

def foot_of_alt(p: [Number], l: [Number]):
  """
  For any point A ≡ [x, y] and any non-null line l ≡ <a : b : c>,
  the altitude n from A to l intersects l at the point:
  F ≡ [x,y]
  x =  b²x - aby - ac  / a² + b²
  y = -abx + a²y - bc /  a² + b²
  """
  if is_null_line(l):
      return None
  else:
    a, b, c, x, y = l[0], l[1], l[2], p[0], p[1]
    sqr_a, sqr_b = pow(a, 2), pow(b, 2)
    dnom = sqr_a + sqr_b
    nx = rational_div((sqr_b * x) - (a * b * y) - (a * c), dnom)
    ny = rational_div(-(a * b * x) + (sqr_a * y) - (b * c), dnom)
    return [nx, ny]

def spread(l1: [Number], l2: [Number]):
  if is_null_line(l1) and is_null_line(l2):
    a1, b1, c1, a2, b2, c2 = l1[0], l1[1], l1[2], l2[0], l2[1], l2[2]
    nom = ((a1 * b2) - (a2 * b1)) ** 2
    dnom = (pow(a1, 2) + pow(b1, 2)) * (pow(a2, 2) + pow(b2, 2))
    return nom / dnom
  else:
    return None

def adj(l):
    return [[l[i - 1], l[i]] for i in range(1, len(l))]

def adjacents(l):
  return [[l[i],l[i + 1]] for i in range(len(l) - 1)]

def is_even(v):
    return v % 2 == 0

def is_odd(v):
    return v % 2 == 1

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

def binomial(n,k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n-k))


def catalan(n):
    """get the n-th catalan number"""
    return int(math.factorial(2 * n) / (math.factorial(n + 1) * math.factorial(n)))

def ngon_triangulations(n):
    """get the number of triangulations of a regular n-gon"""
    return catalan(n - 2)

def ngon_diagonals(n):
    """get all diagonals of a regular n-gon"""
    return [(i, j) for i in range(1, n) for j in range(i + 2, n + 1) if not (i == 1 and j == n)]

def do_intersect(d1, d2):
    """Check if two diagonals intersect."""
    (a, b), (c, d) = sorted(d1), sorted(d2)
    return (a < c < b < d) or (c < a < d < b)

def is_valid_triangulation(d):
    """Check if the set of diagonals forms a valid triangulation."""
    # Check for intersections between diagonals
    for d1, d2 in itertools.combinations(d, 2):
        if do_intersect(d1, d2):
            # print(d1,d2)
            return False
    return True

def generate_triangulations(n):
    """Generate all combinations of diagonals that can form a triangulation"""
    diags = ngon_diagonals(n)
    triangulations = []
    """
    The Triangulation Diagonal Count Relation is a mathematical expression that quantifies the number of diagonals 
    required to triangulate a regular n-gon. 
    It states that the number of non-intersecting diagonals needed to divide a regular n-gon into triangles is given by n - 3
    """
    triangulation_diagonal_count = n - 3
    for diag_comb in itertools.combinations(diags, triangulation_diagonal_count):
        if is_valid_triangulation(diag_comb):
            triangulations.append(diag_comb)

    return triangulations

def generate_frieze_pattern(n, I):
    # n = n-gon, I = choice of triangulation
    id_order = list(range(1, n + 1))
    tri_index = int(I % ngon_triangulations(n))
    tri_diagonals = generate_triangulations(n)[tri_index]
    flat_list = [item for sublist in tri_diagonals for item in sublist]
    frieze_seq = []
    for v in id_order:
        if v in flat_list:
            frieze_seq.append(flat_list.count(v) + 1)
        else:
            frieze_seq.append(1)
    r_1 = [1] * (n + 4)
    r_2 = frieze_seq + frieze_seq[0:4]
    r_3 = [int(((r_2[i - 1] * r_2[i]) - 1) / r_1[i - 1]) for i in range(1, n + 4)]
    r_4 = [int(((r_3[i - 1] * r_3[i]) - 1) / r_2[i]) for i in range(1, n + 3)]
    r_5 = [int(((r_4[i - 1] * r_4[i]) - 1) / r_3[i]) for i in range(1, n + 2)]
    r_6 = [int(((r_5[i - 1] * r_5[i]) - 1) / r_4[i]) for i in range(1, n + 1)]
    #print(r_1)
    return [r_1, r_2, r_3, r_4, r_5, r_6]

def point_on_the_unit_circle(y):
    x = math.sqrt(1 - pow(y, 2))
    return {
        "point": [x, y],
        "reflected_point": [-x, y]
    }

def point_on_circle_sqrt(y, radius):
    x = math.sqrt(radius - pow(y, 2))
    return {
        "point": [x, y],
        "reflected_point": [-x, y]
    }

def spherical_to_cartesian(r, theta, phi):
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return [x, y, z]

def is_number(x):
  return isinstance(x,(int,float))

def is_two_tupple(x):
  return isinstance(x,tuple) and len(x) == 2

def is_two_tupple_numeric(x):
  return is_two_tupple(x) and is_number(x[0]) and is_number(x[1])

def is_two_tupple_product(x):
  return is_two_tupple_numeric(x) and abs(x[0]) == abs(x[1])

def find_min(arr, ind):
    """
    given a list of lists with numberic entries
    returns the element with the highest value at index ind
    :param arr: [[number]]
    :param ind: int
    :return: [number]
    """
    ind_vals = list(map(lambda p: p[ind], arr))
    ind = ind_vals.index(min(ind_vals))
    return arr[ind]

def find_max(arr, ind):
  """
  given a list of lists with numberic entries
  returns the element with the highest value at index ind
  :param arr: [[number]]
  :param ind: int
  :return: [number]
  """
  ind_vals = list(map(lambda p: p[ind], arr))
  ind = ind_vals.index(max(ind_vals))
  return arr[ind]

def find_with_pred(arr, pred):
  for el in arr:
    if pred(el):
      return el
  return None

def point_on_unit_circle(t):
  if t != -1:
    t_sqr = pow(t, 2)
    x = (1 - t_sqr) / (1 + t_sqr)
    y = (2 * t) / (1 + t_sqr)
    return [x, y]
  else:
    raise ArithmeticError("t can not be -1")

def point_on_proj_unit_circle(t):
  t_sqr = pow(t, 2)
  if t_sqr != -1:
    x = t / (1 + t_sqr)
    y = 1 / (1 + t_sqr)
    #print(x, y)
    return [x, y]
  else:
    raise ArithmeticError("t can not be -1")

def point_on_circle(t, r):
  t_sqr = pow(t, 2)
  x = (r - t_sqr) / (r + t_sqr)
  y = ((r * 2) * t) / (r + t_sqr)
  return [x, y]

def vector_add(v1, v2):
  a, b, c, d = v1[0], v1[1], v2[0], v2[1]
  return [a + c, b + d]

def vector_mul(v1, v2):
  a, b, c, d = v1[0], v1[1], v2[0], v2[1]
  return [a * c, b * d]

def vector_scale(num, v):
  #print(v)
  a, b = v[0], v[1]
  return [a * num, b * num]

def pos_quarter_rot(v):
    a, b = v[0], v[1]
    return [-b, a]

def cross_product(v, w):
 a, b, c, d = v[0], v[1], w[0], w[1]
 return (a * d) - (b * c)

def vector_quadrance(v):
  a, b = v[0], v[1]
  return cross_product([a, b], [-b, a])

def vector_spread(v1, v2):
  a, b, c, d = v1[0], v1[1], v2[0], v2[1]
  nom = ((a * d) - (b * c)) ** 2
  dnom = (pow(a, 2) + pow(b, 2)) * (pow(c, 2) + pow(d, 2))
  return Fraction(nom, dnom)

def spread_of_lines(l1, l2):
 a1, b1, c1, a2, b2, c2 = l1[0], l1[1], l1[2], l2[0], l2[1], l2[2]
 nom = ((a1 * b2) - (a2 * b1)) ** 2
 dnom = (pow(a1, 2) + pow(b1, 2)) * (pow(a2, 2) + pow(b2, 2))
 return Fraction(nom, dnom)

def dot_product(v1, v2, c="blue"):
  a, b, c, d = v1[0], v1[1], v2[0], v2[1]
  if c == "blue":
    return (a * c) + (b * d)
  elif c == "red":
    return (a * c) - (b * d)
  elif c == "green":
    return (a * d) + (b * c)
  else:
    raise ArithmeticError("argument of c must be ")

def quadrance_of_distance(p1, p2):
  x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
  return (abs(x1 - x2) ** 2) + (abs(y1 - y2) ** 2)

def midpoint(p1, p2):
  x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
  nx = (x1 + x2) / 2
  ny = (y1 + y2) / 2
  return [nx, ny]

def line_through_points(p1, p2):
  x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
  a = y1 - y2
  b = x2 - x1
  c = (x1*y2) - (x2*y1)
  return [a, b, c]

def intersection_point(l1, l2):
  """
  If the lines l1 and l2 are not parallel, then there
  is a unique point A ≡ l1 l2 which lies on them both.
  If l1 ≡ ha1 : b1 : c1 i and l2 ≡ a2 : b2 : c2  then:
  x = b1 c2 − b2 c1 / a1 b2 − a2 b1
  y = c1 a2 − c2 a1 / a1 b2 − a2 b1
  """
  a1, b1, c1, a2, b2, c2 = l1[0], l1[1], l1[2], l2[0], l2[1], l2[2]
  dnom = (a1 * b2) - (a2 * b1)
  #print(dnom)
  x_nom = (b1 * c2) - (b2 * c1)
  #print(x_nom)
  y_nom = (c1 * a2) - (c2 * a1)
  #print(y_nom)
  x = x_nom / dnom
  y = y_nom / dnom
  return [x, y]

def line_has_point(l, p):
    """
    The point p ≡ [x, y] lies on the line l ≡ <a : b : c>,
    or equivalently the line l passes through the point p,
    precisely when ax + by + c = 0.
    """
    a, b, c, x, y = l[0], l[1], l[2], p[0], p[1]
    return (a * x) + (b * y) + c == 0

def is_collinear(p1, p2, p3):
    """
    The points [x1 , y1 ], [x2 , y2 ] and [x3 , y3 ] are collinear precisely when
    x1 y2 − x1 y3 + x2 y3 − x3 y2 + x3 y1 − x2 y1 = 0.
    """
    x1, y1, x2, y2, x3, y3 = p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]
    return (x1 * y2) - (x1 * y3) + (x2 * y3) - (x3 * y2) + (x3 * y1) - (x2 * y1) == 0

def is_null_line(l):
  a, b, c = l[0], l[1], l[2]
  return (a ** 2) + (b ** 2) == 0

def forms_null_line(p1, p2):
    #check if p1 and p2 are distinct
    if len(set([p1, p2])) != 2:
        raise ArithmeticError("arguments must be distinct")
    return quadrance_of_distance(p1, p2) == 0

def cross(l1, l2):
  if is_null_line(l1) or is_null_line(l2):
    return None
  elif lines_are_perp(l1, l2):
    return 0
  else:
    a1, b1, a2, b2 = l1[0], l1[1], l2[0], l2[1]
    nom = pow((a1*a2 + b1*b2), 2)
    dnom = (pow(a1, 2) + pow(b1, 2)) * (pow(a2, 2) + pow(b2, 2))
    return nom / dnom

def twist(l1, l2):
  if lines_are_perp(l1, l2):
    return None
  else:
    a1, b1, a2, b2 = l1[0], l1[1], l2[0], l2[1]
    nom = ((a1*b2) - (a2*b1)) ** 2
    dnom = ((a1*a2) + (b1*b2)) ** 2
    return nom / dnom

def forms_right_triangle(p1, p2, p3):
  l1 = line_through_points(p1, p2)
  l2 = line_through_points(p1, p3)
  q1 = quadrance_of_distance(p2, p3)
  q3 = quadrance_of_distance(p1, p3)
  return spread_of_lines(l1, l2) == q1 / q3

def spread_ratio(p1, p2, p3):
  if forms_right_triangle(p1, p2, p3):
    return quadrance_of_distance(p2,p3) / quadrance_of_distance(p1, p3)
  else:
    return None

def cross_ratio(p1, p2, p3):
  if forms_right_triangle(p1, p2, p3):
    l1 = line_through_points(p1, p2)
    l2 = line_through_points(p1, p3)
    q2 = quadrance_of_distance(p1, p2)
    q3 = quadrance_of_distance(p1, p2)
    res1 = cross(l1, l2)
    res2 = q2 / q3
    return res2 if res1 == res2 else None
  else:
    return None

def lines_are_perp(l1, l2):
  a1, b1, a2, b2 = l1[0], l1[1], l2[0], l2[1]
  return (a1 * a2) + (b1 * b2) == 0

def lines_is_concurrent(lines):
  if len(lines) >= 3:
    p = intersection_point(lines[0], lines[1])
    return all(map(lambda l: line_has_point(l, p), lines))
  else:
    raise ArithmeticError("expected argument with a list of three lines")

def alt(p, l):
  """
  For any point p ≡ [x, y] and any line l ≡ <a : b : c>
  there is a unique line n, called the altitude from p to l,
  which passes through p and is perpendicular to l, namely:
        n = <−b : a : bx − ay>
  """
  a, b, c, x, y = l[0], l[1], l[2], p[0], p[1]
  na = -b
  nb = a
  nc = (b * x) - (a * y)
  return [na, nb, nc]

def perpendicular_bisector(p1, p2):
  mp = midpoint(p1, p2)
  side = line_through_points(p1, p2)
  ret = alt(mp, side)
  return ret

def circumcenter_triangle(p1, p2, p3):
  bs1 = perpendicular_bisector(p1, p2)
  bs2 = perpendicular_bisector(p2, p3)
  bs3 = perpendicular_bisector(p3, p1)
  it1 = intersection_point(bs1, bs2)
  it2 = intersection_point(bs2, bs3)
  it3 = intersection_point(bs3, bs1)
  #print(it1, it2, it3)
  return it1 if all([it1, it2, it3]) else None

def orthocenter_triangle(p1, p2, p3):
  alt1 = alt(p1, p2)
  alt2 = alt(p2, p3)
  alt3 = alt(p3, p1)
  i1 = intersection_point(alt1, alt2)
  i2 = intersection_point(alt2, alt3)
  i3 = intersection_point(alt3, alt1)
  return i1 if all([i1, i2, i3]) else None

def centeroid_triangle(p1, p2, p3):
  mp1, mp2, mp3 = midpoint(p1, p2), midpoint(p2, p3), midpoint(p3, p1)
  median1, median2, median3 = line_through_points(p1, mp2), line_through_points(p2, mp3), line_through_points(p3, mp1)
  it1 = intersection_point(median1, median2)
  it2 = intersection_point(median2, median3)
  it3 = intersection_point(median3, median1)
  return it1 if all([it1, it2, it3]) else None

def circumquadrance_triangle(p1, p2, p3, c=None):
  _c = circumcenter_triangle(p1, p2, p3) if c is None else c
  q1, q2, q3 = quadrance_of_distance(_c, p1), quadrance_of_distance(_c, p2), quadrance_of_distance(_c, p3)
  return q1 if all([q1, q2, q3]) else None

def signed_area_triangle(p1, p2, p3):
  m = np.matrix([p1 + [1], p2 + [1], p2 + [1]])
  return (1 / 2) * m

def triangle(p1, p2, p3):
  v1 = find_min([p1, p2, p3], 0)
  v3 = find_max([p1, p2, p3], 0)
  v2 = find_with_pred([p1, p2, p3], lambda p: p[0] > v1[0] and p[0] < v3[0])
  s1, s2, s3 = line_through_points(v1, v2), line_through_points(v2, v3), line_through_points(v3, v1)
  mp1, mp2, mp3 = midpoint(v1, v2), midpoint(v2, v3), midpoint(v3, v1)
  alt1, alt2, alt3 = alt(v1, s2), alt(v2, s3), alt(v3, s1)
  median1, median2, median3 = line_through_points(v1, mp2), line_through_points(v2, mp3), line_through_points(v3, mp1)
  ortho_center = orthocenter_triangle(v1, v2, v3)
  centeroid = centeroid_triangle(v1, v2, v3)
  circumcenter = circumcenter_triangle(v1, v2, v3)
  circumquadrance = circumquadrance_triangle(v1, v2, v3, circumcenter)

  ret = {
      "verticies": [v1, v2, v3],
      "sides": [s1, s2, s3],
      "midpoints": [mp1, mp2, mp3],
      "altitudes": [alt1, alt2, alt3],
      "medians": [median1, median2, median3],
      "ortho_center": ortho_center,
      "centeroid": centeroid,
      "circumcenter": circumcenter,
      "circumquadrance": circumquadrance
  }

def quadrea(p1, p2, p3):
  q1 = quadrance_of_distance(p2, p3)
  q2 = quadrance_of_distance(p1, p3)
  q3 = quadrance_of_distance(p1, p2)
  t1 = sum([q1, q2, q3]) ** 2
  t2 = 2 * sum([pow(q1, 2), pow(q2, 2), pow(q3, 2)])
  return t1 - t2

def wedge(v1,v2):
  e = dict()
  for i in range(len(v1)):
      for j in range(i + 1, len(v1)):
        k = "e"+str(i)+str(j)
        if i != 0:
          e[k] = (v1[i] * v2[j] - v1[j] * v2[i])
  return e, sum(e.values())

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

def lcm(a, b):
  return a * b / gcd(a, b)

def is_square(n):
  return math.sqrt(n) % 1 == 0

def harm_mean(A, B):
  nom = 2 * (A*B)
  dnom = A + B
  return nom / dnom

def arithmetic_mean(A, B):
  return (A + B) / 2

def gnomonic_number(n):
    return 1 if n == 1 else pow(n, 2) - pow(n - 1, 2)

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


