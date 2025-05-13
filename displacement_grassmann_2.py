import math_utils

"""
----------------------------------
grassmanns anticommutative law states that: ab = -ba
this means that if we have a reference order of two multiplicative terms a,b
then the product of the permuted order b,a is equal to the product of the reference order multiplied by its negative sign 
this rule also allow us to treat multiplication as a unary operator working on tupples instead as a binary operator in normal arthmetic.
"""
import numpy as np

def is_number(x):
  return isinstance(x,(int,float))

def is_two_tupple(x):
  return isinstance(x,tuple) and len(x) == 2

def is_two_tupple_numeric(x):
  return is_two_tupple(x) and is_number(x[0]) and is_number(x[1])

def is_two_tupple_product(x):
  return is_two_tupple_numeric(x) and abs(x[0]) == abs(x[1])

def grassmann_mul(tup):
    if is_two_tupple(tup):
      a,b = tup[0],tup[1]
      # Case 1: If both elements are tuples, handle recursively
      if is_two_tupple(a) and is_two_tupple(b):
        newa = a if is_two_tupple_product(a) else grassmann_mul(a)
        newb = b if is_two_tupple_product(b) else grassmann_mul(b)
        _p = grassmann_mul((newa[0],newb[0])) if all([is_two_tupple_product(newa),is_two_tupple_product(newb)]) else grassmann_mul((newa,newb))
        return _p
      # Case 2: If one element is a number and the other a tuple
      elif is_number(a) and is_two_tupple(b):
        return grassmann_mul(((a, -a), grassmann_mul(b)))

      elif is_two_tupple(a) and is_number(b):
        return grassmann_mul((grassmann_mul(a), (-b, b)))

      # Case 3: If both are numbers, apply Grassmann's rule
      elif is_number(a) and is_number(b):
        return (a*b,-b*a)

      else:
        raise ArithmeticError("Elements must be tuples or numbers")
    else:
      raise ArithmeticError("Argument must be a tuple with two elements")

def grassmann_add(a,b):
  if all([is_number(a),is_number(b)]):
    return a + b
  elif all([is_two_tupple_product(a),is_two_tupple_product(b)]):
    return grassmann_mul(a,b)
  else:
    return (a,b)

"""
I associate the concept of the elementary magnitude with the solution of a
simple problem by which I first arrived at this concept, and which seems to
me to be most appropriate for its natural development.
Problem. Let three elements al, a2, p, and a further element p be given;
one is then to find the element b2 satisfying the equation: [pa1] + [pa2] = [pb1] + [pb2]
Solution. If one shifts the terms on the left side to the right,
then since —[pa] = [ap] and [ap] + [pb] = [ab],
one has: [ab1] + [ab2] = 0
"""

def grassmann_example(scalar, coeffs):
  a1, a2, b1, p = coeffs[0], coeffs[1], coeffs[2], scalar
  pa1, pa2, pb1 = (p * a1), (p * a2), (p * b1)
  eq_left = pb1 + p
  eq_right = pa1 + pa2
  eq_diff = max(eq_left,eq_right) - min(eq_left,eq_right)
  solve = (eq_diff / p) + 1
  b2 = solve
  pb2 = (p*b2)
  ap1, ap2 = -pa1, -pa2
  ab1 = ap1 + pb1
  ab2 = ap2 + pb2
  print(ab1,ab2)
  return ab1 + ab2  # = 0

#print(grassmann_example(13, [27, 666, 360]))  # any arguments should return 0
#print(grassmann_example(7, [4, 13, 16]))  # any arguments should return 0
#print(grassmann_example(8, [7, 40, 45]))  # any arguments should return 0
#print(grassmann_example(20, [6, 14, 7]))  # any arguments should return 0

print(grassmann_mul((3,5)))






#print(grassmann_mul(()))

"""
I associate the concept of the elementary magnitude with the solution of a
simple problem by which I first arrived at this concept, and which seems to
me to be most appropriate for its natural development.
Problem. Let three elements al, a2, p, and a further element p be given;
one is then to find the element b2 satisfying the equation: [pa1] + [pa2] = [pb1] + [pb2]
Solution. If one shifts the terms on the left side to the right,
then since —[pa] = [ap] and [ap] + [pb] = [ab],
one has: [ab1] + [ab2] = 0
"""
#def grassmann_example(val,scalar,coeffs):
#  a1,a2,b1,b2 = coeffs[0],coeffs[1],coeffs[2],val
#  p = scalar
# pa1,pa2,pb1,pb2 = (p*a1),(p*a2),(p*b1),(p*b2)
#  eq1 = pb1 + pb2
#  eq2 = pa1 + pa2
#  eq = abs(eq1-eq2)
#  ap1,ap2 = -pa1,-pa2
#  ab1 = ap1 + pb1
#  ab2 = ap2 + p
#  #solve = (eq / p) + 1
#  print("ab1+ab2=",ab1+ab2)
#grassmann_example(19,7,[2,6,7])

#pa2 = grassmann_mul((p,a2))[0] #12
#ap2 = pa2[1] #-12
#pb1 = grassmann_mul((p,b1))[0]
#correct = grassmann_add(pa1,pa2) # 22
#b2 = (correct - pb) / p
#pb1, pb2 = pb, (p * b2)
#solve = (pa1 + pa2) - (pb1 + pb2)

"""
def angle_between_displacements(a,b):
  return [a[0]/b[0],a[1]/b[1]]

def a_to_b(a,b):
  a_dir,b_dir = a[1], b[1]
  dir_diff = abs(a_dir-b_dir)
  mag = a[0] # assuming a and b have equal mag
  n = mag * dir_diff
  ang1 = a[1] / b[1]
  ang2 = b[1] / a[1]
  n1 = mag / ang1
  n2 = mag / ang2
  res = ang1 * n
  #el = b[1] / a[1]
  print(res)
  #l = a[0] + (b[0] - a[0])


print(a_to_b([4,6],[4,9]))
#print(angle_between_displacements([4,6],[4,9]))


def a_from_b(a,b):
  return (a/b) * b

def b_from_a(a,b):
  return (b/a) * a



def disp(a,b,n):
  if n > 0:
      _b = -b if n % 2 == 0 else b
      return [a*n,_b]
  else:
    raise ArithmeticError("n must be a postive integer > 0")


#test = [disp(3,6,i) for i in range(10)]

def is_number(x):
    return True if type(x) == 'float' or 'int' else False

def is_non_zero_number(x):
    return is_number(x) and x != 0


class grassmann_tupple:

    def __init__(self,tup):
        if isinstance(tup,list) and len(tup) == 2:
            self.tup = tup
        else:
            raise ValueError("grassmann must be a tupple of two values")

    def is_point(self):
        return all([is_number(self.tup[0]),self.tup[1] == 0])

    def is_line(self):
        return all([self.tup[0] == 0,is_number(self.tup[1])])

    def is_line_seg(self):
        return all([is_non_zero_number(self.tup[0]),is_non_zero_number(self.tup[1])])

    def get_line(self):
      if self.is_line():
          return self
      else:



    def area(self):
      if self.is_line():

      else:





def is_segment(tup):





def get_fund(disp):
  facts = math_utils.factors(disp[0])
  if len(facts) > 1:
    return [facts[1],disp[1]]
  else:
    return [1,disp[1]]

def disp_n(disp,n):
    return [disp[0] * n,disp[1]]


def evolution_1(disp):
  fund = get_fund(disp)
  n = 0
  m = fund[0]
  d = fund[1]
  nxt = 0
  while True:
    yield [[nxt,d],[nxt,d + -d]]
    n += 1
    nxt += m

def evolution_2(disp_list):
   evolutions = list(map(lambda d: evolution_1(d),list(disp_list)))
   n = 0
   nxt = next(evolutions[0])
   while True:
       yield nxt
       n = (n + 1) % len(evolutions)
       nxt = next(evolutions[n])

test_2 = evolution_2([[2,6],[1,8],[8,4],[5,4],[3,3]])
test_2_out = [next(test_2) for _ in range(20)]
print(test_2_out)
"""
