from numbers import Number
from fractions import Fraction
import math

def det(m):
  dim = len(m)
  t1 = []
  t2 = []
  for it1 in range(dim):
    t1.append(math.prod([m[j1][(it1 + j1) % dim] for j1 in range(dim)]))
  for it2 in list(reversed(range(dim))):
    t2.append(math.prod([m[j2][(it2 - j2) % dim] for j2 in range(dim)]))
  return sum(t1) - sum(t2)


def rational_div(nom, dnom):
    if nom // dnom - nom / dnom == 0:
        return nom // dnom
    else:
        return Fraction(nom, dnom)


def lies_on_line(A:[Number], l:[Number]):
    """
    The point A ≡ [x, y] lies on the line l ≡ <a : b : c>,
    or equivalently the line l passes through the point A,
    precisely when ax + by + c = 0.
    """
    a, b, c, x, y = l[0], l[1], l[2], A[0], A[1]
    return (a * x) + (b * y) + c == 0


def is_collinear(p1: [Number], p2: [Number], p3: [Number]):
    """
    The points [x1 , y1 ], [x2 , y2 ] and [x3 , y3 ] are collinear precisely when
    x1 y2 − x1 y3 + x2 y3 − x3 y2 + x3 y1 − x2 y1 = 0.
    """
    x1, y1, x2, y2, x3, y3 = p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]
    return (x1 * y2) - (x1 * y3) + (x2 * y3) - (x3 * y2) + (x3 * y1) - (x2 * y1) == 0


def quadrance_of_distance(p1: [Number], p2: [Number]):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    return (abs(x1 - x2) ** 2) + (abs(y1 - y2) ** 2)


def midpoint(p1: [Number], p2: [Number]):
    x1, y1, x2, y2 = p1[0], p1[1], p2[0], p2[1]
    nx = x1 + x2 / 2
    ny = y1 + y2 / 2
    return tuple([nx, ny])


def forms_null_line(p1: [Number], p2: [Number]):
    #check if p1 and p2 are distinct
    if len({p1, p2}) != 2:
        raise ArithmeticError("arguments must be distinct")
    return quadrance_of_distance(p1, p2) == 0

def is_null_line(l: [Number]):
  a, b, c = l[0], l[1], l[2]
  return (a ** 2) + (b ** 2) == 0


def is_central_line(l: [Number]):
    return l[2] == 0


def pass_through_point(l: [Number], p: [Number]):
    a, b, c, x, y = l[0], l[1], l[2], p[0], p[1]
    return (a * x) + (b * y) + c == 0

def abs_diff(vec):
    return abs(vec[0] - vec[1])


def intersection_point(l1, l2):
  """
  if two lines are not parallel, there exist a unique point which lies on them both where:
  x = ((b1 * c2) − (b2 * c1)) / ((a1 * b2) − (a2 * b1))
  y = ((c1 * a2) - (c2 * a1)) / ((a1 * b2) - (a2 * b1))
  """
  pl1 = abs(l1[0] - l1[1])
  pl2 = abs(l2[0] - l2[1])
  if pl1 != pl2:
    xnom = (l1.b * l2.c) - (l2.b * l1.c)
    xdnom = (l1.a * l2.b) - (l2.a * l1.b)
    ynom = (l1.c * l2.a) - (l2.c * l1.a)
    ydnom = (l1.a * l2.b) - (l2.a * l1.b)
    x = xnom / xdnom
    y = ynom / ydnom
    return [x, y]

  else:
    return None

def is_perp(l1, l2):
  a1, b1, a2, b2 = l1[0], l1[1], l2[0], l2[1]
  return (a1 * a2) + (b1 * b2) == 0


def lines_is_concurrent(lines):
  if len(lines) >= 3:
    p = intersection_point(lines[0], lines[1])
    return all(map(lambda l: pass_through_point(l, p), lines))

def Alt_to_line(A: [Number], l: [Number]):
  """
  For any point A ≡ [x, y] and any line l ≡ <ha : b : c>
  there is a unique line n, called the altitude from A to l,
  which passes through A and is perpendicular to l, namely:
        n = <−b : a : bx − ay>
  """
  na = l[0]
  nb = l[0]
  nc = (l[1] * A[0]) - (l[0] * A[0])
  return [na, nb, nc]

def foot_of_alt(A: [Number], l: [Number]):
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
    a, b, c, x, y = l[0], l[1], l[2], A[0], A[1]
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














