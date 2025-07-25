from fractions import Fraction
from coll_utils import arr_eq, find_min, find_max, find_with_pred
import numpy as np

normalize_vector = lambda v: v / np.linalg.norm(v)

def lies_on_line(p,l):
    """
    The point A ≡ [x, y] lies on the line l ≡ <a : b : c>,
    or equivalently the line l passes through the point A,
    precisely when ax + by + c = 0.
    """
    a, b, c, x, y = l[0], l[1], l[2], p[0], p[1]
    return (a * x) + (b * y) + c == 0

def is_central_line(l): return l[2] == 0

def pass_through_point(l, p):
    a, b, c, x, y = l[0], l[1], l[2], p[0], p[1]
    return (a * x) + (b * y) + c == 0

def is_perp(l1, l2):
  a1, b1, a2, b2 = l1[0], l1[1], l2[0], l2[1]
  return (a1 * a2) + (b1 * b2) == 0

def lines_concurrent(lines):
  if len(lines) >= 3:
    p = intersection_point(lines[0], lines[1])
    return all(map(lambda l: pass_through_point(l, p), lines))

def Alt_to_line(p, l):
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

def foot_of_alt(p, l):
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
    nx = ((sqr_b * x) - (a * b * y) - (a * c)) // dnom
    ny = (-(a * b * x) + (sqr_a * y) - (b * c)) // dnom
    return [nx, ny]

def spread(l1, l2):
  if is_null_line(l1) and is_null_line(l2):
    a1, b1, c1, a2, b2, c2 = l1[0], l1[1], l1[2], l2[0], l2[1], l2[2]
    nom = ((a1 * b2) - (a2 * b1)) ** 2
    dnom = (pow(a1, 2) + pow(b1, 2)) * (pow(a2, 2) + pow(b2, 2))
    return nom / dnom
  else: return None

def vector_add(v1, v2):
    a, b, c, d = v1[0], v1[1], v2[0], v2[1]
    return [a + c, b + d]

def vector_mul(v1, v2):
    a, b, c, d = v1[0], v1[1], v2[0], v2[1]
    return [a * c, b * d]

def vector_scale(num, v):
    # print(v)
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
    return


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
    c = (x1 * y2) - (x2 * y1)
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
    # print(dnom)
    x_nom = (b1 * c2) - (b2 * c1)
    # print(x_nom)
    y_nom = (c1 * a2) - (c2 * a1)
    # print(y_nom)
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
    # check if p1 and p2 are distinct
    if arr_eq(p1,p2): raise ArithmeticError("arguments must be distinct")
    return quadrance_of_distance(p1, p2) == 0


def cross(l1, l2):
    if is_null_line(l1) or is_null_line(l2):
        return None
    elif lines_are_perp(l1, l2):
        return 0
    else:
        a1, b1, a2, b2 = l1[0], l1[1], l2[0], l2[1]
        nom = pow((a1 * a2 + b1 * b2), 2)
        dnom = (pow(a1, 2) + pow(b1, 2)) * (pow(a2, 2) + pow(b2, 2))
        return nom / dnom


def twist(l1, l2):
    if lines_are_perp(l1, l2):
        return None
    else:
        a1, b1, a2, b2 = l1[0], l1[1], l2[0], l2[1]
        nom = ((a1 * b2) - (a2 * b1)) ** 2
        dnom = ((a1 * a2) + (b1 * b2)) ** 2
        return nom / dnom


def forms_right_triangle(p1, p2, p3):
    l1 = line_through_points(p1, p2)
    l2 = line_through_points(p1, p3)
    q1 = quadrance_of_distance(p2, p3)
    q3 = quadrance_of_distance(p1, p3)
    return spread_of_lines(l1, l2) == q1 / q3


def spread_ratio(p1, p2, p3):
    if forms_right_triangle(p1, p2, p3):
        return quadrance_of_distance(p2, p3) / quadrance_of_distance(p1, p3)
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
    # print(it1, it2, it3)
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
    v2 = find_with_pred([p1, p2, p3], lambda p: v1[0] < p[0] < v3[0])
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
