import math
import numpy as np


def point_quadrence(p1, p2, g="blue"):
  terms = []
  for i in len(p1):
    terms.append((p1[i] - p2[i]) ** 2)
  if g == "blue":
    return sum(terms)
  elif g == "red":
    res = terms[0]
    for j in range(1,len(terms)):
      res -= terms[j]
    return res
  elif g == "green":
    return 2 * math.prod(terms)
  else:
    raise IOError("expects argument of g to be: (blue | red | green)")

def spread(l1,l2,g="blue"):
  a1 = l1[0]
  a2 = l2[0]
  b1 = l1[1]
  b2 = l2[1]
  sqr_a1 = a1 ** 2
  sqr_a2 = a2 ** 2
  sqr_b1 = b1 ** 2
  sqr_b2 = b2 ** 2
  if(g == "blue"):
    num = -(((a1 * b2) - (a2 * b1)) ** 2)
    dnum = (sqr_a1 - sqr_b1) * (sqr_a2 - sqr_b2)
    return num / dnum
  elif(g == "red"):
    num =  (((a1 * b2) - (a2 * b1)) ** 2)
    dnum = (sqr_a1 + sqr_b1) * (sqr_a2 + sqr_b2)
    return num / dnum
  elif(g == "green"):
    return 1
  else:
    raise IOError("spread function expects argument of g to be a string blue,red or green")

def cross(l1,l2):
  a1 = l1[0]
  a2 = l2[0]
  b1 = l1[1]
  b2 = l2[1]
  sqr_a1 = a1 ** 2
  sqr_a2 = a2 ** 2
  sqr_b1 = b1 ** 2
  sqr_b2 = b2 ** 2
  num = ((a1 * a2) - (b1 * b2)) ** 2
  dnum = (sqr_a1 - sqr_b1) * (sqr_a2 - sqr_b2)
  return num / dnum

def Point(arr):
  ax_keys = ["x","y","z","w","q","p"]
  dim = len(arr)
  ret = dict(
    dim=dim,
    array=np.array(arr),
  )
  for i in range(dim):
    ret[ax_keys[i]] = arr[i]
  #print(ret)
  return ret

def dot(v1, v2, g="blue"):
  if g == "blue":
    terms = [v1[i] * v2[i] for i in range(len(v1))]
    return sum(terms)
  if g == "red":
    terms = [v1[i] * v2[i] for i in range(len(v1))]
    res = terms[0]
    for j in range(1,len(terms)):
      ret -= terms[i]
    return res
  elif g == "green":
    terms = []
    for l in range(1, len(v1)):
      a1 = v1[l - 1]
      a2 = v1[l]

def is_perpendicular(v1, v2, g="blue"):
  return dot(v1, v2, g) == 0

def vector_quadrance(v,g="blue"):
  return dot(v, v, "blue")


def join_vector(p1,p2):
  return [abs(t[0] - t[1]) for t in zip(p1,p2)]
  #return [p1[i] - p2[i] for i in range(len(p1))]


def quadrance_between(p1, p2):
 jv = join_vector(p1,p2)
 return vecor_quadrance(jv)

def vector_scale(scalar,v):
  return np.array([scalar * v[i] for i in range(len(v))])


def vector_add(*args):
  if isinstance(vlist,numpy.ndarray):
    print(np.shape(vlist))
    return vlist.sum(0)
  elif isinstance(vlist, list):
    return np.array(vlist).sum(0)
  else:
    raise TypeError("argument must be an array of vectors")


def vector_prod(vlist):
  if isinstance(vlist,numpy.ndarray):
    print(np.shape(vlist))
    return vlist.prod(0)
  elif isinstance(vlist, list):
    return np.array(vlist).prod(0)
  else:
    raise TypeError("argument must be an array of vectors")

def vector_proj(v1,v2):
  scalar = dot(v1, v2) / dot(v1, v1)
  return vector_scale(scalar,v1)

def is_null(v, g="red"):
  return quadrance(v, g) == 0


def mid_point(p1, p2):
  if np.shape(p1) == np.shape(p2) and len(np.shape(p1)) == len(np.shape(p2)):
    dim = len(p1)
    ret = []
    for i in range(len(p1)):
      ret.append((p1[i] + p2[i]) / 2)
    return np.array(ret)

def polar_angle_from_altitude(y):
  # Calculate the polar angle theta from the altitude y
  theta = math.acos(y)
  return theta


def point_on_the_unit_circle(y):
  x = math.sqrt(1 - pow(y, 2))
  return {
    "point": [x, y],
    "reflected_point": [-x, y]
  }


def point_on_circle(y, radius):
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















