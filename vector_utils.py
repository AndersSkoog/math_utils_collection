import numpy as np
from itertools import combinations
from math import cos, sin, radians

def axis_pairs(dim): return list(combinations(range(dim),2))

def normalize_vector(vec):
  vec = np.asarray(vec)
  n = np.linalg.norm(vec)
  if n == 0: return vec
  return vec / n

def perpendicular_vector(v, ref=np.array([0, 0, 1])):
  """Return a vector perpendicular to v (cross with reference axis)."""
  v = np.array(v)
  perp = np.cross(v, ref)
  if np.linalg.norm(perp) == 0:
    # v parallel to ref, use another reference
    perp = np.cross(v, np.array([0, 1, 0]))
  return normalize_vector(perp)


def unit(v):
    return v / np.linalg.norm(v)

def orthonormal_frame(v):
    v = unit(v)
    tmp = np.array([1,0,0]) if abs(v[0]) < 0.9 else np.array([0,1,0])
    n1 = unit(np.cross(v, tmp))
    n2 = np.cross(v, n1)
    return n1, n2

def antipodes(p):
  x,y,z = p
  return [
      [x,y,z],[-x,y,z],[x,-y,z],[-x,-y,z],
      [x,y,-z],[-x,y,-z],[x,-y,-z],[-x,-y,-z]
  ]

def orthogonal_ref(plane):
    return [[0,0,1],[0,1,0],[1,0,0]][["xy","xz","yz"].index(plane)]

def orthonormal_u(p,direction):
  ref = orthogonal_ref(direction)
  return normalize_vector(np.cross(ref,p))

def orthonormal_v(p,direction):
  u = orthonormal_u(p,direction)
  return normalize_vector(np.cross(p,u))

def xy_vector(ang):
  a = radians(ang)
  x,y,z = sin(a),cos(a),0
  return np.array([x,y,z])

def xz_vector(ang):
  a = radians(ang)
  x, y, z = sin(a), 0, cos(a)
  return np.array([x,y,z])

def yz_vector(ang):
  a = radians(ang)
  x,y,z = 0,sin(a),cos(a)
  return np.array([x,y,z])

def combine_orientation(vec_xy,vec_yz,scale=1):
  A = np.array(vec_xy)
  Ax,Ay,Az = A[0],A[1],A[2]
  B = np.array(vec_yz)
  Bx,By,Bz = B[0],B[1],B[2]
  Cx = 1.0
  Cy = (Ay/ Ax) * Cx
  Cz = (Bz / Bx) * Cx
  return np.array([Cx,Cy,Cz]) * scale