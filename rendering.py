import numpy as np
from math import sin, cos, tan, atan2, acos, pi
from num import is_even
from vector_utils import normalize_vector

def project_point_to_hyperplane(view_pt, target_pt, normal, d):
  eps = 1/(1<<16)
  v = np.asarray(view_pt, float)
  p = np.asarray(target_pt, float)
  n = np.asarray(normal, float)
  look_dir = v - p
  denom = n @ look_dir
  if abs(denom) < eps: return view_pt  #behind camera
  t = (d - n @ v) / denom
  return v + t * dir

def perspective(pos,target):
  forward = normalize_vector(target - pos)
  world_up = np.array([0, 0, 1], float)
  if abs(np.dot(forward, world_up)) > 0.99: world_up = np.array([0, 1, 0], float)
  right = normalize_vector(np.cross(forward, world_up))
  up = np.cross(right, forward)
  return {"forward":forward,"right":right,"up":up,"persp_pos":pos,"persp_target":target}


def world_to_perspective_point(persp_pos,persp_target,world_pt):
  eps = 1/(1 << 16)
  forward = normalize_vector(persp_target - persp_pos)
  world_up = np.array([0, 0, 1], float)
  if abs(np.dot(forward, world_up)) > 0.99: world_up = np.array([0, 1, 0], float)
  right = normalize_vector(np.cross(forward, world_up))
  up = np.cross(right, forward)
  q = world_pt - persp_pos
  x = np.dot(q,right)
  y = np.dot(q,up)
  z = np.dot(q,forward)
  return np.asarray([x,y,z])

# -----------------------------
# Curvilinear rendering
# -----------------------------

def direction_to_angles(p):
  """Return (theta, phi) where theta = angle from forward axis (z), phi = atan2(y,x)."""
  d = normalize_vector(np.array(p))
  theta = acos(max(-1.0, min(1.0, d[2])))
  phi = atan2(d[1], d[0])
  return np.array([theta, phi])

def fisheye_equidistant(p,f:float=1.0):
  theta, phi = direction_to_angles(p)
  r = f * theta
  return np.array([r * cos(phi), r * sin(phi)])

def fisheye_equisolid(p,f:float=1.0):
  theta, phi = direction_to_angles(p)
  r = 2.0 * f * sin(theta / 2.0)
  return np.array([r * cos(phi), r * sin(phi)])

def fisheye_stereographic(p,f:float=1.0):
  theta, phi = direction_to_angles(p)
  r = 2.0 * f * tan(theta / 2.0)
  return np.array([r * cos(phi), r * sin(phi)])

def orthographic_onto_disc(p,f:float=1.0):
  theta, phi = direction_to_angles(p)
  r = f * sin(theta)
  return np.array([r * cos(phi), r * sin(phi)])

def equirectangular(p):
  d = normalize_vector(p)
  theta = acos(max(-1.0, min(1.0, p[2])))  # 0..pi
  phi = atan2(d[1], d[0])  # -pi..pi
  u = (phi + pi) / (2.0 * pi)
  v = 1.0 - (theta / pi)
  return np.array([u, v])


# ---------------------------
#  Generic n→n−1 Projections
# ---------------------------

def central_project(dir_p, view_p, plane_n, zeros=0, eps=1e-12):
  dir_p  = np.array(dir_p, float)
  view_p = np.array(view_p, float)  
  d = dir_p - view_p
  if abs(d[-1]) < eps:
    return None

  t = (plane_n - view_p[-1]) / d[-1]
  s = view_p + t*d
  return np.concatenate([s[:-1], np.zeros(zeros)])


def stereo_project(p, zeros=0, eps=1e-12):
  p = np.array(p, float)
  if abs(1 - p[-1]) < eps:
    return None
  scale = 1/(1 - p[-1])
  xy = scale*p[:-1]
  return np.concatenate([xy, np.zeros(zeros)])


def persp_project(p, f=1.0, zeros=0):
    p = np.array(p, float)
    if p[-1] <= 0:
        return None

    xy = (f*p[:-1])/p[-1]
    return np.concatenate([xy, np.zeros(zeros)])


# ---------------------------
#  Projection chain n→3
# ---------------------------

def project_chain(p, method="perspective", f=1.0, plane_n=1.0):
  p = np.array(p, float)
  dim = len(p)
  if dim <= 3:
    raise ValueError("Need dim > 3")
    pts = [p[-i:] for i in range(dim, 2, -1)]
    out = []

  for zeros, sub in enumerate(pts):
      if method == "perspective":
          q = persp_project(sub, f=f, zeros=zeros)
      elif method == "stereo":
          q = stereo_project(sub, zeros=zeros)
      elif method == "central":
          q = central_project(sub, np.zeros_like(sub), plane_n, zeros=zeros)
      else:
          raise ValueError("unknown projection")  
      out.append(q)

  return np.stack(out)
