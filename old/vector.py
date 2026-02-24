import numpy as np
import math
from SO import

from matplotlib.image import pil_to_array

dir_index_map = {"east":0,"west":1,"front":2,"back":3,"north":4,"south":5}

def sphere_to_cart_coord(arr):
  r,phi,theta = arr[0],deg2rad(wrap_deg_360(arr[1])),deg2rad(wrap_deg_360(arr[2]))
  x = r * np.sin(phi) * np.cos(theta)
  y = r * np.sin(phi) * np.sin(theta)
  z = r * np.cos(phi)
  return np.array([x, y, z])

def value_index(arr,val):
  if val in arr: return arr.index(val)
  else: raise ValueError("val not in arr")



def axis_vector(direction,val):
  if   direction == "north":  return np.array([0.0,0.0, val])
  elif direction == "south":  return np.array([0.0,0.0,-val])
  elif direction == "east":   return np.array([val, 0.0,0.0])
  elif direction == "west":   return np.array([-val,0.0,0.0])
  elif direction == "front":  return np.array([0.0,val,0.0 ])
  elif direction == "back":   return np.array([0.0,-val,0.0])
  else: raise ValueError("direction must be up|down|right|left|front|back")



def dir_index(direction):return dir_index_map[direction]
  #return ["xy","xy","xz","xz","yz","yz"][i]

def dir_plane(direction):
  return ["xy","xy","xz","xz","yz","yz"][dir_index(direction)]


def dir_vals(p):
  px,py,pz = p[0],p[1],p[2]
  apx,apy,apz = -px,-py,-pz
  return {
    "north":max(pz,apz),
    "south":min(pz,apz),
    "east" :max(px,apx),
    "west" :min(px,apx),
    "front":max(py,apy),
    "back" :min(py,apy)
  }

def dir_coords(dv):
  ret = {
    "efn" :[dv["east"],dv["front"],dv["north"]],
    "efs" :[dv["east"],dv["front"],dv["south"]],
    "ebn" :[dv["east"], dv["back"],dv["north"]],
    "ebs" :[dv["east"], dv["back"],dv["south"]],
    "wfn": [dv["west"],dv["front"],dv["north"]],
    "wfs": [dv["west"],dv["front"],dv["south"]],
    "wbn": [dv["west"],dv["back"], dv["north"]],
    "wbs": [dv["west"],dv["back"], dv["south"]],
  }
  return ret



def antipodes(p):
  x,y,z = p
  return [
      [x,y,z],[-x,y,z],[x,-y,z],[-x,-y,z],
      [x,y,-z],[-x,y,-z],[x,-y,-z],[-x,-y,-z]
  ]

def orthogonal_ref(plane):
    return [[0,0,1],[0,1,0],[1,0,0]][["xy","xz","yz"].index(plane)]

def othogonal_u(p,direction):
  ref = orthogonal_ref(direction)
  return normalize(np.cross(ref,p))

def orthogonal_v(p,direction):
  u = othogonal_u(p,direction)
  return normalize(np.cross(p,u))

def normalize(v):
  n = np.linalg.norm(v)
  return v if n < 1e-12 else v / n


def Rz(yaw):
  c, s = math.cos(yaw), math.sin(yaw)
  return np.array([[c, -s, 0],
                   [s, c, 0],
                   [0, 0, 1]])

def Ry(pitch):
  c, s = math.cos(pitch), math.sin(pitch)
  return np.array([[c, 0, s],
                   [0, 1, 0],
                   [-s, 0, c]])

def Rx(roll):
  c, s = math.cos(roll), math.sin(roll)
  return np.array([[1, 0, 0],
                   [0, c, -s],
                   [0, s, c]])


def rotation_matrix(yaw,pitch,roll):
  return Rz(yaw) @ Ry(pitch) @ Rx(roll)


def pole(rot_mtx,direction,radius): return normalize(rot_mtx @ axis_vector(direction,radius))
def basis(rot_mtx,direction): return normalize(rot_mtx @ axis_vector(direction,1))
def rotate_points(points,rot_mtx): return np.dot(points, rot_mtx.T)


def orientation(sphere_radius,yaw,pitch,roll):
  rot_mtx     = rotation_matrix(yaw,pitch,roll)
  pole_north  = pole(rot_mtx=rot_mtx,direction="north" ,radius=sphere_radius)
  pole_south  = -pole_north
  pole_east   = pole(rot_mtx=rot_mtx,direction="east"  ,radius=sphere_radius)
  pole_west   = -pole_east
  pole_front  = pole(rot_mtx=rot_mtx,direction="front"  ,radius=sphere_radius)
  pole_back   = -pole_front
  basis_north    = basis(rot_mtx=rot_mtx,direction="north")
  basis_south  = -basis_north
  basis_east = basis(rot_mtx=rot_mtx,direction="east")
  basis_west   = -basis_east
  basis_front = basis(rot_mtx=rot_mtx,direction="front")
  basis_back  = -basis_front
  return {
      "rot_vals"   :[yaw,pitch,roll],
      "pole_north" : pole_north,
      "pole_south" : pole_south,
      "pole_east"  : pole_east,
      "pole_west"  : pole_west,
      "pole_front" : pole_front,
      "pole_back"  : pole_back,
      "basis_front": basis_front,
      "basis_back" : basis_back,
      "basis_east" : basis_east,
      "basis_west" : basis_west,
      "basis_north": basis_north,
      "basis_south" :basis_south,
      "rot_mtx"    : rot_mtx
    }


def orient_with_up(forward, up=np.array([0, 0, 1])):
  """
  Given a forward vector and a desired 'up' vector,
  return a dict with the orthonormal basis:
  {frwd, back, right, left, up, down}.
  """
  frwd = normalize(forward)
  up = normalize(up)

  # Handle case when forward and up are parallel
  if np.abs(np.dot(frwd, up)) > 0.9999:
    up = np.array([1, 0, 0]) if np.abs(frwd[0]) < 0.9 else np.array([0, 1, 0])
    up = normalize(up)

  right = normalize(np.cross(frwd, up))
  true_up = normalize(np.cross(right, frwd))

  return {
    "frwd": frwd,
    "back": -frwd,
    "right": right,
    "left": -right,
    "up": true_up,
    "down": -true_up
  }


def xy_vector(ang):
  a = math.radians(ang)
  x,y,z = math.sin(a),math.cos(a),0
  return np.array([x,y,z])

def xz_vector(ang):
  a = math.radians(ang)
  x, y, z = math.sin(a), 0, math.cos(a)
  return np.array([x,y,z])

def yz_vector(ang):
  a = math.radians(ang)
  x,y,z = 0,math.sin(a),math.cos(a)
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


