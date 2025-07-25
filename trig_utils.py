from num_utils import is_num, is_num_between
import numpy as np
import math
import cmath

def ang_theta(num):
  if is_num_between(num,-np.pi,np.pi): return num
  elif is_num_between(num,-180,180): return np.radians(num)
  else: raise ValueError("must be an angle between 0-pi")

def ang_phi(num):
  if is_num_between(num,-np.pi/2,np.pi/2): return num
  elif is_num_between(num,-90,90): return np.radians(num)
  else: raise ValueError("must be an angle between 0-tau")


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




def rotate_about(z, center, angle):
    return (z - center) * cmath.exp(1j * angle) + center


def stereo_proj(point,sphere_radius):
  x,y = point[0],point[1]
  r = sphere_radius
  ret_x = 2*x / (pow(x,2)+pow(y,2)+r)
  ret_y = 2*y / (pow(x,2)+pow(y,2)+r)
  ret_z = (pow(x,2)+pow(y,2)-r) / (pow(x,2)+pow(y,2)+r)
  print(ret_z)
  return [ret_x,ret_y,ret_z]

def sphere_to_complex_plane(sphere_point):
    x,y,z = sphere_point[0],sphere_point[1],sphere_point[2]
    return complex(x/(1-z),y/(1-z))

def sphere_to_plane(sphere_point):
    x,y,z = sphere_point[0],sphere_point[1],sphere_point[2]
    return [x/(1-z),y/(1-z)]


def sphere_rot(point,xy_ang,yz_ang,xz_ang):
    x,y = point[0],point[1]
    sphere_point = stereo_proj(point,1)
    rot = lambda z, ang: z * cmath.exp(1j * ang)
    #center_rot = lambda z, ang, center: (z - center) * cmath.exp(1j * ang) + center
    to_euclidian = lambda z: [z.real,z.imag]
    xy_rot = rot(complex(x,y),xy_ang)
