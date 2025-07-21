from math_utils import is_num_between, is_num, normalize_vector, tau, ang_theta, ang_phi
import numpy as np
import math


def sphere_coord(arr,center):
  r,theta,phi = is_num(arr[0]), ang_theta(arr[1]), ang_phi(arr[2])
  x = center[0] + np.sin(phi) * np.cos(theta)
  y = center[1] + np.sin(phi) * np.sin(theta)
  z = center[2] + np.cos(phi)
  return np.array([x, y, z])

def orthogonal_vectors(sc):
  ref = np.array([0,1,0])
  if np.allclose(sc, ref) or np.allclose(sc, -ref): ref = np.array([0, 0, 1])
  u = normalize_vector(np.cross(ref,sc))
  v = normalize_vector(np.cross(sc,u))
  return u,v,ref

def stereographic_project(p, north_pole=np.array([0,0,1]), sphere_radius=1):
    x, y, z = p
    denom = 1 - z
    if np.isclose(denom, 0):  # Avoid division by zero at north pole
        return np.inf + 1j * np.inf
    return (x + 1j * y) / denom


def sphere_ellipses(p,center,res=360):
    r1 = math.sqrt(1 - pow(abs(p[0]),2))
    r2 = math.sqrt(1 - pow(abs(p[1]), 2))
    th = np.linspace(0, tau, res)
    x_arr_pos = np.full_like(th,p[0])
    y_arr_pos = np.full_like(th,p[1])
    x_arr_neg = np.full_like(th,-p[0])
    y_arr_neg = np.full_like(th,-p[1])
    x_cos_arr = r1 * np.cos(th)
    x_sin_arr = r1 * np.sin(th)
    y_cos_arr = r2 * np.cos(th)
    y_sin_arr = r2 * np.sin(th)
    left = list(zip(center[0] + x_arr_pos,center[1] + x_cos_arr,center[2] + x_sin_arr))
    right = list(zip(center[0] + x_arr_neg, center[1] + x_cos_arr, center[2] + x_sin_arr))
    front = list(zip(center[0] + y_cos_arr,center[1] + y_arr_pos, center[2] + y_sin_arr))
    back = list(zip(center[0] + y_cos_arr, center[1] + y_arr_neg, center[2] + y_sin_arr))
    return [left,right,front,back]


def sphere_circle(p,r,center,res=360):
    z = p[2]
    #if abs(z) > r: raise ValueError("Point is not on the sphere")
    r_h = np.sqrt(r**2 - z**2)
    return np.array([
        [center[0] + r_h * np.cos(theta), center[1] + r_h * np.sin(theta), center[2] + z]
        for theta in np.linspace(0, 2*np.pi, res)
    ])


def great_circle_1(p,r,center,num_points=360):
    ref = np.array([0, 0, 1])
    if np.allclose(p, ref) or np.allclose(p, -ref):
        ref = np.array([0, 1, 0])  # Fallback if n is near the z-axis
    u = np.cross(ref, p)
    u = u / np.linalg.norm(u)  # Normalize
    v = np.cross(p, u)
    v = v / np.linalg.norm(v)  # Normalize
    t = np.linspace(0, 2 * np.pi, num_points)
    circle_points = r * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    return center + circle_points

def great_circle_2(p,r,center,num_points=360):
    z = np.array([0, 0, 1])
    if np.allclose(np.cross(z, p), 0):
        u = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
    else:
        n = np.cross(z, p)
        n = n / np.linalg.norm(n)
        u = p / np.linalg.norm(p)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)

    t = np.linspace(0, 2 * np.pi, num_points)
    circle_points = r * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    return center + circle_points

def line_to_sphere_point(center, point_on_sphere, num_points=100):
    t_vals = np.linspace(0, 1, num_points)
    line_points = [(1 - t) * center + t * point_on_sphere for t in t_vals]
    px, py, pz = map(np.array, zip(*line_points))
    return [px,py,pz]

class SpherePoint:
    def __init__(self,r,theta,phi,center,res=360):
        self.center = np.array(center)
        self.res = res
        self.r = r
        self.theta = ang_theta(theta)
        self.phi = ang_phi(phi)
        self.coord = sphere_coord([self.r,self.theta,self.phi],self.center)
        self.ref_coord = np.array([-self.coord[0],-self.coord[1],self.coord[2]])
        self.ap_coord = np.array([-self.coord[0],-self.coord[1],-self.coord[2]])
        self.ref_ap_coord = np.array([-self.ap_coord[0],-self.ap_coord[1],self.ap_coord[2]])
        self.orthogonal_vectors = orthogonal_vectors(self.coord)
        self.u = self.orthogonal_vectors[0]
        self.v = self.orthogonal_vectors[1]
        self.ref = self.orthogonal_vectors[2]
        self.xy_circ = sphere_circle(self.coord,self.r,self.center,self.res)
        self.ap_xy_circ = sphere_circle(self.ap_coord,self.r,self.center,self.res)
        self.great_circ1 = great_circle_1(self.coord,self.r,self.center,self.res)
        self.great_circ2 = great_circle_2(self.coord,self.r,self.center,self.res)
        self.elipses = sphere_ellipses(self.coord,self.center)
        self.line_seg = line_to_sphere_point(self.center,self.coord)

    def plane_coord(self):
        X, Y, Z = self.coord[0], self.coord[1], self.coord[2]
        # Avoid division by zero (Z near 0 means "at viewer's eye")
        if Z == 0: Z = 1e-8
        return [X / (self.r - Z), Y / (self.r - Z)]


def plane_to_sphere_point(x,y,side_length):
  r = side_length / 2 # can be the radius value for a spherical coordinate
  theta = np.arctan2(y,x) # can be the theta value for a spherical coordinate
  phi = np.sqrt(pow(x, 2) + pow(y, 2)) # can be the polar angle for the spherical coordinate
  return SpherePoint(r,theta,phi,center=[0,0,r])


def stereographic_projection(p):
    x,y,z = p[0],p[1],p[2]
    denom = 1 - z
    if denom == 0:
        return np.inf  # handle north pole
    return complex(x, y) / denom

