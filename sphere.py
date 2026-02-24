import math
import numpy as np
from math import tau, pi, sin, cos, sqrt, tan, prod, factorial,floor
from vector_utils import normalize_vector
from num import normalize

def paralell_radius(phi:float):return tan(phi / 2)

class SphereCoord:

    def __init__(self, radius: float, theta: float, phi: float):
        assert 0 <= theta <= tau, "theta must be radian between 0 - 2pi"
        assert 0 <= phi <= pi, "theta must be radian between 0 - pi"
        self.radius = radius
        self.theta = theta
        self.phi = phi

def center_vector(P, C = np.array([0.0, 0.0, 1.0])):
    v = C - P
    norm = np.linalg.norm(v)
    if norm == 0: return np.array([0.0, 0.0, 0.0])
    return v / norm

def parallel_radius(P,r):
    z = P[2]
    return np.sqrt(pow(r, 2) - pow(z, 2))




def sphere_point_orientation(pt, R):
    pt = np.array(pt, dtype=float)
    r = R
    u = normalize_vector(pt)  # radial direction
    cv = center_vector(pt,np.array([0,0,r]))

    # XY-plane oriented tangent plane
    xy_n = normalize_vector(np.cross([0,0,r], pt))  # tangent vector
    xy_v = normalize_vector(np.cross(xy_n, u))         # second tangent vector
    xy_plane = (xy_n, xy_v)

    # XZ-plane oriented tangent plane
    xz_n = normalize_vector(np.cross([0,r,0], pt))
    xz_v = normalize_vector(np.cross(xz_n, u))
    xz_plane = (xz_n, xz_v)

    # YZ-plane oriented tangent plane
    yz_n = normalize_vector(np.cross([r, 0, 0], pt))
    yz_v = normalize_vector(np.cross(yz_n, u))
    yz_plane = (yz_n, yz_v)

    return {
        "pt":pt,
        "center_vector":cv,
        "radial": u,
        "xy_plane": xy_plane,
        "xz_plane": xz_plane,
        "yz_plane": yz_plane
    }


def sphere_to_cart(r,theta,phi):
    x = r * sin(phi) * cos(theta)
    y = r * sin(phi) * sin(theta)
    z = r * cos(phi)
    return [x,y,z]



def parallel_circ(r,theta,phi,angles):
    r, phi = r, phi
    z = r * np.cos(phi)
    rho = np.sqrt(pow(r, 2) - pow(z, 2))
    pt = lambda a: np.array([rho * cos(a), rho * sin(a), z])
    return np.array([pt(a) for a in angles])

def meridian_circ(r,theta,phi,angles):
    return np.array([[r * np.sin(a) * np.cos(theta),r * np.sin(a) * np.sin(theta),r * np.cos(a)]
        for a in angles
    ])

def great_circ_xy(r,theta,phi, angles):
    r = r
    ref = [0, 0, r]
    p = sphere_to_cart(r,theta,phi)
    n = normalize_vector(np.cross(ref, p))
    u = normalize_vector(p)
    v = normalize_vector(np.cross(n, u))
    return r * (np.outer(np.cos(angles), u)) + np.outer(np.sin(angles), v)

def great_circ_xz(r,theta,phi, angles):
    r = r
    ref = [0, r, 0]
    p = sphere_to_cart(r,theta,phi)
    n = normalize_vector(np.cross(ref, p))
    u = normalize_vector(p)
    v = normalize_vector(np.cross(n, u))
    return r * (np.outer(np.cos(angles), u)) + np.outer(np.sin(angles), v)

def great_cric_yz(r,theta,phi,angles):
    ref = [r, 0, 0]
    p = sphere_to_cart(r,theta,phi)
    n = normalize_vector(np.cross(ref, p))
    u = normalize_vector(p)
    v = normalize_vector(np.cross(n, u))
    return r * (np.outer(np.cos(angles), u)) + np.outer(np.sin(angles), v)

def ellipse_xz(r,theta,phi,angles):
    pv = r * sin(phi) * sin(theta)
    r = sqrt(1 - pow(abs(pv),2))
    return np.array([[r*cos(a),pv,r*sin(a)] for a in angles])

def ellipse_xz_neg(r,theta,phi, angles):
    pv = r * sin(phi) * sin(theta)
    r = sqrt(1 - pow(abs(pv), 2))
    return np.array([[r * cos(a), -pv, r * sin(a)] for a in angles])

def ellipse_yz(r,theta,phi,angles):
    pv = r * sin(phi) * cos(theta)
    r = sqrt(1 - pow(abs(pv), 2))
    return np.array([[pv, r * cos(ang), r * sin(ang)] for ang in angles])

def ellipse_yz_neg(r,theta,phi,angles):
    pv = r * sin(phi) * cos(theta)
    r = sqrt(1 - pow(abs(-pv), 2))
    return np.array([[-pv, r * cos(ang), r * sin(ang)] for ang in angles])


def sphere_radius(r,t): return sqrt(pow(r,2) - pow(t,2))

def area_sphere(r):return 4*pi*pow(r,2)

def n_ang_cos(ang):
  return cos(pi * normalize(ang,pi))

def n_ang_sin(ang):
  return sin(tau * normalize(ang,tau))

def n_coord(arr,i):
    if i == 1:
        return arr[0] * math.cos(arr[1])
    elif i == len(arr):
        l = [arr[0]] + [math.sin(v) for v in arr[1:i]]
        print(l)
        return prod(l)
    else:
        if i > len(arr):
            raise ValueError("i not in range")
        else:
          sin_vals = [math.sin(v) for v in arr[1:i-1]]
          cos_val = [math.cos(arr[-1])]
          l = [arr[0]] + sin_vals + cos_val
          print(l)
          return prod(l)

def n_coord_normalize(arr,i):
  if i <= len(arr):
    is_sec_last = i == (len(arr) - 1)
    is_first = i == 1
    if is_first:
      parr = [arr[0]*n_ang_cos(arr[1])]
      return [prod(parr),len(parr)]
    elif is_sec_last:
      parr = [arr[0]] + [n_ang_sin(v) for v in arr[1:i-1]] + [n_ang_cos(arr[i])]
      return [prod(parr),len(parr)]
    else:
      parr = [arr[0]] + [n_ang_sin(v) for v in arr[1:i]]
      return [prod(parr),len(parr)]
  else:
      raise ValueError("i must be in range of arr")

def n_coords(arr):
  if len(arr) >= 3:
    dim = len(arr)
    r = arr[0]
    ret = {}
    for i in range(1,dim+1):
        ret[f"x{i}"] = n_coord(arr,i)
    return ret

def conj_unit(num_of_decimals):return round(tau,num_of_decimals) - round(tau,num_of_decimals - 1)

def derivative(x):
    h,f = 1e-5, lambda v: v**2 + 1
    return (f(x+h)-f(x-h))/(2*h)

def n_sphere_vol(n, r):
    return (pow(2,n) * pow(r,n) * pow((tau / 4),floor(n/2))) / factorial(factorial(n))

def n_sphere_area(n, r):
    Rn = pow(r,n-1) * r
    cp = Rn / 2
    sp = cp / 2
    return derivative(n_sphere_vol(n,Rn))

def hypersphere_parallel(radius,ang_cnt=90,cnt=10):
    angles = np.linspace(0,tau,ang_cnt)
    chis = np.linspace(0, np.pi, cnt)
    phis = np.linspace(0,np.pi,cnt)
    circles = []
    for chi in chis:
        r_eff = radius * np.sin(chi)
        z_offset = radius * np.cos(chi)
        # project the 2-sphere (x,y,z) part
        sph = [parallel_circ(r_eff, 0.0, phi,angles) + np.array([0.0,0.0, z_offset]) for phi in phis]
        circles.append(sph)
    return circles


def hypersphere_meridian(radius, theta, cnt=10):
    chis = np.linspace(0, np.pi,cnt)
    phis = np.linspace(0, np.pi,cnt)
    curves = []
    for chi in chis:
        r_eff = radius * np.sin(chi)
        z_offset = radius * np.cos(chi)
        circ = np.array([
            [r_eff * np.sin(phi) * np.cos(theta),
             r_eff * np.sin(phi) * np.sin(theta),
             r_eff * np.cos(phi) + z_offset]
            for phi in phis
        ])
        curves.append(circ)
    return curves

def hypermeridian(radius,ang_cnt=90, cnt=10):
    thetas = np.linspace(0, tau, ang_cnt)
    return [hypersphere_meridian(radius, th, cnt) for th in thetas]



