import numpy as np
import cmath
from math import tau,pi,cos,sin

def mag(z:complex): return abs(z)
arg = cmath.phase

def ang_sector(ang:float):
  a = np.rad2deg(ang) % 360
  if 0   <= a <  90: return 1
  if 90  <= a < 180: return 2
  if 180 <= a < 270: return 3
  if 270 <= a < 360: return 4
  else: raise ValueError("Unknown Error")

def angle_between(a1:float,a2:float):
  s1,s2 = ang_sector(a1),ang_sector(a2)
  if s1 == 4 and s2 == 1: return (tau - a1) + a2
  elif s2 == 4 and s1 == 1: return (tau - a2) + a1
  else: return max(a1,a2) - min(a1,a2)


"""return the complex number marking the center of the circle intersecting p and q in the poincare disc"""
def hyperbolic_cirlce_center(p:complex,q:complex,eps=1e-10) -> complex:
  c_num = (p * (1 - mag(q) ** 2)) - (q * (1 - mag(p) ** 2))
  c_den = (p * np.conj(q)) - (np.conj(p) * q)
  return c_num / c_den

"""given a complex number in the poincare disc marking the center of a circle 
find the two complex numbers on the unit circle that the circle intersects"""
def hyperbolic_circle_intersections(circle_center:complex):
  c = circle_center
  magc = mag(c)
  if magc <= 1: raise ValueError("Center must satisfy |C|>1 for an orthogonal circle.")
  argc = arg(c)
  cosc = np.arccos(1.0/magc)
  z1 = cmath.exp(1j * (argc + cosc)) # circle intersecting unit circle at z1 and z2
  z2 = cmath.exp(1j * (argc - cosc))
  return z1,z2

"""find the hyperbolic line between p and q"""
def hyperbolic_line(p: complex, q: complex, res=100, eps=1e-10):
  if abs(p - q) < eps: raise ValueError("Points p and q are too close to define a unique hyperbolic line.")

  # Handle the straight line case (p and q colinear with origin)
  if (abs(np.conj(p) * q) - (p * np.conj(q))) < eps:
    t = np.linspace(0, 1, res)
    pts = p + t * (q - p)
    return np.array(list(map(lambda pt: [pt.real,pt.imag], pts)))

  c = hyperbolic_cirlce_center(p,q)
  z1,z2 = hyperbolic_circle_intersections(c)
  r = abs(p - c) #radius of circle
  a1  = min(arg(z1),arg(z2))
  a2  = max(arg(z1),arg(z2))
  angs = int(np.rad2deg(angle_between(a1,a2)))
  lp   = np.linspace(a1,a2,angs)
  return np.array([[np.cos(a),np.sin(a)] for a in lp])

"""hyperbolic line based on circle center an radius"""
def hyperbolic_line2(c:complex,r:float,eps=1e-10):
    z1, z2 = hyperbolic_circle_intersections(c)
    a1 = min(arg(z1), arg(z2))
    a2 = max(arg(z1), arg(z2))
    angs = int(np.rad2deg(angle_between(a1, a2)))
    lp = np.linspace(a1, a2, angs)
    return np.array([[np.cos(a), np.sin(a)] for a in lp])

"""Generate a regular hyperbolic {p,q}-gon centered at `center'"""
def hyperbolic_polygon(p: int, q: int, center: complex = 0.0 + 0.0j, res: int = 50) -> list:
  # Compute the inner angle of the polygon (hyperbolic geometry)
  inner_angle = tau / q
  # Compute the side length (hyperbolic distance) using the hyperbolic cosine rule
  cosh_r = (cos(pi/p)*cos(pi/q)+cos(inner_angle/2))/(sin(pi/p)*sin(pi/q))
  r = np.arccosh(cosh_r)  # Radius of the circumscribed circle
  # Convert hyperbolic radius to Poincaré disk radius
  pr = (np.exp(r) - 1) / (np.exp(r) + 1)
  # Generate vertices uniformly spaced around the center
  ang = lambda i: (tau * i) / p
  vertices = [center + pr * cmath.exp(1j*ang(i)) for i in range(p)]
  return vertices