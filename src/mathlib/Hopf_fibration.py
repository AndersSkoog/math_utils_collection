import numpy as np
import cmath
from math import tau
from S2 import stereo_project_R2_R3
from SO import SO_3
from SU2 import SU2

def torsion_angle(disc_points,index):
  assert 0 <= index, "index out of range"
  li = len(disc_points) - 1
  if (index + 1) <= li:
    p1,p2 = disc_points[index],disc_points[index+1]
    a,b = [-p1[1],p1[0]], [-p2[1],p2[0]]
    det = a[0] * b[1] - a[1] * b[0]
    return np.arctan2(det, np.dot(a, b))
  else: return 0.0


def base_fiber(res: int):
  angs = np.linspace(0, tau, res, endpoint=False)
  return np.asarray([(cmath.exp(1j*t),0.0+0.0j) for t in angs],dtype=complex)

base_fiber_360 = base_fiber(360)

def hopf_link_from_disc_point(pts,index,R=1.0,res=360):
  assert 0 <= index <= (len(pts)-1), "index out of range"
  tor_ang = torsion_angle(pts,index)
  dp = pts[index]
  sp_1 = stereo_project_R2_R3(dp,R)
  x,y,z = sp_1
  theta,phi = np.arctan2(y,x), y*np.arccos(z / R)  # colatitude
  orient = SO_3(theta,phi,tor_ang)
  sp_2 = sp_1 @ orient.T  # or should it be orient @ sp[0] ?
  U1 = SU2(sp_1,1.0)
  U2 = SU2(sp_2,1.0)
  fiber1 = base_fiber_360 @ U1 if res == 360 else base_fiber(res) @ U1
  fiber2 = base_fiber_360 @ U2 if res == 360 else base_fiber(res) @ U1
  return fiber1,fiber2

def hopf_fibration(pts,R=1.0):
  return [hopf_link_from_disc_point(pts,i,R) for i in range(len(pts))]

def hopf_fiber_to_R4(fiber):
  # fiber shape: (N, 2) complex
  z1 = fiber[:, 0]
  z2 = fiber[:, 1]
  return np.column_stack([z1.real, z1.imag, z2.real, z2.imag])

def stereo_S3_to_R3(w, eps=1e-9):
  denom = 1.0 - w[:,3]
  denom = np.where(np.abs(denom) < eps, eps, denom)
  return w[:, :3] / denom[:, None]

def proj_hopf_fiber(fiber):
  w = hopf_fiber_to_R4(fiber)
  return stereo_S3_to_R3(w)

def proj_hopf_link(link): return proj_hopf_fiber(link[0]),proj_hopf_fiber(link[1])

def proj_hopf_fibration(fibration):
 circles1 = []
 circles2 = []
 for link in fibration:
   fib1,fib2 = link
   circ1,circ2 = proj_hopf_fiber(fib1),proj_hopf_fiber(fib2)
   circles1.append(circ1)
   circles2.append(circ2)
 return circles1,circles2