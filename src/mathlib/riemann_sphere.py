import numpy as np
from math import sin, cos, pi, tau, radians
from constants import MIN_VAL
from itertools import product

right_angle = pi/2
def abs_angle(val:float): return radians(val) % right_angle

def clip(val:float,low:float,high:float):
  if val < low: return low
  elif val > high: return high
  else: return val

#pauli matrices
o1 = np.array([[0,1],[1,0]], dtype=complex)
o2 = np.array([[0,-1j],[1j,0]], dtype=complex)
o3 = np.array([[1,0],[0,-1]], dtype=complex)
#SU2 identity
I = np.eye(2, dtype=complex)

#find rotation mobius_coef for complex number
def rot_mobius_coef(z:complex):
  x,y = z.real,z.imag
  norm=np.sqrt(1+((x*x)+(y*y)))
  a=1/norm
  b=z/norm
  c=-np.conjugate(z)/norm
  d=1/norm
  return a,b,c,d

def inv_stereographic_proj(z:complex):
  m2 = abs(z)**2
  d=1+m2
  x=(2*z.real)/d
  y=(2*z.imag)/d
  z=(m2-1)/d
  return np.array([x,y,z])

class riemann_sphere_coord:

  def __init__(self,theta:float,phi:float,spart:(int,int,int)):
    assert spart in sparts, "not valid sphere part argument"
    self._spart = spart
    self._theta = np.radians(theta) % right_angle
    self._phi = np.radians(phi) % right_angle
    self._pr = cos(self._phi)
    self._x = self._pr * cos(self._theta)
    self._y = self._pr * sin(self._theta)
    self._z = np.sin(self._phi)

  @staticmethod
  def disc_point_constructor(p):
    x,y = p
    zeta=complex(x,y)
    r=abs(zeta)
    theta=np.arctan2(y,x)
    phi=np.arccos(r)
    return riemann_sphere_coord(theta,phi)

  @property
  def at_inf(self): return np.isclose(self.z,1.0,MIN_VAL,MIN_VAL)

  @property
  def on_equator(self): return np.isclose(self.z,0.0,MIN_VAL,MIN_VAL)

  @property
  def equator_pt(self): return np.array([cos(self._theta),sin(self._theta),0.0])

  @property
  def theta(self):return self._theta

  @theta.setter
  def theta(self,val:float):
    self._theta = np.radians(val) % right_angle
    self._x = self._pr * np.cos(self._theta)
    self._y = self._pr * np.sin(self._theta)

  @property
  def phi(self): return self._phi

  @phi.setter
  def phi(self,val:float):
    self._phi = np.radians(val) % right_angle
    self._pr = np.cos(self._phi)
    self._x = self._pr * np.cos(self._theta)
    self._y = self._pr * np.sin(self._theta)
    self._z = np.sin(self._phi)

  @property
  def spart(self): return self._spart

  @spart.setter
  def spart(self,val:(int,int,int)):
    assert spart in sparts, "not valid sphere part argument"
    self._spart = val

  @property
  def x(self): return self._x * self._spart[0]

  @property
  def y(self): return self._y * self._spart[1]

  @property
  def z(self): return self._z * self._spart[2]

  @property
  def zeta(self): return complex(self._x,self._y) / (1 - self.z)

  @property
  def xi(self): return complex(self._x,-self._y) / (1 + self.z)

  @property
  def north_a(self): return np.array([inf,inf,inf]) if self.at_inf else np.array([self._x,self._y,self._z])

  @property
  def south_a(self): return np.array([0.0,0.0,-1.0]) if self.at_inf else np.array([self._x,self._y,-self._z])

  @property
  def north_b(self): return np.array([inf,inf,inf]) if self.at_inf else np.array([-self._x,self._y,self._z])

  @property
  def south_b(self): return np.array([0.0,0.0,-1.0]) if self.at_inf else np.array([-self._x,self._y,-self._z])

  @property
  def north_c(self): return np.array([inf,inf,inf]) if self.at_inf else np.array([-self._x,-self._y,self._z])

  @property
  def south_c(self): return np.array([0.0,0.0,-1.0]) if self.at_inf else np.array([-self._x,self._y,-self._z])

  @property
  def north_d(self): return np.array([inf,inf,inf]) if self.at_inf else np.array([self._x,-self._y,self._z])

  @property
  def south_d(self): return np.array([0.0,0.0,-1.0]) if self.at_inf else np.array([self._x,-self._y,-self._z])

  @property
  def zeta_rot_mobius_coef(self):
    return rot_mobius_coef(self.zeta)

  @property
  def xi_rot_mobius_coef(self):
    return rot_mobius_coef(self.xi)

  def zeta_rot_mobius_trans(self,z:complex):
    a,b,c,d = self.zeta_rot_mobius_coef
    return ((z*a)+b)/((z*c)+d)

  def xi_rot_mobius_trans(self,z:complex):
    a,b,c,d = self.xi_rot_mobius_coef
    return ((z*a)+b)/((z*c)+d)




























