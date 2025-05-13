import cmath
import math
import functools
import numpy as np
from typing import TypedDict, Union, NewType, Annotated
from math_utils import normalize as normalize2,circum,linlin
from Quaternion import Quaternion

Number = Union[int, float]
Theta = Annotated[Number, "Must be in range [0, π]"]
Phi   = Annotated[Number, "Must be in range [0, 2π]"]

class P3(TypedDict):
    x:Number
    y:Number
    z:Number

def sin(value:Number):  return round(math.sin(value),8)
def cos(value:Number):  return round(math.sin(value),8)
def tan(value:Number):  return round(math.tan(value),8)
def atan(value:Number): return round(math.atan(value),8)
def acos(value:Number): return round(math.acos(value),8)
def asin(value:Number): return round(math.asin(value),8)
#quaternion directions
east = Quaternion(a=0,  b=1,  c=1,  d=0)
west = Quaternion(a=0,  b=-1, c=-1, d=0)
nrth = Quaternion(a=0,  b=-1j,c=1j, d=0)
soth = Quaternion(a=0,  b=1j, c=-1j,d=0)
frwd = Quaternion(a=1,  b=0,  c=0,  d=-1)
back = Quaternion(a=-1, b=0,  c=0,  d=1j)


def proj_real_line(ang_deg):
    z = cmath.exp(math.radians(ang_deg))
    expr = 2 * z.real / 1 + (cmath.sqrt((z.real**2)+(z.imag**2)))
    print(expr)
    #z = complex(math.radians(ang_deg)),math.radians(ang_deg)
    #print(z)
    return z

test = [proj_real_line(n) for n in range(344)]
print(test)


def onS2(p) -> bool:
  px,py,pz = p["x"],p["y"],p["z"]
  return pow(px,2) + pow(py,2) + pow(pz,2) == 1
def in_range(value:Number,low:Number,high:Number) -> Number:
    """Ensures value is in the range [0, π]."""
    if value < low or value > high:
        raise ValueError(f"Value {value} out of range")
    return value

def P_S2(r: Number, theta: Theta, phi: Phi) -> P3:
    # Convert spherical to Cartesian coordinates
    x = r * math.sin(theta) * math.cos(phi)
    y = r * math.sin(theta) * math.sin(phi)
    z = r * math.cos(theta)
    on_s2 = (x**2) + (y**2) + (z ** 2)
    return P3(x=x, y=y, z=z)
def P_S2z(p:P3) -> Number:
    px,py = p["x"] ** 2,p["y"] ** 2
    return (1/2) * (1 - px - py)
def P_S2w(p:P3) -> Number:
    px,py = p["x"] ** 2,p["y"] ** 2
    return (1/2) * (1 + px + py)
class SP:

    def __init__(self,point):
      self.ex,ey,ez = [1,0,0],[0,1,0],[0,0,1]

      self.theta = in_range(point[0],0,math.pi)
      self.phi = in_range(point[1],0,math.tau)
      self.x = math.sin(self.theta) * math.cos(self.phi)
      self.y = math.sin(self.theta) * math.sin(self.phi)
      self.z = (1/2) * (1 - self.x - self.y)
      self.w = (1/2) * (1 + self.x + self.y)
      self.mag = ((self.x**2) + (self.y**2)) ** 1/2
      self.u = [self.z/self.w,self.x/self.w,self.y/self.w]
      self.v = [self.z,self.x,self.y]
      self.lon = math.atan2(self.y,self.x)
      self.lat = ((1/2)*math.pi) - (2*math.atan(self.mag))
      self.clat = 2 * math.atan(self.mag)
      self.sx = math.tan((1/2)*((1/2)*math.pi-self.lat) * math.cos(self.lon))
      self.sy = math.tan((1 / 2) * ((1 / 2) * math.pi - self.lat) * math.sin(self.lon))
      self.cos_lon  =  self.x / self.mag
      self.tan_lon  =  self.y / self.x
      self.sin_lon  =  self.y / self.mag
      self.tan_clat =  self.mag / self.z
      self.cos_clat =  self.z / self.w
      self.sin_clat =  self.mag / self.w


    def rect_proj(self):
        return [self.lon,self.lat]


    def tan_proj(self):
        return [self.x / self.z,self.y / self.z]


    def orthograph_proj(self):
        return [self.x/self.w,self.y/self.w]


    def angle_sum(self,other):
        if isinstance(other,SP):
            x1,x2 = self.x,other.x
            return (x1*x2) / 1 - (x1*x2)
        elif isinstance(other,list) and all([isinstance(el,SP) for el in other]):
            xl = [self.x] + [el.x for el in other]
            nom = sum(xl) - math.prod(xl)
            dnom = 1
            for ind in range(1,len(other) + 1):
                v = xl[ind-1]*xl[ind] if ind < (len(other) - 1) else xl[-1] * xl[0]
                dnom -= v
            return nom / dnom
    @staticmethod
    def inv_tan_proj(ta,sign):
        ta_x,ta_y = ta[0],ta[1]
        if sign == 1:
            ret_x = (ta_x / 1) + ((1+pow(ta_x,2)) ** (1/2))
            ret_y = (ta_y / 1) + ((1 + pow(ta_y, 2)) ** (1 / 2))
            return [ret_x,ret_y]
        elif sign == -1:
            ret_x = (ta_x / 1) - ((1 + pow(ta_x, 2)) ** (1 / 2))
            ret_y = (ta_y / 1) - ((1 + pow(ta_y, 2)) ** (1 / 2))
            return [ret_x, ret_y]
        else:
            raise ValueError("sign must be -1 or 1")


#returns an map function that draw an elipsoid and its antipodal on S2 with radius r, with its center point displaced from the orgin along axis by coef between <= 0 and >1
#where 0 is the full hemispherical circle along the given axis, and 1 is the vanishing point.
def elipsoid_straight(axis,disp):
    if isinstance(axis,int) and 0 <= axis <= 2 and 0 <= disp < 1:
      elip_scale = lambda d: np.sqrt(1 - (d**2))
      ax = ["x","y","z"][axis]
      #map function that returns the (pn) point of (pcount) points for an elipsoid on a sphere of radius (sr) and its antipodal
      pmap = lambda pcount,sr,pn: [
          [
            (elip_scale(disp)*np.cos(math.tau*((1/pcount)*(pn%pcount))))*sr if ax!="x"else(math.tau*((1/pcount)*(pn%pcount)))*sr,
            (elip_scale(disp)*np.cos(math.tau*((1/pcount)*(pn%pcount))))*sr if ax!="x"else(math.tau*((1/pcount)*(pn%pcount)))*sr,
            (elip_scale(disp)*np.sin(math.tau*((1/pcount)*(pn%pcount))))*sr if ax!="x"else(math.tau*((1/pcount)*(pn%pcount)))*sr
          ],
          [
           (elip_scale(-disp)*np.cos(math.tau*((1/pcount)*(pn%pcount))))*sr if ax!="x"else(math.tau*((1/pcount)*(pn%pcount)))*sr,
           (elip_scale(-disp)*np.cos(math.tau*((1/pcount)*(pn%pcount))))*sr if ax!="x"else(math.tau*((1/pcount)*(pn%pcount)))*sr,
           (elip_scale(-disp)*np.sin(math.tau*((1/pcount)*(pn%pcount))))*sr if ax!="x"else(math.tau*((1/pcount)*(pn%pcount)))*sr
          ]
      ]
      return pmap

    else: raise ValueError("arguments out of range or has wrong type")
#rotated version of elipsoid along axis with a tilt by angle
def elipsoid_tilted(axis,disp,tilt):
  def pmap(pcount,sr,pn):
      pts = elipsoid_straight(axis,disp)(pcount,sr,pn)
      tilt_scale = lambda d: np.sqrt(1-(d**2))
      e1 = [pts[0][0],pts[0][1],pts[0][2]]
      e2 = [pts[1][0],pts[1][1],pts[1][2]] if disp != 0 else e1
      rot_e1 = Quaternion.rotate(e1,axis,tilt)
      rot_e2 = Quaternion.rotate(e2,axis,tilt) if disp != 0 else rot_e1
      return [rot_e1,rot_e2]
  return pmap
def mobius_elipsoid(coord):
    #v has no z, h has z
    phi,theta = coord[2], coord[1]
    vdisp,hdisp = np.cos(np.radians(90) - coord[1]),np.sin(np.radians(theta))
    vr, hr = math.sqrt(1 - pow(abs(vdisp),2)),math.sqrt(1 - pow(abs(hdisp),2))
    th = np.linspace(0, 2 * np.pi, 360)
    v_elips_x = np.full_like(th,vdisp)
    v_elips_y = vr * np.cos(th)
    v_elips_z = vr * np.sin(th)
    h_elips_x = vr * np.cos(th)
    h_elips_y = vr * np.sin(th)
    h_elips_z = np.full_like(th,hdisp)
    print([v_elips_z,v_elips_y,v_elips_x],[h_elips_x,h_elips_y,h_elips_z])

class ExtendedComplex:
    def __init__(self, value):
        self.value = value  # Can be a complex number or 'inf'

    def is_infinite(self):
        return self.value == 'inf'

    def __add__(self, other):
        if self.is_infinite() or isinstance(other, ExtendedComplex) and other.is_infinite():
            return ExtendedComplex('inf')
        return ExtendedComplex(self.value + other.value)

    def __sub__(self, other):
        if self.is_infinite() or isinstance(other, ExtendedComplex) and other.is_infinite():
            return ExtendedComplex('inf')
        return ExtendedComplex(self.value - other.value)

    def __mul__(self, other):
        if self.is_infinite() or isinstance(other, ExtendedComplex) and other.is_infinite():
            return ExtendedComplex('inf')
        return ExtendedComplex(self.value * other.value)

    def __truediv__(self, other):
        if other.is_infinite():
            return ExtendedComplex(0)
        if self.is_infinite():
            return ExtendedComplex('inf')
        return ExtendedComplex(self.value / other.value)

    def conjugate(self):
        if self.is_infinite():
            return self
        return ExtendedComplex(self.value.conjugate())

    def magnitude(self):
        if self.is_infinite():
            return float('inf')
        return abs(self.value)

    def phase(self):
        if self.is_infinite():
            return None
        return math.atan2(self.value.imag, self.value.real)

    def __repr__(self):
        return str(self.value)

class RiemannSphere:
    def __init__(self, radius=1):
        self.radius = radius

    def stereographic_projection(self, p):
        x, y, z = p
        return ExtendedComplex(x / (self.radius - z) + 1j * y / (self.radius - z))

    def inverse_stereographic_projection(self, c):
        if isinstance(c,ExtendedComplex):
            if c.is_infinite():
                return ExtendedComplex(0)
        x, y = c.value.real, c.value.imag
        denom = 1 + x ** 2 + y ** 2
        r, th, z = (2 * x) / 1 + pow(x, 2), y, (pow(x, 2) - 1) / (pow(y, 2) + 1)
        return tuple((2 * x / denom, 2 * y / denom, (denom - 1) / denom * self.radius))

def p2( x, y ):                    return [x,y,0]
def p3( x, y, z ):                 return [x,y,z]
def add( a, b ):                   return p3( a[0] + b[0], a[1] + b[1], a[2] + b[2] )
def sub( a, b ):                   return p3( a[0] - b[0], a[1] - b[1], a[2] - b[2] )
def mul( a, f ):                   return p3( a[0] * f, a[1] * f, a[2] * f )
def average( a, b ):               return mul( add(a, b),0.5 )
def dot( a, b ):                   return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
def len2( a ):                     return dot(a,a)
def dist2( a, b ):                 return len2( sub( a, b ) )
def dist( a, b ):                  return math.sqrt( len2( sub( a, b ) ) )
def mul_complex( a, b ):           return p2( a[0] * b[0] - a[1] * b[1], a[1] * b[0] + a[0] * b[1] )
def div_complex( a, b ):           return p2( ( a[0] * b[0] + a[1] * b[1] ) / len2( b ), ( a[1] * b[0] - a[0] * b[1] ) / len2( b ) )
def negative( a ):                 return p2( -a[0], -a[1] )
def magnitude( a ):                return math.sqrt( len2( a ) )
def normalize( a ):                return mul( a, 1 / magnitude( a ) )
def cross( a, b ):                 return p3( a[1] * b[2] - a[2] * b[1], a[2]*b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0] )
def phase( a ):                    return math.atan2( a[1], a[0] )
def toPolar( a ):                  return [ magnitude( a ), math.atan2( a[1], a[0] ) ]
def fromPolar( mag, _phase ):      return p2( mag * math.cos(_phase), mag*math.sin(_phase) )
def sqrt_complex( a ):             return fromPolar( math.sqrt( magnitude(a) ), phase(a) / 2.0 )
def complex_conjugate( _p ):       return p2( _p[0], -_p[1] )
def sphereInversion( _p, sphere ): return add( sphere.p, mul( sub( _p, sphere.p ), sphere.r * sphere.r / dist2( _p, sphere.p ) ) )
def get_mobius_determinant( m ):   return sub( mul_complex( m[0], m[3] ), mul_complex( m[1], m[2] ) )
def pow_complex( a, _p ):
    pol = toPolar( a )
    r,theta = pol[0],pol[1]
    return fromPolar( pow( r, _p ), theta * _p )
def mobius_normalize( m ):
    sqrt_ad_minus_bc = sqrt_complex( get_mobius_determinant( m ) )
    for i in range(0,4):
        m[i] = div_complex( m[i], sqrt_ad_minus_bc )
def mobius_make_unitary( m ):
    m[2] = mul( complex_conjugate( m[1] ), -1 )
    m[3] = complex_conjugate( m[0] )
def get_mobius_inverse( m ):
    return [m[3], mul(m[1], -1.0), mul(m[2], -1.0), m[0]]
def get_mobius_composed(args):
    return functools.reduce(lambda a, b: [
        add(mul_complex(a[0], b[0]), mul_complex(a[1], b[2])),
        add(mul_complex(a[0], b[1]), mul_complex(a[1], b[3])),
        add(mul_complex(a[2], b[0]), mul_complex(a[3], b[2])),
        add(mul_complex(a[2], b[1]), mul_complex(a[3], b[3]))
    ],args)
def mobius_make_nonloxodromic( m ):
    m[3].y = -m[0].y
def mobius_identity( m ):
    m[0] = p2( 1, 0 )
    m[1] = p2( 0, 0 )
    m[2] = p2( 0, 0 )
    m[3] = p2( 1, 0 )
def mobius_on_point( m, _z ):
    if math.isnan(_z[0]) or math.isnan(_z[1]): return div_complex( m[0], m[2] )
    else: return div_complex( add( mul_complex( m[0], _z ), m[1] ), add( mul_complex( m[2], _z ), m[3] ) )
def mobius_on_circle(m, _c):
    _z = sub( _c.p, div_complex( p2(_c[1] * _c[1], 0), complex_conjugate( add( div_complex( m[3], m[2] ), _c[0]) ) ) )
    _q = mobius_on_point(m, _z)
    #//console.log('q:',q);
    #//console.log('m(p):',mobius_on_point(m, c.p));
    #//console.log('m(p+r):',mobius_on_point(m, add(c.p, p2(c.r, 0.0))));
    s = dist(_q, mobius_on_point(m, add(_c[0], p2(_c[1], 0.0))))
    #//console.log('s:',s);
    return [_q,s]
def z_not_zero(cmplx): return isinstance(cmplx,complex) and cmplx != complex(0,0)
def z_conj(cmplx):
  if isinstance(cmplx,complex): return cmplx - complex(cmplx.real, -cmplx.imag)
  else: raise ArithmeticError("expected Complex argument")
def mobius_trans(z:complex,coef) -> complex:
  a,b,c,d = coef[0],coef[1],coef[2],coef[3]
  return ((a*z) + b) /((c*z) + d)
def mobius_transl(z:complex,b:complex) -> complex:
  return mobius_trans(z,[1,b,0,1])
def mobius_scale(z,s):
    return mobius_trans(z,[s,0,0,1])
def mobius_inv(z):
    return mobius_trans(z,[0,1,1,0])
def mobius_rot(z,ang):
    return mobius_trans(z, [cmath.exp(1j * ang), 0, 0, 1])
def rad_distance(tup):
    u,v = tup[0],tup[1]
    return (pow(u,2) + pow(v,2)) ** (1/2)
def polar(ang):
    return math.sin(ang) / (1 - math.cos(ang))
def sphere_to_polar(tup):
    r,th,z = tup[0],tup[1],tup[2]
    return tuple((r/1-z,th))
def polar_to_sphere(tup):
    _p1,_p2 = tup[0],tup[1]
    r,th,z = (2*_p1)/1+pow(_p1,2),p2,(pow(_p1,2) - 1) / (pow(_p1,2) + 1)
    return [r,th,z]
def cart_plane_to_sphere(tup):
    u, v = tup[0],tup[1]
    su,sv = pow(u,2),pow(u,v)
    u2,v2,r2 = pow(rad_distance(tup),2),u*2,v*2
    #x,y,z = u2/(1+r2),v2/(1+r2),(1 - r2) / (1 + r2)
    x,y,z = u2 / (1+su+sv), v2 / (1+su+sv), (1-su-sv) / (1+su+sv)
    return tuple((x, y, z))
def cart_sphere_to_plane(tup):
    x,y,z = tup[0],tup[1],tup[2]
    u,v = x / (1-z), y / (1-z)
    return tuple((u, v))
def n_ang_cos(ang): return math.cos(math.pi * normalize2(ang,math.pi))
def n_ang_sin(ang): return math.sin(math.tau * normalize2(ang,math.tau))
def n_coord(arr,i):
    if i == 1:
        return arr[0] * math.cos(arr[1])
    elif i == len(arr):
        l = [arr[0]] + [math.sin(v) for v in arr[1:i]]
        print(l)
        return math.prod(l)
    else:
        if i > len(arr):
            raise ValueError("i not in range")
        else:
          sin_vals = [math.sin(v) for v in arr[1:i-1]]
          cos_val = [math.cos(arr[-1])]
          l = [arr[0]] + sin_vals + cos_val
          print(l)
          return math.prod(l)
def n_coord_normalized(arr,i):
  if i <= len(arr):
    is_sec_last = i == (len(arr) - 1)
    is_first = i == 1
    if is_first:
      parr = [arr[0]*n_ang_cos(arr[1])]
      return [math.prod(parr),len(parr)]
    elif is_sec_last:
      parr = [arr[0]] + [n_ang_sin(v) for v in arr[1:i-1]] + [n_ang_cos(arr[i])]
      return [math.prod(parr),len(parr)]
    else:
      parr = [arr[0]] + [n_ang_sin(v) for v in arr[1:i]]
      return [math.prod(parr),len(parr)]
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
#map a point in the xy cart plane to point with its antipodal on s2 with radius
def xy_to_sphere(p,sr):
    x,y,x2,y2,r = p[0],p[1],p[0]**2,p[1]**2,sr
    r2,rx,ry = r**2,r*x,r*y
    return [2*rx / (x2 + y2 + (4*r2)),2*ry/(x2+y2+(4*r2)),((2*r)-(x2+y2))/(x2+y2+(4*r2))]

#return a maping function for an elipsoid with its antipodal on s2 with radius
"""
+y / -y	Moves ellipsoid up/down (north/south).
+x / -x	Moves ellipsoid left/right (east/west).
+z / -z	Moves ellipsoid forward/backward.
hypot(+x, +y) / hypot(-x, -y)	Tilts ellipsoid diagonally (northeast/southwest).
hypot(+x, -y) / hypot(-x, +y)	Tilts ellipsoid diagonally (northwest/southeast).
"""















