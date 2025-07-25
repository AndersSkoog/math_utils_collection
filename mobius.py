import cmath
import math
import functools
from num_utils import normalize as normalize2

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

def mobius_trans(z: complex, a: complex, b: complex, c: complex, d: complex) -> complex:
    return ((a * z) + b) / ((c * z) + d)

def mobius_transl(z: complex, b: complex) -> complex:
    return mobius_trans(z, a=1+0j, b=b, c=0+0j, d=1+0j)

def mobius_scale(z: complex, a: complex):
    return mobius_trans(z, a=a, b=0+0j, c=0+0j, d=1+0j)

def mobius_inv(z):
    if z == 0:
        return complex('inf')
    return mobius_trans(z, a=0+0j, b=1+0j, c=1+0j, d=0+0j)

def mobius_rot(z, ang):
    a = cmath.exp(1j * ang)
    return mobius_trans(z, a=a, b=0+0j, c=0+0j, d=1+0j)

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















