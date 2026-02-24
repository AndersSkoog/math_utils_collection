import numpy as np
import cmath
from math import tan, tau, pi, sin, cos, sqrt

angles = np.linspace(0,tau,360)
def mag(z:complex) -> float: return abs(z)
def arg(z:complex) -> float: return cmath.phase(z)
def norm_s(z:complex) -> float: return mag(z) / sqrt(1.0 + pow(mag(z),2))
def norm_c(z:complex) -> float: return 1.0 / sqrt(1.0 + pow(mag(z),2))

def hopf_zpair(theta: float, phi: float, t: float):
    """
    Returns (z1,z2) on S^3 given Hopf coords (theta,phi,t).
    Safe for all phi in [0, pi].
    """
    e_it = cmath.exp(1j * t)
    z1 = cmath.cos(phi/2) * e_it
    z2 = cmath.sin(phi/2) * cmath.exp(1j*(t + theta))
    return z1, z2



# Convert complex pair (z1,z2) -> unit quaternion components (w,x,y,z)
# q = w + x i + y j + z k
def zpair_to_quaternion(z1: complex, z2: complex):
    w = z1.real
    x = z1.imag
    y = z2.real
    z = z2.imag
    # optionally renormalize to avoid numerical drift
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    if norm == 0:
        return 1.0, 0.0, 0.0, 0.0
    return [w / norm, x / norm, y / norm, z / norm]


def C2_Line_S3_np(theta: float, phi: float, num: int = 200) -> np.ndarray:
    """
    Return S3 circle points corresponding to a complex line from spherical coordinate
    """
    a = tan(phi/2) * cmath.exp(1j*theta)
    m = abs(a)
    s = norm_s(a)
    c = norm_c(a)
    ts = np.linspace(0, tau, num)
    # vectorized creation
    exp_it = np.exp(1j*ts)
    z1 = c * exp_it
    z2 = s * (a/m if m != 0 else 1) * exp_it
    return np.column_stack((z1, z2))  # shape (num, 2)



def C2_Line(theta, phi, num=200):
    """
    Return a parametrization of the complex line through the origin in C^2
    corresponding to the S^2 point (1,theta,phi) of a sphere centered at the origin
    """
    # slope in CP^1
    a = tan(phi / 2) * cmath.exp(1j * theta)
    # parameter along the line
    ts = [(tau * k) / num for k in range(num)]
    line = []
    for t in ts:
        z1 = cmath.exp(1j * t)
        z2 = a * z1
        line.append((z1,z2))
    return line

def C2_Line_S3(theta, phi, num=200):
    """
    Return the circle on S^3 ⊂ C^2
    corresponding to the line through the origin in C^2 with slope a = tan(ϕ/2) e^iθ.
    where θ is the polar angle, and ϕ is the azimuthal angle of a spherical coordinate.
    This is the intersection of the complex line with the 3-sphere.
    """
    a = tan(phi/2) * cmath.exp(1j * theta)
    """
        r = tan(phi/2);
        i = vec2(0,1);
        exp = c_scale(i,theta);
        a = c_exp() 
    """
    m = mag(a)
    s = norm_s(a)
    c = norm_c(a)
    phase_a = a / m if m != 0 else 1  # unit complex number
    ts = np.linspace(0, tau, num)
    circle = []

    for t in ts:
        z1 = c * cmath.exp(1j*t)
        z2 = s * phase_a * cmath.exp(1j*t)
        circle.append((z1, z2))

    return np.array(circle)

#torus_point: given c = |z1|, s = |z2|, and angles u,v return stereographic image in R3
def C2_TorusPoint(c,s,u,v):
    d = (1.0 - s) * sin(v)
    # protect against denom ~ 0 numerically
    if abs(d) < 1e-9: d = 1e-9 if d >= 0 else -1e-9
    x = (c * cos(u)) / d
    y = (c * sin(u)) / d
    z = (s * cos(v)) / d
    return np.array([x,y,z])