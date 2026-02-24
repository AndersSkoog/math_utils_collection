from math import sin, cos
import numpy as np

# torus_point: given c = |z1|, s = |z2|, and angles u,v return stereographic image in R3
def torus_point(c: float, s: float, u: float, v: float) -> np.array:
    denom = 1.0 - s * sin(v)
    # protect against denom ~ 0 numerically
    if abs(denom) < 1e-9: denom = 1e-9 if denom >= 0 else -1e-9
    X = (c * cos(u)) / denom
    Y = (c * sin(u)) / denom
    Z = (s * cos(v)) / denom
    return np.array([X,Y,Z])


def villarceau(c: float, s: float, alpha: float, angles) -> np.array:
    """
    Villarceau (Hopf) circle corresponding to v = u + alpha
    """
    #us = np.linspace(0, tau, num)
    pts = [torus_point(c, s, u, u + alpha) for u in angles]
    return np.array(pts)

def villarceau_mirror(c: float, s: float, alpha: float, angles) -> np.array:
    """
    Mirrored Villarceau: v = -u + alpha (another slanted circle on the same torus)
    """
    #us = np.linspace(0, tau, num)
    pts = [torus_point(c, s, u, -u + alpha) for u in angles]
    return np.array(pts)

def meridian(c: float, s: float, u_fixed: float, angles) -> np.array:
    #vs = np.linspace(0, tau, num)
    pts = [torus_point(c, s, u_fixed, v) for v in angles]
    return pts

def parallel(c: float, s: float, v_fixed: float, angles) -> np.array:
    pts = [torus_point(c, s, u, v_fixed) for u in angles]
    return np.array(pts)
