"""
Library of curve formulas, Under development, Not correct, Not usable ATM.
"""
from math import cos, sin, pi, tau,cosh,sinh,sqrt,exp
import numpy as np

def lin_interp(v,inmin,inmax,outmin,outmax): return (v - inmin)/(inmax - inmin) * (outmax - outmin) + outmin
def linwin(win:int,res): return np.linspace(-win,win,res)
def algebraic_polar_domain(t,div:int): return lin_interp(t,0,1,-pi/div,pi/div)
def cyclic_polar_domain(t): return lin_interp(t,0,1,0,tau)
def rational_domain(t,win:int): return lin_interp(t,0,1,-win,win)
def win_domain(t, tmin, tmax): return lin_interp(t,0,1,tmin,tmax)

#------------
# DEGREE 2
#------------
def circular(a,t,**kwargs):
    """Circle: x = r*cos(t), y = r*sin(t)"""
    #t = cyclic_polar_domain(res)
    x = a * cos(t)
    y = a * sin(t)
    return [x,y]

def elliptic(a,b,t,**kwargs):
    """Ellipse: x = a*cos(t), y = b*sin(t)"""
    x = a * cos(t)
    y = b * sin(t)
    return [x,y]

def parabolic(a,t,**kwargs):
    """Parabola: y = a*x²"""
    x = t
    y = a * pow(x,2)
    return [x,y]

def hyperbolic(a,b,t,**kwargs):
    """Hyperbola (right branch): x = a*cosh(t), y = b*sinh(t)"""
    x = a * cosh(t)
    y = b * sinh(t)
    return [x,y]

# ---------------------------
# Rational Degree 3
# cubic
# trident
# semiparabola
# folium
# serpemtine
# strophoid
# witch_agnesi
# cissoid

# Polar Degree 3
# trisectrix
# conchoid
# ---------------------------

def cubic_planar(a,t,**kwargs):
    # simplest cubic polynomial y = a*x^3
    x = t
    y = a * pow(x,3)
    return [x,y]

def trident(a,t,**kwargs):
    # Trident curve: y^2 = x^3 + a*x^2 + a*x
    x = t
    y = sqrt(max(0, pow(x,3) + (a*pow(x,2)) + a*x))
    return [x,y]

def semicubical_parabola(a,b,t,**kwargs):
    # y^2 = a^2 * x^3  → y = ±a * x^(3/2)
    x = t
    y = a * x**b
    return [x,y]

def folium_descartes(a,t,**kwargs):
    # Cartesian Folium: x^3 + y^3 = 3*a*x*y
    x = (3*a*t) / (1 + t**3)
    y = (3*a*t**2) / (1 + t**3)
    return [x,y]

def serpemtine(a,t,**kwargs):
    # Serpentine curve: y = x / (1 + x**2)
    x = t
    y = a * x / (1 + x**2)
    return [x,y]

def strophoid(a,t,**kwargs):
    # Strophoid curve: y^2 = x*(a + x)*(a - x)
    x = t
    y = sqrt(max(0, x*(a + x)*(a - x)))
    return [x,y]

def witch_agnesi(a,t,**kwargs):
    # Witch of Agnesi: y = a^3 / (x^2 + a^2)
    x = (4*a) * t
    y = pow(a,3) / (pow(x,2) + pow(a,2))
    return [x,y]

def cissoid(a,b,h,t,**kwargs):
    # y^2 = x^3 / (2*a - x)
    x = lin_interp(t,0,1,-b*a,h*a)
    y = sqrt(max(0, pow(x,3) / (2*a - x)))
    return [x,y]


def limacon(a,t,**kwargs):
    theta = cyclic_polar_domain(t)
    v = (a*(1+2*cos(theta)))
    x = v * cos(theta)
    y = v * sin(theta)
    return [x,y]


def trisect_trichinhaus(a,n,t,**kwargs):
    # Trisectrix of Tschirnhaus (aka cubic Tschirnhaus)
    _t = algebraic_polar_domain(t,n)
    r = a * pow(cos(_t),2) / cos(3*_t)
    x = r * cos(_t)
    y = r * sin(_t)
    return [x,y]


def trisect_maclaurin(a,n,t,**kwargs):
    outmin = -pi/n + 0.01
    outmax = pi/n + 0.01
    _t = lin_interp(t,0,1,outmin,outmax)
    r = a * (4 * pow(cos(_t),3) / (3*pow(cos(_t),2)) + 1)
    x = r * cos(_t)
    y = r * sin(_t)
    return [x,y]


def conchoid_de_sluze(a,b,t,**kwargs):
    # r = a * cos(t) + b / cos(t)
    _t = lin_interp(t,0,1,-pi/3,pi/3)
    r = a * np.cos(_t) + b / np.cos(_t)
    x = r * np.cos(_t)
    y = r * np.sin(_t)
    return [x,y]


# ---------------------------
# RHODONEA / ROSE
# r = a * cos(k t)
# ---------------------------
def rhodonea(a,k,t,**kwargs):
    _t = lin_interp(t,0,1,0,tau)
    r = a * cos(k * _t)
    x = r * cos(_t)
    #y = r * sin(_t)
    return [x,_t]

# ---------------------------
# ASTROID (superellipse special)
# x = a cos^3 t, y = a sin^3 t
# ---------------------------
def astroid(a,tmin,tmax,t,**kwargs):
    _t = win_domain(t,tmin,tmax)
    x = a * pow(cos(_t),3)
    y = a * pow(sin(_t),3)
    return [x,y]

# ---------------------------
# DELTOID (a.k.a. Steiner deltoid)
# param: x = 2R cos t + cos 2t, y = 2R sin t - sin 2t  (scaled variants)
# common normalized form (R=1) -> deltoid with cusp triangle
# ---------------------------
def deltoid(a,t,tmin,tmax,**kwargs):
    _t = win_domain(t,tmin,tmax)
    x = a * (2 * cos(_t) + cos(2*_t)) / 3.0  # scaled so size ~ a
    y = a * (2 * sin(_t) - sin(2*_t)) / 3.0
    return [x,y]

# ---------------------------
# NEPHROID
# classic param x = 3a cos t - cos 3t, y = 3a sin t - sin 3t (scale a)
# ---------------------------
def nephroid(a,t,tmin,tmax,**kwargs):
    _t = win_domain(t,tmin,tmax)
    x = a * (3 * cos(t) - cos(3*t)) / 4.0  # scaling factor to normalize size
    y = a * (3 * sin(t) - sin(3*t)) / 4.0
    return [x,y]

# ---------------------------
# CARDIOID
# a is the fixed circle radius
# b is the radius of rolling circle
# start_angle
# polar: r = a(1 - cos t)  (or 1+cos t); here use 1 - cos t for cusp at origin
# ---------------------------
def cardioid(a,b,t,**kwargs):
    _t = lin_interp(t,0,1,0,tau)
    r = a * (b + cos(_t))
    x = r * cos(_t)
    y = r * sin(_t)
    return [x,y]

# ---------------------------
# TROCHOID  (generalized cycloid)
# param: x = a*t - b*sin(t), y = a - b*cos(t)
# note: a = radius of rolling circle (or step), b = offset of drawing point from center
# ---------------------------
def trochoid(a,b,tmin,tmax,t,**kwargs):
    _t = win_domain(t,tmin,tmax)
    x = (a * _t) - (b * sin(_t))
    y = (a - b) * cos(_t)
    return [x,y]

# ---------------------------
# CYCLOID (special trochoid with b = a)
# x = r(t - sin t), y = r(1 - cos t)
# ---------------------------
def cycloid(a,tmin,tmax,t,**kwargs):
    _t = win_domain(t,tmin,tmax)
    x = a * (_t - sin(_t))
    y = a * (1 - cos(_t))
    return [x,y]

# ---------------------------
# EPICYCLOID / EPITROCHOID / HYPOTROCHOID wrappers
# a = fixed circle radius, b = rolling circle radius, h = offset
# epicycloid: h = b
# hypotrochoid: rolling inside -> use (a-b), sign adjustments handled below
# ---------------------------
def epitrochoid(a,b,h,tmin,tmax,t,**kwargs):
    if h is None: h = b
    _t = win_domain(t,tmin,tmax)
    x = (a + b) * cos(_t) - h * cos(((a + b) / b) * _t)
    y = (a + b) * sin(_t) - h * sin(((a + b) / b) * _t)
    return [x,y]


def hypotrochoid(a,b,h,tmin,tmax,t,**kwargs):
    if h is None: h = b
    _t = win_domain(t,tmin,tmax)
    x = (a - b) * cos(_t) + h * cos(((a - b) / b) * _t)
    y = (a - b) * sin(_t) - h * sin(((a - b) / b) * _t)
    return [x,y]

# ---------------------------
# EPICYCLOID / HYPOCYCLOID quick helpers using n ratio
# ---------------------------
def epicycloid(a,n,tmin,tmax,t,**kwargs):
    b = a / n
    return epitrochoid(a, b, b, tmin,tmax,t)

def hypocycloid(a,n,tmin,tmax,t,**kwargs):
    b = a / n
    return hypotrochoid(a, b, b,tmin,tmax,t)

# ---------------------------
# LISSAJOUS
# x = A sin(p t + delta), y = B sin(q t)
# ---------------------------
def lissajous(a,b,p,q,delta,tmin,tmax,t,**kwargs):
    _t = win_domain(t,tmin,tmax)
    x = a * sin(p * t + delta)
    y = b * sin(q * t)
    return [x,y]


# ---------------------------
# SPIRALS
# archimedean: r = a + b t
# logarithmic: r = a * exp(b t)
# hyperbolic: r = a / t  (avoid t=0)
# fermat: r = a * sqrt(t)
# lituus: r = a / sqrt(t)
# ---------------------------
def archimedean_spiral(a,b,tmin,tmax,t,**kwargs):
    _t = win_domain(t,tmin,tmax)
    r = a + b * _t
    x = r * cos(_t)
    y = r * sin(_t)
    return [x,y]

def logarithmic_spiral(a,b,tmin,tmax,t,**kwargs):
    _t = win_domain(t,tmin,tmax)
    r = a * exp(b * _t)
    x = r * cos(_t)
    y = r * sin(_t)
    return [x,y]


def hyperbolic_spiral(a,tmin,tmax,t,**kwargs):
    _t = win_domain(t,tmin,tmax)
    r = a / t
    x = r * cos(_t)
    y = r * sin(_t)
    return [x,y]


def fermat_spiral(a,tmin,tmax,t,**kwargs):
    _t = win_domain(t,tmin,tmax)
    r = a * sqrt(_t)
    x = r * cos(_t)
    y = r * sin(_t)
    return [x,y]

def lituus_spiral(a,tmin,tmax,t,**kwargs):
    _t = win_domain(t,tmin,tmax)
    r = a / sqrt(_t)
    x = r * cos(_t)
    y = r * sin(_t)
    return [x,y]

curve_dict = {
    "parabolic":parabolic,
    "hyperbolic":hyperbolic,
    "circular":circular,
    "elliptic":elliptic,
    "cubic":cubic_planar,
    "trident":trident,
    "semiparabolic":semicubical_parabola,
    "serpemtine":serpemtine,
    "strophoid":strophoid,
    "agnesi":witch_agnesi,
    "cissoid":cissoid,
    "trisect_trichinhaus":trisect_trichinhaus,
    "trisect_maclaurin":trisect_maclaurin,
    "conchoid_de_sluze":conchoid_de_sluze,
    "limacon":limacon,
    "archimedean_spiral":archimedean_spiral,
    "logarithmic_spiral":logarithmic_spiral,
    "hyperbolic_spiral":hyperbolic_spiral,
    "fermat_spiral":fermat_spiral,
    "lituus_spiral":lituus_spiral,
    "folium":folium_descartes,
    "cardioid":cardioid,
    "epicycloid":epicycloid,
    "hypocycloid":hypocycloid,
    "cycloid":cycloid,
    "trochoid":trochoid,
    "deltroid":deltoid,
    "nephroid":nephroid,
    "astroid":astroid,
    "lissajous":lissajous
}
"""
def composed_cycloid(disc_radius,ratio,arm_len,curve_fn,curve_args,res):
    curve_pts = curve_fn(**curve_args)
    xb,yb = zip(*curve_pts)
    R = disc_radius
    r = R / ratio
    l = len(xb)
    t = np.linspace(0,tau,l)
    phi = (R / r) * t
    x = xb + arm_len * np.cos(phi)
    y = yb + arm_len * np.sin(phi)
    return [[x[i],y[i]] for i in range(l)]
"""

def cycloid_inner(gen_args,curve_args):
    def fn(cfn,t,args):
        _args = args
        _args["t"] = t
        return curve_fn(**_args)

    curve_fn = curve_dict[gen_args["curve_sel"]]
    curve_pts = [fn(curve_fn,t,curve_args) for t in np.linspace(0,1,gen_args["res"])]
    R = gen_args["R"]
    r = R / gen_args["r"]
    res = gen_args["res"]
    al = gen_args["arm_len"]
    cusp_cnt = gen_args["cusp_cnt"]
    cusp_angs = np.linspace(0, tau, cusp_cnt)
    cr = R + 1 * r
    curves = [circular(R,t) for t in np.linspace(0,1,res)]
    for i in range(cusp_cnt):
        curve = []
        for j in range(res):
            curve_val = curve_pts[j]
            ang = cusp_angs[i]
            cx = cr * cos(ang)
            cy = cr * sin(ang)
            phi = (cr / r) * ang
            cusp_ang = cusp_angs[i] + phi
            x = cx + al * cos(cusp_ang) + curve_val[0]
            y = cy + al * sin(cusp_ang) + curve_val[1]
            print(x)
            print(y)
            curve.append([x,y])
        curves.append(curve)
    return curves