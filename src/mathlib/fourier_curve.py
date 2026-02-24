from math import tau, cos, sin
import numpy as np
import cmath
from DataFile import DataFile
from utils import proj_file_path


def cyclic_fourier(t:float,a,b,phi,psi,M) -> complex:
    coef_lens = [len(a),len(b),len(phi),len(psi)]
    assert all(x == coef_lens[0] for x in coef_lens), "all coef lists must be of equal length"
    rt = a[0]
    theta_t = tau * M * t
    N = len(a)
    for k in range(1, N):
        rt += a[k] * cos(tau*k*t + phi[k])
        theta_t += b[k] * sin(tau*k*t + psi[k])
    gamma_t = rt * cmath.exp(1j*theta_t)
    #point = [gamma_t.real,gamma_t.imag]
    return gamma_t

def tessellate_fourier_segments(a,b,phi,psi,M,L,res,**kwargs):
    tv = np.linspace(0, 1, res)
    segments = []
    for k in range(L):
        rot = cmath.exp(1j * tau * k / L)
        arr = []
        for t in tv:
            v = rot * cyclic_fourier(t,a,b,phi,psi,M)
            pt = np.array([v.real,v.imag])
            arr.append(pt)
        segments.append(np.asarray(arr))
    return segments

def tessellate_fourier_pts(a,b,phi,psi,M,L,res,**kwargs):
    tv = np.linspace(0, 1, res)
    segments = []
    for k in range(L):
        rot = cmath.exp(1j * tau * k / L)
        arr = []
        for t in tv:
            v = rot * cyclic_fourier(t,a,b,phi,psi,M)
            pt = np.array([v.real,v.imag])
            arr.append(pt)
        segments.append(np.asarray(arr))
    ret = np.asarray(segments).reshape((res*L,2))
    return ret