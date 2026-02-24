"""
Library of complex grid transformations
"""
import cmath
import numpy as np
from num import is_even

def complex_square_grid(size:int,q=100):
    s = size if is_even(size) else size + 1
    ls = np.linspace(-s//2,s//2,q)
    ret = []
    for i in range(q):
        row = [complex(v,ls[i]) for v in ls]
        ret.append(row)
    #print(ret)
    return np.array(ret)

def cgrid_trans(grid,fn):
  l = len(grid)
  g = np.copy(grid)
  for i in range(l):
      for j in range(l):
          z = g[i][j]
          g[i][j] = fn(z)
  return g

def cgrid_trans_sqr(cgrid):         return cgrid_trans(cgrid,lambda z: pow(z,2))
def cgrid_trans_exp(cgrid):         return cgrid_trans(cgrid,lambda z: cmath.exp(z))
def cgrid_trans_inv(cgrid):         return cgrid_trans(cgrid,lambda z: -1/z)
def cgrid_trans_holomorph(cgrid,k): return cgrid_trans(cgrid,lambda z: (z+k) / z)
def cgrid_scale_down(cgrid,div):    return cgrid_trans(cgrid,lambda z: z / (z/div))
def cgrid_times_i(cgrid,mul):       return cgrid_trans(cgrid,lambda z: z * (1j * mul))
def cgrid_minus_over(cgrid,k):      return cgrid_trans(cgrid,lambda z: -k / z)
def cgrid_over_kz(cgrid,k):         return cgrid_trans(cgrid,lambda z: z / (1 - k*z))
