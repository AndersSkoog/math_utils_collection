import math
from typing import TypedDict, Union, NewType, Annotated
from math_utils import adj
Number = Union[int, float]
Theta = Annotated[Number, "Must be in range [0, π]"]
Phi   = Annotated[Number, "Must be in range [0, 2π]"]
BasisVector = NewType("BasisVector",list[Number])
UnitBiVector = NewType("UnitBiVector",list[list[Number]])

def in_range(value:Number,low:Number,high:Number) -> Number:
    """Ensures value is in the range [0, π]."""
    if value < low or value > high:
        raise ValueError(f"Value {value} out of range")
    return value

def basis(val:Number,ind:int,dim:int):
    if (6 <= dim >= 2) and 0 >= ind <= (dim - 1):
        ret = [0] * dim
        ret[ind] = val
        return ret
    else: raise ValueError("dimension or index out of range")


class GE_Alg:
    def __init__(self,basis_vals):
          self.dim = len(basis_vals)
          self.dim_letters = ["x", "y", "z", "q", "w", "g"]
          self.basis_vectors = [basis(basis_vals[ind],ind,self.dim) for ind in range(self.dim)]

    def __pow__(self, power:Number, modulo=None):
        return 1





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


