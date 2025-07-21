import itertools
import math

import math_utils
from math_utils import adjacents, spherical_to_cartesian, cart_to_sphere

perm = lambda vals: list(itertools.permutations(vals))
bin_perm = lambda vals,rep: list(itertools.product(vals,repeat=rep))
def switch_sign(v):
    if v != 0: return abs(v) if v < 0 else -v
    else: return v

def reflections(coord):
  x,y,z = coord[0],coord[1],coord[2]
  org_reflection = (switch_sign(x),switch_sign(y),switch_sign(z))
  x_reflection = (x,switch_sign(y),switch_sign(z))
  y_reflection = (switch_sign(x),y,switch_sign(z))
  z_reflection = (switch_sign(x),switch_sign(y),z)
  return [org_reflection,x_reflection,y_reflection,z_reflection]

def reg_polyhedron(n):
    ang = (2*math.pi) / n
    coord = (math.sin(ang),math.cos(ang),0)
    ref_1 = reflections(coord)
    ref_2 = reflections(ref_1[0])
    ref_3 = reflections(ref_1[1])
    ref_4 = reflections(ref_1[2])
    ref_5 = reflections(ref_1[3])
    ret = set([coord] + ref_1 + ref_2 + ref_3 + ref_4)
    return ret

print(reg_polyhedron(5))
print(len(reg_polyhedron(5)))



def reg_polygon(n,r,center=(0,0,0)):
  angs = [math.radians(360/n) * i for i in range(n)]
  return [[round(r * math.cos(ang),5),round(r * math.sin(ang),5), 0] for ang in angs]

def midpoint(vertex_pair):
   dim = len(vertex_pair[0])
   p1,p2 = vertex_pair[0],vertex_pair[1]
   return [round((p1[i] + p2[i]) / 2,5) for i in range(dim)]

def midpoints(vertices):
  vpairs = adjacents(vertices)
  print(vpairs)
  return [midpoint(vp) for vp in vpairs]

def translate_polygon(vertices, center):
    cx, cy, cz = center
    return [(x + cx, y + cy, z + cz) for (x, y, z) in vertices]

pentagon = reg_polygon(5,1)
trans_pentagon = translate_polygon(pentagon, (5,3,0))
print(pentagon)
print(trans_pentagon)

def dodecahedron():
  phi = (1 + math.sqrt(5)) / 2
  return bin_perm((1,-1),3) + perm((0,phi,1/phi)) + perm((0,-phi,-(1/phi)))






