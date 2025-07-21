import math
import numpy as np
import matplotlib.pyplot as plt
from mesh_geometry import reflections

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

xy_circ_normals = [np.array([1,0,0]),np.array([0,1,0]),np.array([-1,0,0]),np.array([0,-1,0])]
xz_circ_normals = [np.array([0,0,1]),np.array([0,1,0]),np.array([0,0,-1]),np.array([0,-1,0])]


two_d_img = [[-1,0.3]]



def cart_sphere(p):
  u,v = p[0],p[1]
  dnom = 1 + pow(u,2) + pow(v,2)
  x,y,z = (2*u) / dnom, (2*v) / dnom, (-1 + pow(u,2) + pow(v,2)) / dnom
  return [x,y,z]

def sphere_cart(p):
    r,phi,azi = p[0],p[1],p[2]
    x = r * np.sin(phi) * np.cos(azi)
    y = r * np.sin(phi) * np.sin(azi)
    z = r * np.cos(phi)
    return [x, y, z]

def sphere_angle_between(p1,p2):
    phi1 = math.radians(p1[1])
    azi1 = math.radians(p1[2])
    phi2 = math.radians(p2[1])
    azi2 = math.radians(p2[2])
    cos_gamma = (np.sin(phi1) * np.sin(phi2) * np.cos(azi1 - azi2) + np.cos(phi1) * np.cos(phi2))
    clamp = max(min(cos_gamma, 1), -1)
    return math.acos(clamp)

def xy_circle(radius):
    return [np.array([radius*np.cos(math.radians(i)),radius*np.sin(math.radians(i)),0]) for i in range(360)]

def yz_circle(radius):
    return [np.array([0,radius*np.cos(math.radians(i)),radius*np.sin(math.radians(i))]) for i in range(360)]

def xz_circle(radius):
    return [np.array([radius*np.cos(math.radians(i)),0,radius*np.sin(math.radians(i))]) for i in range(360)]

def sphere_normals(radius):
  return list(map(lambda p: radius*p,xy_circ_normals+xz_circ_normals))

def vector_between_points(A, B):
  x,y,z = B[0]-A[0],B[1]-A[1],B[2]-A[2]
  return np.array([x,y,z])

def matplot_circle(point_array):
  x_array,y_array,z_array = [],[],[]
  for p in point_array:
    x_array.append(p[0])
    y_array.append(p[1])
    z_array.append(p[2])
  return [x_array,y_array,z_array]

def matplot_line(start_point,end_point):
  return [
      np.linspace(start=start_point[0],stop=end_point[0],num=100),
      np.linspace(start=start_point[1],stop=end_point[1],num=100),
      np.linspace(start=start_point[2],stop=end_point[2],num=100)
  ]

def convex_ref_angle(ins,norm):
  diff = max(ins,norm) - min(ins,norm)
  if diff == 0: return (ins + 90) % 360
  else: return diff * 2

def concave_raf_angle(ins,norm):
  diff = max(ins,norm) - min(ins,norm)
  return diff * 2 if diff > 0 else norm

def diff(a,b):
    return max(a,b) - min(a,b)

def ref_ang(src,norm):
    neg_ang = 360 - (norm - diff(src,norm)) if (norm - diff(src,norm)) < 0 else (norm - diff(src,norm))
    pos_ang = (norm + diff(src,norm)) % 360
    if src > norm: return neg_ang
    elif src < norm: return  pos_ang
    else: return 0

def reflection(src,norm):
  r = 2 if norm[1] == 1 else 1
  if diff(src[1],norm[1]) > 90 or diff(src[2],norm[2]) > 90: raise ArithmeticError("too large angle")
  else: return [norm,[r,ref_ang(src[1],norm[1]),ref_ang(src[2],norm[2])]]

def plot_point(p):
    c = sphere_cart(p)
    ax.plot(c[0], c[1], c[2], color="purple", linewidth=1, marker="x")

def plot_line(p1,p2):
    #c1,c2 = sphere_cart(p1),sphere_cart(p2)
    line = matplot_line(p1,p2)
    ax.plot(line[0],line[1],line[2],color="black",linewidth=1,linestyle="--")


def great_circle(r, theta, phi, num_points=360):
    # Step 1: Normal vector n from spherical coordinates (theta, phi)
    nx = np.sin(phi) * np.cos(theta)
    ny = np.sin(phi) * np.sin(theta)
    nz = np.cos(phi)
    n = np.array([nx, ny, nz])

    # Step 2: Find orthogonal vectors u and v in the plane perpendicular to n
    # Reference vector (arbitrary, usually z-axis unless n is parallel to it)
    ref = np.array([0, 0, 1])
    if np.allclose(n, ref) or np.allclose(n, -ref):
        ref = np.array([0, 1, 0])  # Fallback if n is near the z-axis

    u = np.cross(ref, n)
    u = u / np.linalg.norm(u)  # Normalize
    v = np.cross(n, u)
    v = v / np.linalg.norm(v)  # Normalize

    # Step 3: Generate points on the great circle
    t = np.linspace(0, 2 * np.pi, num_points)
    circle_points = r * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    return circle_points


def plot_circle(sp,color):
  circle = matplot_circle(great_circle(sp[0],sp[1],sp[2]))
  ax.plot(circle[0], circle[1], circle[2], color=color, linewidth=1)
  #return [[r,] for i in range(360)]

ref_1 = reflection([2,0,180],[1,0,180])


plot_circle([2,0,0],"blue")
plot_circle([2,0,math.radians(90)],"blue")
plot_circle([2,math.radians(90),math.radians(90)],"blue")
plot_circle([1,0,0],"red")
plot_circle([1,0,math.radians(90)],"red")
plot_circle([1,math.radians(90),math.radians(90)],"red")
plot_line([0,0,2],[0,0,1])
plot_line([0,0,1],[2-np.sin(math.radians(90)),0,1])


#plot_circle([2,0,90],"blue")

#plot_circle([2,0,0],[1,0,1],"blue")

#plot_circle([2,90,0],[1,0,1],"blue")


ax.set_xlim(-2, 2)
ax.set_ylim(-2, 2)
ax.set_zlim(-2, 2)
ax.set_box_aspect([1, 1, 1])  # Critical for correct proportions
plt.show(block=True)


"""



  r = (inner_point[0] * 2) % 360
  phi = (inner_point[1] * 2) % 360
  azi = (inner_point[2] * 2) % 360
  return [r,phi,azi]

def outer_to_inner_point(outer_point):
  r = (outer_point[0] / 2) % 360
  phi = (outer_point[1] / 2) % 360
  azi = (outer_point[2] / 2) % 360
  return [r,phi,azi]




in_to_out_ang = lambda deg: int(deg * 3)
out_to_in_ang = lambda deg: int(deg / 3)



c2_xy = matplot_circle(xy_circle(2))
c2_xz = matplot_circle(xz_circle(2))
c2_yz = matplot_circle(yz_circle(2))
c1_xy = matplot_circle(xy_circle(1))
c1_xz = matplot_circle(xz_circle(1))
c1_yz = matplot_circle(yz_circle(1))

first_ray = matplot_line([0,0,2],[0,0,1])
ref_pt_0 = sphere_point(1,0,1)
ref_pt_1 = sphere_point()

ref_pt_1 = [c2_xz[0][30],c2_xz[1][30],c2_xz[2][30]]
ref_pt_2 = [c1_xz[0][10],c1_xz[1][10],c1_xz[2][10]]
ref_pt_3 = [c2_xz[0][340],c2_xz[1][340],c2_xz[2][340]]
ref_pt_4 = [c1_xz[0][320],c1_xz[1][320],c1_xz[2][320]]
ref_pt_5 = [c2_xz[0][20],c2_xz[1][20],c2_xz[2][20]]
ref_pt_6 = [c1_xz[0][40],c1_xz[1][40],c1_xz[2][40]]


ref_ray_1 = matplot_line([0,0,1],ref_pt_1)
ref_ray_2 = matplot_line(ref_pt_1,ref_pt_2)
ref_ray_3 = matplot_line(ref_pt_2,ref_pt_3)
ref_ray_4 = matplot_line(ref_pt_3,ref_pt_4)
ref_ray_5 = matplot_line(ref_pt_4,ref_pt_5)
ref_ray_6 = matplot_line(ref_pt_5,ref_pt_6)

#ref_ray_3 = matplot_line([0,0,1],[0,2*np.sin(math.radians(60)),1])
#ref_ray_4 = matplot_line([0,0,1],[0,-2*np.sin(math.radians(60)),1])

#print(c2_yz[2][61]-c2_yz[2][60])

ax.plot(ref_pt_1[0],ref_pt_1[1],ref_pt_1[2],color="purple",linewidth=1,marker="x")
ax.plot(ref_pt_2[0],ref_pt_2[1],ref_pt_2[2],color="purple",linewidth=1,marker="x")
ax.plot(ref_pt_3[0],ref_pt_3[1],ref_pt_3[2],color="purple",linewidth=1,marker="x")
ax.plot(ref_pt_4[0],ref_pt_4[1],ref_pt_4[2],color="purple",linewidth=1,marker="x")
ax.plot(ref_pt_5[0],ref_pt_5[1],ref_pt_5[2],color="purple",linewidth=1,marker="x")
ax.plot(ref_pt_6[0],ref_pt_6[1],ref_pt_6[2],color="purple",linewidth=1,marker="x")

#ax.plot(ref_pt_4[0],ref_pt_4[1],ref_pt_4[2],color="purple",linewidth=1,marker="x")

ax.plot(ref_ray_1[0],ref_ray_1[1],ref_ray_1[2],color="black",linewidth=1,linestyle="--")
ax.plot(ref_ray_2[0],ref_ray_2[1],ref_ray_2[2],color="black",linewidth=1,linestyle="--")
ax.plot(ref_ray_3[0],ref_ray_3[1],ref_ray_3[2],color="black",linewidth=1,linestyle="--")
ax.plot(ref_ray_4[0],ref_ray_4[1],ref_ray_4[2],color="black",linewidth=1,linestyle="--")
ax.plot(ref_ray_5[0],ref_ray_5[1],ref_ray_5[2],color="black",linewidth=1,linestyle="--")
ax.plot(ref_ray_6[0],ref_ray_6[1],ref_ray_6[2],color="black",linewidth=1,linestyle="--")


ax.plot(first_ray[0],first_ray[1],first_ray[2],color="black",linewidth=1,linestyle="--")


#ax.plot(ref_ray_1[0],ref_ray_1[1],ref_ray_1[2],color="black",linewidth=1,linestyle="--")

ax.plot(c2_xy[0],c2_xy[1],c2_xy[2],color="green", linewidth=1)
ax.plot(c2_yz[0],c2_yz[1],c2_yz[2],color="green", linewidth=1)
ax.plot(c2_xz[0],c2_xz[1],c2_xz[2],color="green",linewidth=1)
ax.plot(c1_xy[0],c1_xy[1],c1_xy[2],color="red", linewidth=1)
ax.plot(c1_yz[0],c1_yz[1],c1_yz[2],color="red", linewidth=1)
ax.plot(c1_xz[0],c1_xz[1],c1_xz[2],color="red", linewidth=1)
for pt in sphere_normals(2):
    ax.plot(pt[0],pt[1],pt[2],color="blue", lw=1, marker='o')

for pt in sphere_normals(1):
    ax.plot(pt[0],pt[1],pt[2],color="blue", lw=1, marker='o')



plt.show(block=True)
"""

