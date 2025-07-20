from math_utils import is_num_between, is_num, normalize_vector, tau
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


def ang_theta(num):
  if is_num_between(num,-np.pi,np.pi): return num
  elif is_num_between(num,-180,180): return np.radians(num)
  else: raise ValueError("must be an angle between 0-tau")

def ang_phi(num):
  if is_num_between(num,-(np.pi/2),np.pi/2): return num
  elif is_num_between(num,-90,90): return np.radians(num)
  else: raise ValueError("must be an angle between 0-pi")

def sphere_coord(arr,center):
  r,theta,phi = is_num(arr[0]), ang_theta(arr[1]), ang_phi(arr[2])
  x = center[0] + np.sin(phi) * np.cos(theta)
  y = center[1] + np.sin(phi) * np.sin(theta)
  z = center[2] + np.cos(phi)
  return np.array([x, y, z])

def orthogonal_vectors(sc):
  ref = np.array([0,1,0])
  if np.allclose(sc, ref) or np.allclose(sc, -ref): ref = np.array([0, 0, 1])
  u = normalize_vector(np.cross(ref,sc))
  v = normalize_vector(np.cross(sc,u))
  return u,v,ref

def stereographic_project(p, north_pole=np.array([0,0,1]), sphere_radius=1):
    x, y, z = p
    denom = 1 - z
    if np.isclose(denom, 0):  # Avoid division by zero at north pole
        return np.inf + 1j * np.inf
    return (x + 1j * y) / denom


def sphere_ellipses(p,center,res=360):
    r1 = math.sqrt(1 - pow(abs(p[0]),2))
    r2 = math.sqrt(1 - pow(abs(p[1]), 2))
    th = np.linspace(0, tau, res)
    x_arr_pos = np.full_like(th,p[0])
    y_arr_pos = np.full_like(th,p[1])
    x_arr_neg = np.full_like(th,-p[0])
    y_arr_neg = np.full_like(th,-p[1])
    x_cos_arr = r1 * np.cos(th)
    x_sin_arr = r1 * np.sin(th)
    y_cos_arr = r2 * np.cos(th)
    y_sin_arr = r2 * np.sin(th)
    left = list(zip(center[0] + x_arr_pos,center[1] + x_cos_arr,center[2] + x_sin_arr))
    right = list(zip(center[0] + x_arr_neg, center[1] + x_cos_arr, center[2] + x_sin_arr))
    front = list(zip(center[0] + y_cos_arr,center[1] + y_arr_pos, center[2] + y_sin_arr))
    back = list(zip(center[0] + y_cos_arr, center[1] + y_arr_neg, center[2] + y_sin_arr))
    return [left,right,front,back]


def sphere_circle(p,r,center,res=360):
    z = p[2]
    #if abs(z) > r: raise ValueError("Point is not on the sphere")
    r_h = np.sqrt(r**2 - z**2)
    return np.array([
        [center[0] + r_h * np.cos(theta), center[1] + r_h * np.sin(theta), center[2] + z]
        for theta in np.linspace(0, 2*np.pi, res)
    ])


def great_circle_1(p,r,center,num_points=360):
    ref = np.array([0, 0, 1])
    if np.allclose(p, ref) or np.allclose(p, -ref):
        ref = np.array([0, 1, 0])  # Fallback if n is near the z-axis
    u = np.cross(ref, p)
    u = u / np.linalg.norm(u)  # Normalize
    v = np.cross(p, u)
    v = v / np.linalg.norm(v)  # Normalize
    t = np.linspace(0, 2 * np.pi, num_points)
    circle_points = r * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    return center + circle_points

def great_circle_2(p,r,center,num_points=360):
    z = np.array([0, 0, 1])
    if np.allclose(np.cross(z, p), 0):
        u = np.array([1, 0, 0])
        v = np.array([0, 1, 0])
    else:
        n = np.cross(z, p)
        n = n / np.linalg.norm(n)
        u = p / np.linalg.norm(p)
        v = np.cross(n, u)
        v = v / np.linalg.norm(v)

    t = np.linspace(0, 2 * np.pi, num_points)
    circle_points = r * (np.outer(np.cos(t), u) + np.outer(np.sin(t), v))
    return center + circle_points



def line_to_sphere_point(center, point_on_sphere, num_points=100):
    t_vals = np.linspace(0, 1, num_points)
    line_points = [(1 - t) * center + t * point_on_sphere for t in t_vals]
    px, py, pz = map(np.array, zip(*line_points))
    return [px,py,pz]


class SpherePoint:
    def __init__(self,r,theta,phi,center,res=360):
        self.center = np.array(center)
        self.res = res
        self.r = r
        self.theta = ang_theta(theta)
        self.phi = ang_phi(phi)
        self.coord = sphere_coord([self.r,self.theta,self.phi],self.center)
        self.ref_coord = np.array([-self.coord[0],-self.coord[1],self.coord[2]])
        self.ap_coord = np.array([-self.coord[0],-self.coord[1],-self.coord[2]])
        self.ref_ap_coord = np.array([-self.ap_coord[0],-self.ap_coord[1],self.ap_coord[2]])
        self.orthogonal_vectors = orthogonal_vectors(self.coord)
        self.u = self.orthogonal_vectors[0]
        self.v = self.orthogonal_vectors[1]
        self.ref = self.orthogonal_vectors[2]
        self.xy_circ = sphere_circle(self.coord,self.r,self.center,self.res)
        self.ap_xy_circ = sphere_circle(self.ap_coord,self.r,self.center,self.res)
        self.great_circ1 = great_circle_1(self.coord,self.r,self.center,self.res)
        self.great_circ2 = great_circle_2(self.coord,self.r,self.center,self.res)
        self.elipses = sphere_ellipses(self.coord,self.center)
        self.line_seg = line_to_sphere_point(self.center,self.coord)


    def plot_line(self):
        ax.plot(self.line_seg[0],self.line_seg[1], self.line_seg[2], color="blue", linewidth=2)

    def plot_points(self):
        px,py,pz = self.coord[0],self.coord[1],self.coord[2]
        ax.plot(px,py,pz,color="blue",marker="o")
        ax.plot(-px,-py,-pz,color="red",marker="o")
        ax.plot(-px,-py,pz,color="blue",marker="o")
        ax.plot(px,py,-pz,color="red",marker="o")
        ax.plot(self.center[0],self.center[1],self.center[2],color="black",marker="o")

    def plot_circles(self,lw):
        px,py,pz = zip(*self.xy_circ)
        apx,apy,apz = zip(*self.ap_xy_circ)
        ax.plot(px,py,pz,color="black",linewidth=lw)
        ax.plot(apx,apy,apz,color="black",linewidth=lw)

    def plot_elipses(self,lw):
        el1_x,el1_y,el1_z = zip(*self.elipses[0])
        el2_x,el2_y,el2_z = zip(*self.elipses[1])
        el3_x,el3_y,el3_z = zip(*self.elipses[2])
        el4_x, el4_y, el4_z = zip(*self.elipses[3])
        ax.plot(el1_x,el1_y,el1_z,color="cyan",linewidth=lw)
        ax.plot(el2_x,el2_y,el2_z,color="cyan",linewidth=lw)
        ax.plot(el3_x,el3_y,el3_z,color="cyan",linewidth=lw)
        ax.plot(el4_x,el4_y,el4_z,color="cyan",linewidth=lw)

    def plot_great_circles(self,lw):
        px1,py1,pz1 = zip(*self.great_circ1)
        px2, py2, pz2 = zip(*self.great_circ2)
        ax.plot(px1,py1,pz1,color="orange",linewidth=lw)
        ax.plot(px2, py2, pz2, color="orange", linewidth=lw)

    def plot_sphere(self):
        theta_vals = np.linspace(0, tau, 360)
        phi_vals = np.linspace(0, np.pi, 360)
        mesh_theta, mesh_phi = np.meshgrid(theta_vals, phi_vals)
        sphere_x = self.center[0] + self.r * np.sin(mesh_theta) * np.cos(mesh_phi)
        sphere_y = self.center[1] + self.r * np.sin(mesh_theta) * np.sin(mesh_phi)
        sphere_z = self.center[2] + self.r * np.cos(mesh_theta)
        ax.plot_surface(sphere_x, sphere_y, sphere_z, color='#FF5733', alpha=0.2)  # Orange color

def plane_to_sphere_point(x,y,side_length):
  r = side_length / 2 # can be the radius value for a spherical coordinate
  ang = np.arctan2(y,x) # can be the theta value for a spherical coordinate
  z = np.sqrt(pow(x, 2) + pow(y, 2))
  zr =  pow(z,2) - pow(z-r,2) # can be the polar angle for the spherical coordinate
  print(zr)
  print(z)
  return SpherePoint(r,ang,z,center=[0,0,r])

R = 1
C = [0,0,0]
theta_val = 0
phi_val = 20
sphere = plane_to_sphere_point(1.2,0.6,2)

def set_axes_equal():
    """Set equal aspect ratio for a 3D plot so the sphere looks correct."""
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    min_limit, max_limit = limits.min(), limits.max()
    ax.set_xlim([min_limit, max_limit])
    ax.set_ylim([min_limit, max_limit])
    ax.set_zlim([min_limit, max_limit])


def update_theta(val):
    elev, azim = ax.elev, ax.azim  # Store camera angle
    ax.clear()
    global sphere,theta_val,phi_val
    theta_val = val
    sphere = SpherePoint(R,theta=theta_val,phi=phi_val,center=C)
    sphere.plot_points()
    sphere.plot_line()
    sphere.plot_sphere()
    sphere.plot_elipses(lw=2)
    sphere.plot_circles(lw=2)
    sphere.plot_great_circles(lw=2)
    set_axes_equal()
    ax.view_init(elev=elev, azim=azim)
    plt.draw()

def update_phi(val):
    elev, azim = ax.elev, ax.azim  # Store camera angle
    ax.clear()
    global sphere,theta_val,phi_val
    phi_val = val
    sphere = SpherePoint(R,theta=theta_val,phi=phi_val,center=C)
    sphere.plot_line()
    sphere.plot_points()
    sphere.plot_sphere()
    sphere.plot_elipses(lw=2)
    sphere.plot_circles(lw=2)
    sphere.plot_great_circles(lw=2)
    set_axes_equal()
    ax.view_init(elev=elev, azim=azim)
    plt.draw()



ax_slider_1 = plt.axes([0.2, 0.1, 0.65, 0.03])
ax_slider_2 = plt.axes([0.2, 0.03, 0.65, 0.03])
slider1 = Slider(ax_slider_1, 'theta', -180, 180, valinit=0, valstep=1)
slider2 = Slider(ax_slider_2, 'phi', -90, 90, valinit=0, valstep=1)
slider1.on_changed(update_theta)
slider2.on_changed(update_phi)

sphere.plot_line()
sphere.plot_points()
sphere.plot_sphere()
sphere.plot_elipses(lw=2)
sphere.plot_circles(lw=2)
sphere.plot_great_circles(lw=2)
set_axes_equal()


lim = R + 1
ax.set_xlim(-lim,lim)
ax.set_ylim(-lim,lim)
ax.set_zlim(-lim,lim)
ax.set_box_aspect([1, 1, 1])
plt.ion()  # Enable interactive mode
plt.show(block=True)
