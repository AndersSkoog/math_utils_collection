from math_utils import is_num_between, is_num, normalize_vector, tau, ang_theta, ang_phi
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import math
from sphere import SpherePoint, stereographic_projection, plane_to_sphere_point
from SphereImagePerspective import SphereImagePerspective



fig = plt.figure()
ax3d = fig.add_subplot(1,2,1, projection='3d')
ax2d = fig.add_subplot(1,2,2)
# Create slider axes *after* adjusting layout
ax_slider_1 = plt.axes([0.2, 0.1, 0.65, 0.03])
ax_slider_2 = plt.axes([0.2, 0.03, 0.65, 0.03])

# Create sliders
slider1 = Slider(ax_slider_1, 'theta', -180, 180, valinit=0, valstep=1)
slider2 = Slider(ax_slider_2, 'phi', -90, 90, valinit=0, valstep=1)

plt.subplots_adjust(bottom=0.5)

R = 1
C = [0,0,0]
theta_val = 0
phi_val = 20
sphere = plane_to_sphere_point(1.2,0.6,2)
lim = R + 1
ax2d.set_xlim(-lim,lim)
ax2d.set_ylim(-lim,lim)
ax3d.set_xlim(-lim,lim)
ax3d.set_ylim(-lim,lim)
ax3d.set_zlim(-lim,lim)
ax3d.set_box_aspect([1, 1, 1])



def plot_line(sp:SpherePoint):
    ax3d.plot(sp.line_seg[0], sp.line_seg[1], sp.line_seg[2], color="blue", linewidth=2)


def plot_points(sp:SpherePoint):
    px, py, pz = sp.coord[0], sp.coord[1], sp.coord[2]
    ax3d.plot(px, py, pz, color="blue", marker="o")
    ax3d.plot(-px, -py, -pz, color="red", marker="o")
    ax3d.plot(-px, -py, pz, color="blue", marker="o")
    ax3d.plot(px, py, -pz, color="red", marker="o")
    ax3d.plot(sp.center[0], sp.center[1], sp.center[2], color="black", marker="o")


def plot_circles(sp:SpherePoint, lw):
    px, py, pz = zip(*sp.xy_circ)
    apx, apy, apz = zip(*sp.ap_xy_circ)
    ax3d.plot(px, py, pz, color="black", linewidth=lw)
    ax3d.plot(apx, apy, apz, color="black", linewidth=lw)


def plot_elipses(sp:SpherePoint, lw):
    el1_x, el1_y, el1_z = zip(*sp.elipses[0])
    el2_x, el2_y, el2_z = zip(*sp.elipses[1])
    el3_x, el3_y, el3_z = zip(*sp.elipses[2])
    el4_x, el4_y, el4_z = zip(*sp.elipses[3])
    ax3d.plot(el1_x, el1_y, el1_z, color="cyan", linewidth=lw)
    ax3d.plot(el2_x, el2_y, el2_z, color="cyan", linewidth=lw)
    ax3d.plot(el3_x, el3_y, el3_z, color="cyan", linewidth=lw)
    ax3d.plot(el4_x, el4_y, el4_z, color="cyan", linewidth=lw)


def plot_great_circles(sp:SpherePoint, lw):
    px1, py1, pz1 = zip(*sp.great_circ1)
    px2, py2, pz2 = zip(*sp.great_circ2)
    ax3d.plot(px1, py1, pz1, color="orange", linewidth=lw)
    ax3d.plot(px2, py2, pz2, color="orange", linewidth=lw)


def plot_sphere(sp:SpherePoint):
    theta_vals = np.linspace(0, tau, 360)
    phi_vals = np.linspace(0, np.pi, 360)
    mesh_theta, mesh_phi = np.meshgrid(theta_vals, phi_vals)
    sphere_x = sp.center[0] + sp.r * np.sin(mesh_theta) * np.cos(mesh_phi)
    sphere_y = sp.center[1] + sp.r * np.sin(mesh_theta) * np.sin(mesh_phi)
    sphere_z = sp.center[2] + sp.r * np.cos(mesh_theta)
    ax3d.plot_surface(sphere_x, sphere_y, sphere_z, color='#FF5733', alpha=0.2)  # Orange color

def set_axes_equal():
    """Set equal aspect ratio for a 3D plot so the sphere looks correct."""
    limits = np.array([ax3d.get_xlim(), ax3d.get_ylim(), ax3d.get_zlim()])
    min_limit, max_limit = limits.min(), limits.max()
    ax3d.set_xlim([min_limit, max_limit])
    ax3d.set_ylim([min_limit, max_limit])
    ax3d.set_zlim([min_limit, max_limit])


def plot_ellipses_on_complex_plane(sp: SpherePoint):
    curve1 = [stereographic_projection(p) for p in sp.elipses[0]]
    curve2 = [stereographic_projection(p) for p in sp.elipses[1]]
    #curve3 = [stereographic_projection(p) for p in sp.elipses[0]]
    #curve4 = [stereographic_projection(p) for p in sp.elipses[1]]
    curves = [curve1,curve2]
    for curve in curves:
      xs = [z.real for z in curve if not np.isinf(z)]
      ys = [z.imag for z in curve if not np.isinf(z)]
      ax2d.plot(xs,ys)


def update_theta(val):
    elev, azim = ax3d.elev, ax3d.azim  # Store camera angle
    ax3d.clear()
    ax2d.clear()
    global sphere,theta_val,phi_val
    theta_val = val
    sphere = SpherePoint(R,theta=theta_val,phi=phi_val,center=C)
    plot_points(sphere)
    plot_line(sphere)
    plot_sphere(sphere)
    plot_elipses(sphere,lw=2)
    plot_circles(sphere,lw=2)
    plot_great_circles(sphere,lw=2)
    plot_ellipses_on_complex_plane(sphere)
    set_axes_equal()
    ax3d.view_init(elev=elev, azim=azim)
    plt.draw()

def update_phi(val):
    elev, azim = ax3d.elev, ax3d.azim  # Store camera angle
    ax3d.clear()
    ax2d.clear()
    global sphere,theta_val,phi_val
    phi_val = val
    sphere = SpherePoint(R,theta=theta_val,phi=phi_val,center=C)
    plot_line(sphere)
    plot_points(sphere)
    plot_sphere(sphere)
    plot_elipses(sphere,lw=2)
    plot_circles(sphere,lw=2)
    plot_great_circles(sphere,lw=2)
    plot_ellipses_on_complex_plane(sphere)
    set_axes_equal()
    ax3d.view_init(elev=elev, azim=azim)
    plt.draw()


slider1.on_changed(update_theta)
slider2.on_changed(update_phi)
plt.ion()  # Enable interactive mode
plt.show(block=True)