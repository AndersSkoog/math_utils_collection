import math
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Create a dark-themed figure
fig = plt.figure()
fig.patch.set_facecolor('#222222')  # Set background color to dark gray
ax = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#222222')  # Set plot background color to dark gray

# Sphere parameters
center = (0, 0, 0)
radius = 1
phi = np.linspace(0, np.pi, 100)
theta = np.linspace(0, 2 * np.pi, 100)

# Create a mesh grid for the sphere
phi, theta = np.meshgrid(phi, theta)
sphere_x = center[0] + radius * np.sin(phi) * np.cos(theta)
sphere_y = center[1] + radius * np.sin(phi) * np.sin(theta)
sphere_z = center[2] + radius * np.cos(phi)
hcirc_x = [math.cos(math.radians(n)) for n in range(360)]
hcirc_y = [math.sin(math.radians(n)) for n in range(360)]
hcirc_z = [0] * 360  # Moves up/down on Z-axis
vcirc_x = [0] * 360  # Moves left/right on X-axis
vcirc_y = [math.cos(math.radians(n)) for n in range(360)]
vcirc_z = [math.sin(math.radians(n)) for n in range(360)]
th = np.linspace(0, 2 * np.pi, 360)

def set_axes_equal():
    """Set equal aspect ratio for a 3D plot so the sphere looks correct."""
    limits = np.array([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()])
    min_limit, max_limit = limits.min(), limits.max()
    ax.set_xlim([min_limit, max_limit])
    ax.set_ylim([min_limit, max_limit])
    ax.set_zlim([min_limit, max_limit])


def update_v(disp):
    elev, azim = ax.elev, ax.azim  # Store camera angle
    r = math.sqrt(1 - pow(abs(disp),2))
    global vcirc_x,vcirc_y,vcirc_z
    vcirc_x = np.full_like(th,disp)
    vcirc_y = r * np.cos(th)
    vcirc_z = r * np.sin(th)
    ax.clear()
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color='#FF5733', alpha=0.5)
    ax.plot(hcirc_x, hcirc_y, hcirc_z, 'b', linewidth=2, label="Up-Down Circle (XY-plane)")
    ax.plot(vcirc_x, vcirc_y, vcirc_z, 'b', linewidth=2, label="Left-Right Circle (YZ-plane)")
    set_axes_equal()
    ax.view_init(elev=elev, azim=azim)
    plt.draw()


def update_h(disp):
    elev, azim = ax.elev, ax.azim  # Store camera angle
    r = math.sqrt(1 - pow(abs(disp),2))
    global hcirc_x,hcirc_y,hcirc_z
    hcirc_x = r * np.cos(th)
    hcirc_y = r * np.sin(th)
    hcirc_z = np.full_like(th,disp)
    ax.clear()
    ax.plot_surface(sphere_x, sphere_y, sphere_z, color='#FF5733', alpha=0.5)  # Orange color
    ax.plot(hcirc_x, hcirc_y, hcirc_z, 'b', linewidth=2, label="Up-Down Circle (XY-plane)")
    ax.plot(vcirc_x, vcirc_y, vcirc_z, 'b', linewidth=2, label="Left-Right Circle (YZ-plane)")
    set_axes_equal()
    ax.view_init(elev=elev, azim=azim)
    plt.draw()



ax_v_slider = plt.axes([0.2, 0.1, 0.65, 0.03])
ax_h_slider = plt.axes([0.2, 0.03, 0.65, 0.03])
vslider = Slider(ax_v_slider, 'updown', -1, 1, valinit=0, valstep=1 / 1000)
hslider = Slider(ax_h_slider, 'leftright', -1, 1, valinit=0, valstep=1 / 1000)
vslider.on_changed(update_v)
hslider.on_changed(update_h)
set_axes_equal()
ax.plot_surface(sphere_x, sphere_y, sphere_z, color='#FF5733', alpha=0.5)  # Orange color
ax.plot(hcirc_x, hcirc_y, hcirc_z, 'b', linewidth=2, label="Up-Down Circle (XY-plane)")
ax.plot(vcirc_x, vcirc_y, vcirc_z, 'b', linewidth=2, label="Left-Right Circle (YZ-plane)")

plt.ion()  # Enable interactive mode
plt.show(block=True)

