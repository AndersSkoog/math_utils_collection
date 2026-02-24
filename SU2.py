import numpy as np
import cmath
from math import sin, cos, pi, tau
from S2 import S2_to_R3, xy_circle
from lib import normalize_vector, angles

"""
One construction of elements in SU(2):
An element of SU(2) is a distance traveled from the identity by an amount (a) in a direction specified 
by the cartesian conversion of a point on S² (r,theta,phi).
"""
# -------------------------------------------Implementation-------------------------------------------
# pauli matrices
o1 = np.array([[0, 1], [1, 0]], dtype=complex)
o2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
o3 = np.array([[1, 0], [0, -1]], dtype=complex)
# Identity in SU(2)
I = np.eye(2, dtype=complex)
base_circle = xy_circle


def is_SU2(U, tol=1e-9):
    a, b = U.item(0), U.item(1)
    res = pow(abs(a), 2) + pow(abs(b), 2)
    return np.isclose(res, 1.0, atol=tol) and np.isclose(np.linalg.det(U), 1.0, atol=tol)


def SU2(axis, angle):
    nx, ny, nz = axis
    dotsum = nx * o1 + ny * o2 + nz * o3
    U = np.cos(angle / 2) * I - 1j * np.sin(angle / 2) * dotsum
    assert is_SU2(U), "error"
    return U


def SU2_to_SO3(U):
    """
    Convert an SU(2) matrix into the corresponding SO(3) rotation matrix.
    """
    # Pauli matrices
    pm = [o1, o2, o3]
    # initialize rotation matrix
    rot_mtx = np.zeros((3, 3))
    for i, si in enumerate(pm):
        # conjugate action
        X = U @ si @ U.conj().T
        for j, sj in enumerate(pm):
            # trace inner product
            rot_mtx[j, i] = 0.5 * np.trace(X @ sj).real

    return rot_mtx


def SU2_from_r3_sphere_point(x, y, z):
    # assert point has normal local orientation, ref vector therefore to be the north pole.
    ref_vec = np.array([0, 0, 1])
    sp = np.array([x, y, z])
    rot_axis = np.cross(ref_vec, sp)
    n = np.linalg.norm(rot_axis)
    if n < 1e-9: return I  # already north pole
    rot_axis /= n
    angle = np.arccos(z)
    return SU2(rot_axis, angle)


def S2_to_SU2(sp, amt):
    assert 0.0 < amt <= 1.0, "amt must be a scalar between 0 and 1"
    # theta,phi = sphere_pos[1],sphere_pos[2]
    angle = pi * amt
    rot_axis = normalize_vector(sp)
    return SU2(rot_axis, angle)


# return SU(2) element along a corresponding arc between two unit sphere coordinates
def S2_arc_to_SU2(sp1, sp2, scalar, tol=1e-9):
    u = normalize_vector(S2_to_R3(sp1))
    v = normalize_vector(S2_to_R3(sp2))
    axis = np.cross(u, v)
    norm = np.linalg.norm(axis)
    if norm < tol: return None, 0.0  # same or opposite direction
    axis /= norm
    angle = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
    return SU2(axis, scalar * angle)


def R3_arc_to_SU2(sp1, sp2, scalar, tol=1e-9):
    u = normalize_vector(sp1)
    v = normalize_vector(sp2)
    axis = np.cross(u, v)
    norm = np.linalg.norm(axis)
    if norm < tol: return None, 0.0  # same or opposite direction
    axis /= norm
    angle = np.arccos(np.clip(np.dot(u, v), -1.0, 1.0))
    return SU2(axis, scalar * angle)


# rotate points in R3 by an element in SU(2)
def rotate_points(points, sphere_pos, amt):
    U = S2_to_SU2(sphere_pos, amt)
    RM = SU2_to_SO3(U)
    return points @ RM.T


# rotate points in R3 along a spherical arc between two sphere points
def rotate_points_arc(points, fp, tp, amt):
    U = S2_arc_to_SU2(fp, tp, amt)
    RM = SU2_to_SO3(U)
    return points @ RM.T


# ----------------------DEMO------------------------------------------------------
if __name__ == "__main__":
    from PlotContext import PlotContext
    from tkiter_widgets import FloatSlider

    wid_args = {"th1": 0.01, "ph1": 0.01, "th2": 0.01, "ph2": pi, "amt": 0.0}
    pctx = PlotContext(-1, 1, "SU(2) operations demo", proj="3d")

    from_pos = [1, wid_args["th1"], wid_args["ph1"]]
    to_pos = [1, wid_args["th2"], wid_args["ph2"]]
    from_marker = S2_to_R3(from_pos)
    to_marker = S2_to_R3(to_pos)


    # start_pts = rotate_points(base_circ,from_pos,1.0)

    def plot_demo():
        amt = pi * wid_args["amt"]
        pts = rotate_points_arc(base_circle, from_pos, to_pos, amt)
        pctx.clear()
        pctx.plot_marker(from_marker, 10, "red")
        pctx.plot_marker(to_marker, 10, "blue")
        pctx.plot_pointlist(pts, "black", 0.3)
        # pctx.plot_pointlist(base_circ,"black",0.3)


    def wid_change(_id, val):
        wid_args[_id] = val
        global from_pos, to_pos, from_marker, to_marker
        if _id in ("th1", "ph1"):
            from_pos = [1, wid_args["th1"], wid_args["ph1"]]
            from_marker = S2_to_R3(from_pos)
            # start_pts = rotate_points(base_circ,from_pos,1.0)
        if _id in ("th2", "ph2"):
            to_pos = [1, wid_args["th2"], wid_args["ph2"]]
            to_marker = S2_to_R3(to_pos)

        plot_demo()


    from_theta_slider = FloatSlider(pctx, "th1", "from_theta", 0.01, tau, wid_args["th1"], wid_change)
    from_phi_slider = FloatSlider(pctx, "ph1", "from_phi", 0.01, pi, wid_args["ph1"], wid_change)
    to_theta_slider = FloatSlider(pctx, "th2", "to_theta", 0.01, tau, wid_args["th2"], wid_change)
    to_phi_slider = FloatSlider(pctx, "ph2", "to_phi", 0.01, pi, wid_args["ph2"], wid_change)
    amt_slider = FloatSlider(pctx, "amt", "amount", 0.0, 1.0, wid_args["amt"], wid_change)
    plot_demo()
    pctx.run()

# ----------------------DEMO2------------------------------------------------------
# Demo where a rotated base circle determined by one sphere coordinate
# is rotated towards another sphere coordinate by and amount (scalar of pi)
"""
if __name__ == "__main__":

  from PlotContext import PlotContext
  from tkiter_widgets import IntSlider, FloatSlider

  wid_args = {"th1":0,"ph1":0,"th2":0,"ph2":90,"amt":0.0}
  base_circ = np.array([[cos(a),sin(a),0] for a in np.linspace(0,tau,360)])
  pctx = PlotContext("SU(2) operations demo",proj="3d")

  print(float(wid_args["th1"]))

  def plot_demo():
    th1,ph1 = np.deg2rad(float(wid_args["th1"])), np.deg2rad(float(wid_args["ph1"]))
    th2,ph2 = np.deg2rad(float(wid_args["th2"])), np.deg2rad(float(wid_args["ph2"]))
    amt = pi * wid_args["amt"]
    circ1 = rotate_points(base_circ,th1,ph1,pi)
    circ2 = rotate_points(circ1,th2,ph2,amt)
    from_marker = sphere_to_cart([1,th1,ph1])
    to_marker = sphere_to_cart([1,th2,ph2])
    pctx.clear()
    pctx.plot_marker(from_marker,10,"red")
    pctx.plot_marker(to_marker,10,"blue")
    pctx.plot_pointlist(circ1,"red",0.3)
    pctx.plot_pointlist(circ2, "blue", 0.3)
    #pctx.plot_pointlist(base_circ,"black",0.3)

  def wid_change(_id,val):
      wid_args[_id] = val
      plot_demo()

  from_theta_slider = IntSlider(pctx,"th1","from_theta",-180,180,0,wid_change)
  from_phi_slider = IntSlider(pctx,"ph1","from_phi",-90,90,0,wid_change)
  to_theta_slider = IntSlider(pctx,"th2","to_theta",-180,180,0,wid_change)
  to_phi_slider = IntSlider(pctx,"ph2","to_phi",-90,90,90,wid_change)
  amt_slider = FloatSlider(pctx,"amt","amount",0.0,1.0,0.0,wid_change)
  plot_demo()
  pctx.run()
"""

