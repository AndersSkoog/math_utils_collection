import numpy as np
from vector_utils import normalize_vector
from math import sin, cos, pi, tau, acos, atan2, atan, tan, prod
from space_conversion import C2_R4

def project_point_to_hyperplane(view_pt, target_pt, normal, d):
    eps = 1/(1<<16)
    v = np.asarray(view_pt, float)
    p = np.asarray(target_pt, float)
    n = np.asarray(normal, float)
    look_dir = v - p
    denom = n @ look_dir
    if abs(denom) < eps: return view_pt  #behind camera
    t = (d - n @ v) / denom
    return v + t * dir

def perspective(pos,target):
    forward = normalize_vector(target - pos)
    world_up = np.array([0, 0, 1], float)
    if abs(np.dot(forward, world_up)) > 0.99: world_up = np.array([0, 1, 0], float)
    right = normalize_vector(np.cross(forward, world_up))
    up = np.cross(right, forward)
    return {"forward":forward,"right":right,"up":up,"persp_pos":pos,"persp_target":target}


def world_to_perspective_point(persp_pos,persp_target,world_pt):
    eps = 1/(1 << 16)
    forward = normalize_vector(persp_target - persp_pos)
    world_up = np.array([0, 0, 1], float)
    if abs(np.dot(forward, world_up)) > 0.99: world_up = np.array([0, 1, 0], float)
    right = normalize_vector(np.cross(forward, world_up))
    up = np.cross(right, forward)
    q = world_pt - persp_pos
    x = np.dot(q,right)
    y = np.dot(q,up)
    z = np.dot(q,forward)
    return np.asarray([x,y,z])


def persp_project_n_to_minus_n(p,f:float,zeros:int,eps=1e-12) -> np.array:
    p = np.array(p)
    if len(p) < 2: raise ValueError("Need at least 2 dimensions for projection.")
    nth_comp = p[-1]
    if nth_comp <= 0: return None #behind camera
    return np.concatenate([np.array([(f * c) / nth_comp for c in p[:-1]]),np.zeros(zeros)])

def central_project_pt(view_pt,target_pt,plane_n,epsilon=1e-12):
    assert np.shape(view_pt) == np.shape(target_pt)
    p = np.asarray(target_pt,dtype=float)
    v = np.asarray(view_pt,dtype=float)
    d = p - v  # direction vector
    t = (plane_n - v[-1]) / d[-1]
    s = v + t * d
    return s[:-1]


def central_project_pts(view_pt,target_pts,plane_n,drop_last=True,epsilon=1e-12):
    assert len(np.shape(view_pt)) == 1, "only projects from a single view point"
    ret = []
    for pt in target_pts: ret.append(central_project_pt(view_pt,pt,plane_n, epsilon))
    #print(ret)
    return np.asarray(ret)


# ---------------------------------------------------------- #
#  Stereographic Projections between C,C2,R2,R3
# ---------------------------------------------------------- #

#  C2 (S3) → R3  stereographic projection from north pole (0,0,0,R)
def stereo_project_C2_R3(p, R=1.0):
    pl = C2_R4(p)
    x, y, z, w = pl[..., 0], pl[..., 1], pl[..., 2], pl[..., 3]

    denom = (R - w)  # correct denominator
    X = (R * x) / denom
    Y = (R * y) / denom
    Z = (R * z) / denom

    return np.stack([X, Y, Z], axis=-1)


#  C2 → C    (Hopf map)
def stereo_project_C2_C(p, R=1.0):
    pl = np.asarray(p, dtype=complex)
    z1 = pl[..., 0]
    z2 = pl[..., 1]
    return z1 / (R - z2)


#  C → R3 (stereographic lift to sphere)
def stereo_project_C_R3(z, R=1.0):
    zl = np.asarray(z, dtype=complex)
    x, y = zl.real, zl.imag

    r2 = x * x + y * y
    d = r2 + R * R

    X = 2 * R * x / d
    Y = 2 * R * y / d
    Z = (r2 - R * R) / d

    return np.stack([X, Y, Z], axis=-1)


#  R2 → R3
def stereo_project_R2_R3(p, R=1.0):
    pl = np.asarray(p, dtype=float)
    x, y = pl[..., 0], pl[..., 1]

    r2 = x * x + y * y
    d = r2 + R * R

    X = 2 * R * x / d
    Y = 2 * R * y / d
    Z = (r2 - R * R) / d

    return np.stack([X, Y, Z], axis=-1)


#  R3 → C
def stereo_project_R3_C(p):
    pl = np.asarray(p, dtype=float)
    x, y, z = pl[..., 0], pl[..., 1], pl[..., 2]

    r = np.sqrt(x * x + y * y + z * z)
    d = (r - z)

    return (x + 1j * y) / d


#  R3 → R2
def stereo_project_R3_R2(p):
    pl = np.asarray(p, dtype=float)
    x, y, z = pl[..., 0], pl[..., 1], pl[..., 2]

    r = np.sqrt(x * x + y * y + z * z)
    d = (r - z)

    return np.stack([x / d, y / d], axis=-1)

