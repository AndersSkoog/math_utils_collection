"""
A discrete geometric knot is defined from a sequence of K points each associated with a selected point index from a circle of resolution 'res' (e.g., 360 samples),
Each provided spherical coordinate (r, theta, phi) is associated to a distinct index i specifying the point on the cirlce (x=R * sin((tau/res)*i),y=R * cos((tau/res)*i))
where R is the radius of the bounding circle and the bounding ball within which the knot is confined.
The knot is constructed as a cyclic piecewise Bezier curve of degree N (parameter, must divide K) where K is the number of coordinates that should define the knot.
For a knot to be valid it no point along a beizer curve segment can intersect, the Validity check is enforced as follows:
- For every pair of non-adjacent points, compute the angular difference between their radial direction vectors.
- If the angular difference exceeds a the threshold epsilon, the pair is valid.
- If the angular difference is below epsilon, the pair is valid only if their radial distances differ by more than a threshold rho.
- Otherwise, the knot is invalid due to potential self-intersection.
the epsilon parameter thus encodes the "thickness" of the string of the knot,
and the parameter rho encodes the maximal steepness or acuteness of the curve between adjacent curve segment points.
The 2D projection (shadow) of the knot is obtained by orthographically projecting each 3D point along the bezier curve segments onto the disc bounded by the orginal circle in the XY plane (discarding the Z component).
The projection is guaranteed to lie within a disc of radius R.
"""
import itertools
from num import is_even
import numpy as np
from math import tau, sin, cos, comb, factorial
from itertools import combinations, permutations, product

t100 = np.linspace(0,1,100)
t500 = np.linspace(0,1,500)
circ_angles = np.linspace(0,tau,360)


presets = {
  "sym1": {
    "cind":[0, 60, 120, 180, 240, 300],
    "radii":[0.4, 0.6, 0.5, 0.6, 0.4, 0.5],
    "theta_angles":[0, 60, 120, 180, 240, 300],
    "phi_angles":[30, 150, 30, 150, 30, 150]
  }
}


def circ_pt(R,i):
  assert i < 360, "i must be between 0 and 359"
  return np.array([R*cos(circ_angles[i]),R*sin(circ_angles[i]),0.0])


def sphere_to_cart(R,c,angle_unit="deg"):
  assert c[0] < R, "r must be less than R"
  r = c[0]
  theta = np.deg2rad(c[1]) if angle_unit == "deg" else c[1]
  phi = np.deg2rad(c[2]) if angle_unit == "deg" else c[2]
  x = r * sin(phi) * cos(theta)
  y = r * sin(phi) * sin(theta)
  z = r * cos(phi)
  return np.array([x,y,z])




def normalize_vector(vec) -> np.array:
  ret = np.array(vec)
  return ret / np.linalg.norm(ret)


def bezier_pt_3d(t, pts):
  n = len(pts) - 1
  x = 0.0
  y = 0.0
  z = 0.0
  for k in range(n + 1):
    bern = comb(n, k) * (t ** k) * ((1 - t) ** (n - k))
    x += bern * pts[k][0]
    y += bern * pts[k][1]
    z += bern * pts[k][2]
  return [x,y,z]

def bezier_curve_3d(pts,ts=t100):
  return np.array([bezier_pt_3d(t, pts) for t in ts])


def is_valid(knot_points,radial_vectors,epsilon,rho):
    K = len(knot_points)
    # Generate all non-adjacent pairs (cyclic: skip neighbors and first-last)
    pairs = [(i, j) for i, j in combinations(range(K), 2) if abs(i - j) > 1 and not (i == 0 and j == K - 1)]
    radii = [c[0] for c in knot_points]
    # Check if any pair violates epsilon/rho
    return all(
        np.linalg.norm(radial_vectors[i] - radial_vectors[j]) > epsilon
        or abs(radii[i] - radii[j]) > rho
        for i, j in pairs
    )

def int_seq_to_angles(seq_name,cnt):
  assert cnt < len(int_seq_names), "index out of bounds"
  seq_vals = get_seq_from_file(seq_name)[:cnt]
  return [np.deg2rad(seq_val) for seq_val in seq_vals]


def geometrical_knot(R,radii,theta_angles,phi_angles,cind,perm:int,d:int,eps,rho,res=360,**kwargs):
  seg_cnt = len(radii)
  perm_cnt = factorial(seg_cnt)
  assert all([len(l) == seg_cnt for l in [radii,theta_angles,phi_angles,cind]]), "list length mismatch"
  assert all([i <= res for i in cind]), "indices must be lower than circle resolution"
  assert d == 1 or d == -1, "d must be 1 or -1"
  assert perm < perm_cnt, f"chosen permutation must be > {perm_cnt} for a knot with {len(cind)} displacements"
  out_indices = [v+1 for v in cind]
  conn = list(permutations(out_indices))[perm]
  out = []
  for i in range(res):
    if i in cind:
      seg_ind = cind.index(i)
      r,theta,phi = radii[seg_ind],theta_angles[seg_ind],phi_angles[seg_ind]
      in_ind = i - 1 if i != 0 else 359
      out_ind = conn[seg_ind]
      in_pt = [R*cos(np.deg2rad(in_ind)),R*sin(np.deg2rad(in_ind)),0.0]
      disp_pt = sphere_to_cart(R,[r,theta,phi],"deg")
      out_pt = [R*cos(np.deg2rad(out_ind)),R*sin(np.deg2rad(out_ind)),0.0]
      seg = bezier_curve_3d([in_pt,disp_pt,out_pt])
      out.extend(seg)
    else: circ_pt(R,i)


  return np.array(out)