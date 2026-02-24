import numpy as np
from math import comb, tau, pi, cos, sin
from int_sequences import get_seq_from_file

t100  = np.linspace(0,1,100)
t200  = np.linspace(0,1,200)
tcirc = np.linspace(0,tau,360)

def circle(R): return np.array([[R*cos(t),R*sin(t)] for t in tcirc])

def circ_pt(R,ang,au="deg"):
  assert au in ("deg","rad"), "invalid format string"
  a = ang if au == "rad" else np.deg2rad(ang)
  return [R*cos(a),R*sin(a)]

def circ_pt_3d(R,ang,au="deg"):
  assert au in ("deg","rad"), "invalid format string"
  a = ang if au == "rad" else np.deg2rad(ang)
  return [R*cos(a),R*sin(a),0]

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



def cubic_bezier(P0, P1, P2, P3, n=80):
  t = np.linspace(0, 1, n)[:, None]
  return (
          (1 - t) ** 3 * P0 +
          3 * (1 - t) ** 2 * t * P1 +
          3 * (1 - t) * t ** 2 * P2 +
          t ** 3 * P3
  )

def bezier_pt(t, pts):
  n = len(pts) - 1
  x = 0.0
  y = 0.0

  for k in range(n + 1):
    bern = comb(n, k) * (t ** k) * ((1 - t) ** (n - k))
    x += bern * pts[k][0]
    y += bern * pts[k][1]
  return [x,y]

def bezier_curve(pts,ts=t100):
  return np.array([bezier_pt(t, pts) for t in ts])


"""
We do the pappus braiding between the points A,B,C,-A,-B,-C
on a circle with radius R from the input angles pole + a, pole + b, pole + c

We draw segments of cubic beizer curves on a disc.
The order of the control points of the beizer curves is the pappus circle braiding:

A -> -B -> C -> -A -> B -> -C -> A

Interpreted iteratively we can make variation of curved patterns in the disc:
"""

def pappus_beizer(R, pole_1,pole_2,a,b,c,d):
  ang_a = np.deg2rad(pole_1 + a)
  ang_b = np.deg2rad(pole_1 + b)
  ang_c = np.deg2rad(pole_1 + c)
  ang_d = np.deg2rad(pole_2 + d)
  A = np.array([R * cos(ang_a), R * sin(ang_a)])
  B = np.array([R * cos(ang_b), R * sin(ang_b)])
  C = np.array([R * cos(ang_c), R * sin(ang_c)])
  D = np.array([R * cos(ang_d), R * sin(ang_d)])
  s1 = bezier_curve([A,-B,C])
  s2 = bezier_curve([C,-A,B])
  s3 = bezier_curve([B,-C,D])
  return np.concatenate((s1,s2,s3))

"""
def itered_pappus_beizer(R,poles_seq_name,angs_seq_name,**kwargs):
  poles = get_seq_from_file(poles_seq_name)
  angs = get_seq_from_file(angs_seq_name)
  angs_cnt = len(angs)
  poles_cnt = len(poles)
  #assert (offset + it + 3) < angs_cnt, "index out of range"
  #assert (offset + it + 3) < poles_cnt, "index out of range"
  fn = lambda i: pappus_beizer(R,poles[i],poles[i+1%poles_cnt],angs[i],angs[i+1],angs[i+2],angs[i+3])
  return np.array([fn(i) for i in range(min(angs_cnt,poles_cnt)-4)])
"""

def cubic_bezier(P0, P1, P2, P3, n=80):
  t = np.linspace(0, 1, n)[:, None]
  return (
          (1 - t) ** 3 * P0 +
          3 * (1 - t) ** 2 * t * P1 +
          3 * (1 - t) * t ** 2 * P2 +
          t ** 3 * P3
  )

def pappus_bezier_tan(R, pole, angs):
    # angs: length ≥ 3, DO NOT sort
    th = np.deg2rad(pole + np.array(angs[:3]))

    A = R * np.array([np.cos(th[0]), np.sin(th[0])])
    B = R * np.array([np.cos(th[1]), np.sin(th[1])])
    C = R * np.array([np.cos(th[2]), np.sin(th[2])])

    A_, B_, C_ = -A, -B, -C

    pts = [A, B_, C, A_, B, C_, A]

    curves = []
    for P, Q in zip(pts[:-1], pts[1:]):
        # perpendicular tangents (simple, works)
        tP = np.array([-P[1], P[0]])
        tQ = np.array([-Q[1], Q[0]])
        tP /= np.linalg.norm(tP)
        tQ /= np.linalg.norm(tQ)

        curves.append(
            cubic_bezier(
                P,
                P + 0.3 * tP,
                Q - 0.3 * tQ,
                Q
            )
        )

    curve = np.vstack(curves)

    return curve, A, C_   # IMPORTANT

def itered_pappus_bezier(R, poles, angs):
    all_curves = []

    anchor = None  # this will hold C′ₖ

    for i in range(len(poles)):
        curve, A, Cprime = pappus_bezier_tan(
            R,
            pole=poles[i],
            angs=angs[i:i+3]
        )

        if anchor is not None:
            delta = anchor - A
            curve = curve + delta
            Cprime = Cprime + delta

        all_curves.append(curve)
        anchor = Cprime   # THIS is the only “connection”

    return np.concatenate(all_curves)

def pappus_bezier(R,pole,A,B,C):
  a,b,c = np.deg2rad(pole + A), np.deg2rad(pole + B), np.deg2rad(pole + C)
  p1,p2,p3 = [R*cos(a),R*sin(a)],[R*cos(b),R*sin(b)],[R*cos(c),R*sin(c)]
  p4,p5,p6 = [R*cos(-a),R*sin(-a)],[R*cos(-b),R*sin(-b)],[R*cos(-c),R*sin(-c)]
  return bezier_curve([p1,p5,p3,p4,p2,p6,p1])


if __name__ == "__main__":
  from PlotContext import PlotContext
  #from int_sequences import centered_mgon_pyramid, dodecahedral_number
  #octagula_seq = [stella_octangula_number(i) for i in range(0,10)]
  #tetrahedral_seq = [tetrahedral_numbers(i) for i in range(0,10)]
  #pyr_seq = [centered_mgon_pyramid(2,i) for i in range(10)]
  #dod_seq = [dodecahedral_number(i) for i in range(13)]
  plotter = PlotContext("pappus beizer")
  R = 0.1
  angs = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360]
  poles = [0, 15, 30, 45, 60, 75, 90]
  pts = itered_pappus_bezier(R,poles,angs)
  pts_single = pappus_bezier(R,0,15,30,60)

  plotter.plot_pointlist(pts_single)
  plotter.run()
