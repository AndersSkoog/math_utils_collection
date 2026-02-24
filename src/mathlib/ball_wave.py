import numpy as np
from math import cos, sin, pi, tau, sqrt
#from num import is_even

def sphere_circle_radius(R,azi):return R * cos(azi)
def sphere_circle_diameter(R,azi): return 2*sphere_circle_radius(R,azi)
def midpoint(a,b,z):
 ax,ay = a[0],a[1]
 bx,by = b[0],b[1]
 return [ax+bx/2,ay+by/2,z]

def sphere_to_cart(c,au="deg"):
  assert au in ("deg","rad"), "invalid format string"
  r = c[0]
  theta = np.deg2rad(c[1]) if au == "deg" else c[1]
  phi = np.deg2rad(c[2]) if au == "deg" else c[2]
  x = r * np.sin(phi) * np.cos(theta)
  y = r * np.sin(phi) * np.sin(theta)
  z = r * np.cos(phi)
  return np.array([x,y,z])


def sc_z(sp,au="deg"):
  assert au in ("deg","rad"), "invalid format string"
  phi = np.deg2rad(sp[2]) if au == "deg" else sp[2]
  return sp[0] * np.cos(phi)

#radius of circle at sphere coordinate
def sc_circ_radius(sp,au="deg"):
  assert au in ("deg","rad"), "invalid format string"
  phi = np.deg2rad(sp[2]) if au == "deg" else sp[2]
  r = sp[0]
  z = r * np.cos(phi)
  return np.sqrt((r*r)-(z*z))

def sc_circ_diam(sp,au="deg"): return sc_circ_radius(sp,au) * 2

def sc_coord_vars(sc,au="deg"):
  assert au in ("deg","rad"), "invalid format string"
  R = sc[0]
  theta_deg = sc[1] if au == "deg" else np.rad2deg(sc[1])
  theta_rad = np.deg2rad(sc[1]) if au == "deg" else sc[1]
  phi_deg = sc[2] if au == "deg" else np.rad2deg(sc[2])
  phi_rad = np.deg2rad(sc[2]) if au == "deg" else sc[2]
  x = R * sin(phi_rad) * cos(theta_rad)
  y = R * sin(phi_rad) * sin(theta_rad)
  z = R * cos(phi_rad)
  return {"R":R,"phi_deg":phi_deg,"phi_rad":phi_rad,"theta_rad":theta_rad,"theta_deg":theta_deg,"x":x,"y":y,"z":z}

"""
def arc_ball_dimple(sc,res=180,au="deg"):
  cv = sc_coord_vars(sc,au)
  assert 0 <= cv["phi_deg"] <= 90, "phi angle must be between 0 and 90 degrees"
  d = cv["R"] - cv["z"]
  cr = d / 2
  circ_cz = cv["R"] + cr

  pts = [[cr*cos(np.deg2rad(a)),0,cr*sin(np.deg2rad(a))] for a in range(ang_start,ang_end)]
  return pts
"""

def deg2rad(ang:int):return tau*((1/360) * ang)


def arc_angles(ang):
  assert 0 < ang <= 90, "ang must be between 0 and 90deg"
  return [np.deg2rad(i) for i in range(ang,360-ang)]

def dimp_arc(R,ang:int):
  assert 0 < ang <= 90, "angle must be between 1 - 90"
  a1,a2 = ang,360-ang
  a3,a4 = 90+ang,270-ang
  arc_angs = [deg2rad(a) for a in range(a3,a4)]
  x = R*cos(deg2rad(a1))
  y = R*sin(deg2rad(a1))
  d = R-y
  c = [0,y*2]
  circ_pts = [[R*cos(a),R*sin(a)] for a in np.linspace(0,tau,360)]
  arc_pts = [[x*cos(deg2rad(aa)),R+x*sin(aa)] for aa in arc_angs]
  return arc_pts,circ_pts


def cp(R,ang): return [R*cos(ang),R*sin(ang)]
def c_arc_r(R,ang): return R - (R*sin(ang))
def c_arc_pos(R,ang): return [0,c_arc_r(R,ang)*2]

"""
def dimp_wave(R,ang,res):
  assert 0 <= ang
  assert 0 <= i <= res, "index out of range"
  assert is_even(res), "res must be an even integer"
  if i in (0,res): return R
  hres = int(res/2)
  li,hi,prev_i = (0,hres-1),(hres,res),i-1
  dwn = i in li or prev_i in li
  z = R * sin(phi)
  d,t = R-z,(1/res)*i
  a = tau*t
  w = d*sin(a)
  return R-w if dwn else z+w
"""


def circular_arc_dimple(r, z, res=180):
  x_max = sqrt(pow(r,2) - pow(z,2))
  d = r - z
  # circle radius from geometry
  cr = (pow(x_max,2) + pow(d,2)) / (2 * d)
  yc = z + r
  xs = np.linspace(-x_max, x_max, res)
  return [[x,0,yc-sqrt(pow(r,2)-pow(x,2))] for x in xs]


def ball_dimple_variables(c,au):
  ret = sc_coord_vars(c,au)
  ret["d"]  = ret["R"] - ret["z"]
  ret["cr"] = ret["R"] * cos(ret["phi_rad"])
  ret["cd"] = ret["cr"] * 2
  ret["x_max"] = sqrt(pow(ret["R"],2)-pow(ret["z"],2))
  return ret

"""
def ball_dimple_cosine_angle(dimp_vars,t):
  assert -1 <= t <= 1, "t must be between -1 and 1"
  cr,cd = dimp_vars["cr"],dimp_vars["cd"]
  x = t * dimp_vars["cr"]
  return pi * (x+cr)/cd

def ball_dimple_cosine_val(dimp_vars,t):
  r,d = dimp_vars["r"], dimp_vars["d"]
  a = ball_dimple_cosine_angle(dimp_vars,t)
  return r * abs(cos(a))

def ball_dimple_cosine_pts_2d(sc,res,au="deg"):
  dvars = ball_dimple_variables(sc,au)
  ts = np.linspace(-1,1,res)
  return [[t*dvars["cr"],ball_dimple_cosine_val(dvars,t)] for t in ts]

def ball_dimple_cosine_pts_3d(sc,res,au="deg"):
  dvars = ball_dimple_variables(sc,au)
  ts = np.linspace(-1,1,res)
  return [[t*dvars["cr"],0,ball_dimple_cosine_val(dvars,t)] for t in ts]

def cos_curve_2d(sc,au):
  dvars = ball_dimple_variables(sc,au)
  cr = dvars["cr"]
  d = dvars["d"]
  ts = np.linspace(-1,1,180)
  return [[t*cr,d*abs(cos(t*pi))] for t in ts]

def cos_curve_3d(sc,au):
  dvars = ball_dimple_variables(sc,au)
  r = dvars["r"]
  z = dvars["z"]
  cr = dvars["cr"]
  d = dvars["d"]
  x_max = dvars["x_max"]
  x_vals = np.linspace(-x_max, x_max, 180)
  angs = [pi*x/x_max for x in x_vals]
  y_vals = []
  R - dz * (1 + cos(pi * x / x_max)) / 2
  print(d,z,cr)
  cos_vals = [z*abs(cos(np.deg2rad(a))) for a in range(180)]
  print(d)
  x_vals = np.linspace(-x_max,x_max,180)
  return [[x_vals[i],0,r*cos_vals[i]] for i in range(180)]
"""



def ball_dimple_domain(R,z):
 x_max = sqrt(pow(R,2)-pow(z,2))
 return [-x_max,x_max]

def ball_dimple_value(R,z,t):
  x_max = sqrt(pow(R, 2) - pow(z, 2))
  # t can be a normalized value between -1 and 1
  x = t * x_max
  dz = R - z
  return R - dz * (1 * cos(pi * x / x_max)) / 2

"""
def ball_cosine_val(r,d,cr,t):
  assert -1 <= t <= 1, "t must be between -1 and 1"
  x = t * cr
  cd = cr*2
  return (r-d) * abs(cos(pi*((x+cr)/cd)))

def ball_cosine_pts_2d(sc,res,au="deg"):
  c = coord(sc,au)
  x,y,z,r,phi = c["x"],c["y"],c["z"],c["r"],c["phi"]
  d = r - z
  cr = r * cos(phi)
  tx = [t*cr for t in np.linspace(-1,1,res)]
  return [[x,r-d*ball_dimple_cosine_angle(r,phi,x)] for x in tx]

def ball_dimple_curve_3d(R,z,res):
 x_max = sqrt(pow(R,2)-pow(z,2))
 dom = ball_dimple_domain(R,z)
 phases = np.linspace(dom[0],dom[1],res)
 dz = R - z
 return [[x,0,R-dz*(1+cos(pi*x/x_max))/2] for x in phases]

def ball_dimple_curve_2d(R,z,res):
  x_max = sqrt(pow(R, 2) - pow(z, 2))
  phases = np.linspace(-x_max, x_max, res)
  dz = R - z
  return [[x,R - dz * (1 + cos(pi * x / x_max)) / 2] for x in phases]



if __name__ == "__main__":
  from PlotContext import PlotContext
  pctx = PlotContext("ball_wave",proj="2d")
  ts = np.linspace(-1,1,180)
  sc = [1,4,43]
  darc = dimp_arc(sc[1],43)
  #dvars = ball_dimple_variables(sc,"deg")
  #tx = np.linspace(-dvars["x_max"],dvars["x_max"],180)
  #pts = [[tx[i],0,ball_dimple_value(sc[0],dvars["z"],ts[i])] for i in range(180)]
  pctx.plot_pointlist(darc,"black",0.3)
  pctx.run()
"""
