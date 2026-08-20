from math import tau, cos, sin
import numpy as np
import cmath
#from DataFile import DataFile
#from old.draw_beizer import preset_ctrl
#from utils import proj_file_path

def fourier_series_trunc(dc, a, b, k, t):
    """
    Compute a truncated Fourier series up to harmonic k
    dc: float
        The zero-frequency (constant) term.
    a, b: lists or arrays
        Cosine and sine coefficients for all harmonics in the series. Length must be >= num_harmonics.
    k: int
        Number of harmonics to include (1 = fundamental, etc.)
    t: float
        Parameter (can be any real number).
        because the beauty of the fourier series comes from its unit periodicity that will wrap around 1,
        regardless of the value of t
    """
    assert len(a) == len(b), "Coefficient arrays not of same length"
    assert len(a) >= k and len(b) >= k, "Coefficient arrays too short"
    #harmonics = [n for n in range(1,k+1)]
    result = dc + sum([(a[h-1] * cos(tau * h * t)) + (b[h-1] * sin(tau * h * t)) for h in range(1,k+1)])
    return result

def fourier_series(dc,a,b,t):return fourier_series_trunc(dc,a,b,len(a),t)

def scale_coef_to_radius(dc,a,b,R):
  coef_sum = abs(dc) + sum(abs(x) + abs(y) for x, y in zip(a, b))
  scale = R / coef_sum
  dc_scaled = dc * scale
  a_scaled = [x * scale for x in a]
  b_scaled = [y * scale for y in b]
  return dc_scaled,a_scaled,b_scaled

"""
Planar curves functions based on fourier series
"""
def radial_modulation_curve(disc_radius,dc,a,b,res,**kwargs):
  t_vals = np.linspace(0, 1, res)
  dc_scaled,a_scaled,b_scaled = scale_coef_to_radius(dc,a,b,disc_radius)
  r_vals = np.array([fourier_series(dc_scaled, a_scaled, b_scaled, t) for t in t_vals])
  return [[r * cos(tau * t),r * sin(tau*t)] for r,t in zip(r_vals,t_vals)]

def angle_modulation_curve(disc_radius,dc,phi,psi,M,res,**kwargs):
  t_vals = np.linspace(0, 1, res)
  dc_scaled,phi_scaled,psi_scaled = scale_coef_to_radius(dc,phi,psi,disc_radius)
  theta_vals = tau * M * t_vals + np.array([fourier_series(0, phi_scaled, psi_scaled, t) for t in t_vals])
  return [[dc_scaled * cos(theta), dc_scaled * sin(theta)] for theta in theta_vals]

def angle_and_radial_modulation_curve(disc_radius,dc,a,phi,b,psi,M,res,**kwargs):
    t_vals = np.linspace(0, 1, res)
    dc_scaled,a_scaled,b_scaled = scale_coef_to_radius(dc,a,b,disc_radius)
    dummy,phi_scaled,psi_scaled = scale_coef_to_radius(dc,phi,psi,disc_radius)
    r_vals = np.array([fourier_series(dc_scaled,a_scaled,b_scaled,t) for t in t_vals])
    theta_vals = tau * M * t_vals + np.array([fourier_series(0, phi_scaled, psi_scaled, t) for t in t_vals])
    return [[r * cos(theta),r * sin(theta)] for r,theta in zip(r_vals,theta_vals)]


def angle_and_radial_modulation_curve_3d(curve_args1,curve_args2):
  xy_curve = angle_and_radial_modulation_curve(**curve_args1)
  z_curve = angle_and_radial_modulation_curve(**curve_args2)
  l = len(xy_curve)
  return [[xy_curve[i][0],xy_curve[i][1],z_curve[i][2]] for i in range(l)]


if __name__ == "__main__":
  from PlotContext import PlotContext
  from tkiter_widgets import FloatSlider,IntSlider, NumberListEntry, SelectBox, PresetCtrl
  from pathlib import Path
  #from DataFile import DataFile
  BASE_DIR = Path(__file__).resolve().parent

  pctx = PlotContext(-1, 1, "fourier curves", proj="2d")
  args = {"disc_radius":1.0,"dc":0.6,"a":[0.2,0.5,0.7,0.3],"b":[0.9,0.4,0.1,0.7],"phi":[0.4,0.7,0.2,0.7],"psi":[0.7,0.2,0.34,0.73],"M":1,"res":360}
  curve_func_key = "angle_and_radial_modulation"
  func_dict = {
      "radial_modulation":radial_modulation_curve,
      "angle_modulation":angle_modulation_curve,
      "angle_and_radial_modulation":angle_and_radial_modulation_curve
  }
  pts = func_dict[curve_func_key](**args)
  circ = [[args["disc_radius"]*cos(a),args["disc_radius"]*sin(a)] for a in np.linspace(0,tau,360)]

  def plot_pts():
   pctx.clear()
   pctx.plot_pointlist(pts,"black",0.3)
   pctx.plot_pointlist(circ,"purple",0.3)

  def arg_change(_id,val):
    global pts, curve_func_key, circ
    if _id == "curve_func": curve_func_key = val
    else:
      args[_id] = val
      if _id == "disc_radius":
        circ = [[args["disc_radius"]*cos(a),args["disc_radius"]*sin(a)] for a in np.linspace(0,tau,360)]
    pts = func_dict[curve_func_key](**args)
    plot_pts()

  def load_preset(data):
    print(data)
    global args,pts,circ
    args = data
    circ = [[args["disc_radius"] * cos(a), args["disc_radius"] * sin(a)] for a in np.linspace(0, tau, 360)]
    pts = func_dict[curve_func_key](**args)
    plot_pts()


  fn_sel = SelectBox(pctx,"curve_func","curve_func",list(func_dict.keys()),arg_change)
  radius_wid = FloatSlider(pctx,"disc_radius","disc_radius",0.01,5.0,1.0,arg_change)
  dc_wid = FloatSlider(pctx,"dc","dc",0.01,5.0,0.6,arg_change)
  M_wid = IntSlider(pctx,"M","M",1,10,1,arg_change)
  a_wid = NumberListEntry(pctx,"a","a",args["a"],"float",arg_change)
  b_wid = NumberListEntry(pctx,"a","b",args["b"],"float",arg_change)
  phi_wid = NumberListEntry(pctx,"phi","phi",args["phi"],"float",arg_change)
  psi_wid = NumberListEntry(pctx,"psi","psi",args["psi"],"float",arg_change)
  preset_ctrl = PresetCtrl(pctx,str(BASE_DIR)+"/data/fourier_curves.json",lambda: args,load_preset)
  pctx.run()






