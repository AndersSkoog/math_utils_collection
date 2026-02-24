from math import tau, pi, cos, sin
import numpy as np
from math import tau,pi,sin,cos
#from num import decimal_value, decimal_count, is_odd

sr     = 500
ts     = np.linspace(0,1,sr)
ts_ang = np.linspace(0,tau,sr)

"""
def omega(freq,i): return ts_ang[i]
"""

def num_derivative(f,x,d=1e-5):
    a,b = abs(f(x+d)), abs(f(x-d))
    dy = max(a,b) - min(a,b)
    dx = 2*d
    return dy/dx


def multirate_rotation(N:int,amp,freq):
  phase_step = tau/N
  phase_step_2 = phase_step / freq
  omega1 = [phase_step * i for i in range(N)]
  omega2 = [((phase_step * i) + phase_step_2) % tau for i in range(N)]
  cyc1 = [[cos(v),sin(v)] for v in omega1]
  cyc2 = [[amp*cos(v),amp*sin(v)] for v in omega2]
  return cyc1,cyc2


"""
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
Create figure and line object
fig, ax = plt.subplots()
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)
ax.set_aspect('equal')
line1, = ax.plot([], [], lw=1,color="black")
line2, = ax.plot([], [], lw=1,color="red")


Optional: show a circle for reference
circle = plt.Circle((0,0), 1.0, color='gray', fill=False)
ax.add_patch(circle)

Initialization function
def init():
    line1.set_data([], [])
    return line1,

Animation function
def animate(i):
    x = [0, np.cos(wave[i])]
    y = [0, np.sin(wave[i])]
    line.set_data(x, y)
    return line,

anim = FuncAnimation(fig, animate, init_func=init, frames=500, interval=20, blit=True)
plt.show()
"""
