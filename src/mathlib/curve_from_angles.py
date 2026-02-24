from math import cos, sin, pi, tau
import matplotlib.pyplot as plt

def rad2deg(a): return (180/pi)*a

def curve_amt(r,au):
  s=pi/2
  x_amt = (r * cos((3*s)+au))
  y_amt = (r * sin(au))
  return x_amt,y_amt

def step_cnt(a1,a2,au):
  return int(abs(a1-a2)/au)

def radius(a1, a2):
  diff = abs(a1 - a2)
  return diff / tau

def sector(a): return int(rad2deg(a))//90
def sector_sign(a,d):
  q = sector(a)
  xs = [-1,-1,1,1][q] if d == 1 else [1,1,-1,-1][q]
  ys = [1,-1,-1,1][q] if d == 1 else [-1,1,1,-1][q]
  return xs,ys

def curve_step(a_start, j, r, au, direction):
  # current angle along the segment
  theta = a_start + (direction * (au * j))
  # displacement along circle tangent (approximated)
  dx = r * (cos(theta + pi/2) - cos(theta))
  dy = r * (sin(theta + pi/2) - sin(theta))
  return dx, dy

def curve_from_angles(angs, sr):
    au = tau / sr
    ret = [(0.0, 0.0)]
    x, y = 0.0, 0.0

    for i in range(1, len(angs)):
        a1, a2 = angs[i-1], angs[i]
        r = radius(a1, a2)
        cnt = step_cnt(a1, a2, au)
        direction = 1 if a2 > a1 else -1
        for j in range(cnt):
            dx, dy = curve_step(a1, j, r, au, direction)
            x += dx
            y += dy
            ret.append((x, y))

    return ret

angles = [tau*v for v in [0.6,0.3,0.5,0.2,0.1,0.8,0.5,0.73]]
coords = curve_from_angles(angles,1600)
cx,cy = zip(*coords)

fig, ax = plt.subplots()
ax.plot(cx,cy)
ax.grid(ls=':')
plt.show()



"""    
def curve_step(start,end,amt,au,j):
  d = 1 if start < end else -1
  a = start + (au*j) if d == 1 else start - (au*j)
  xs,ys = sector_sign(a,d)
  x = amt[0] if xs == 1 else -amt[0]
  y = amt[1] if ys == 1 else -amt[1]
  return x,y
"""


"""
def curve_from_angles(angs,sr):
  au=tau/sr
  l = len(angs)
  ret = []
  for i in range(1,l):
    a1,a2 = angs[i-1],angs[i]
    r = radius(a1,a2)
    cnt = step_cnt(a1,a2,au)
    amt = curve_amt(r,au)
    coords = [curve_step(a1,a2,amt,au,j) for j in range(cnt+1)]
    ret.extend(coords)
  return ret
"""

