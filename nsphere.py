import math

normalize = lambda x,maximum: x / (maximum * x)

def sphere_radius(r,t): return math.sqrt(pow(r,2) - pow(t,2))

def area_sphere(r):return 4*math.pi*pow(r,2)

def n_ang_cos(ang):
  return math.cos(math.pi * normalize(ang,math.pi))

def n_ang_sin(ang):
  return math.sin(math.tau * normalize(ang,math.tau))

def n_coord(arr,i):
    if i == 1:
        return arr[0] * math.cos(arr[1])
    elif i == len(arr):
        l = [arr[0]] + [math.sin(v) for v in arr[1:i]]
        print(l)
        return math.prod(l)
    else:
        if i > len(arr):
            raise ValueError("i not in range")
        else:
          sin_vals = [math.sin(v) for v in arr[1:i-1]]
          cos_val = [math.cos(arr[-1])]
          l = [arr[0]] + sin_vals + cos_val
          print(l)
          return math.prod(l)

def n_coord_normalized(arr,i):
  if i <= len(arr):
    is_sec_last = i == (len(arr) - 1)
    is_first = i == 1
    if is_first:
      parr = [arr[0]*n_ang_cos(arr[1])]
      return [math.prod(parr),len(parr)]
    elif is_sec_last:
      parr = [arr[0]] + [n_ang_sin(v) for v in arr[1:i-1]] + [n_ang_cos(arr[i])]
      return [math.prod(parr),len(parr)]
    else:
      parr = [arr[0]] + [n_ang_sin(v) for v in arr[1:i]]
      return [math.prod(parr),len(parr)]
  else:
      raise ValueError("i must be in range of arr")

def n_coords(arr):
  if len(arr) >= 3:
    dim = len(arr)
    r = arr[0]
    ret = {}
    for i in range(1,dim+1):
        ret[f"x{i}"] = n_coord(arr,i)
    return ret

def conj_unit(num_of_decimals):return round(math.tau,num_of_decimals) - round(math.tau,num_of_decimals - 1)

def derivative(x):
    h,f = 1e-5, lambda v: v**2 + 1
    return (f(x+h)-f(x-h))/(2*h)

def n_sphere_vol(n, r):
    return (pow(2,n) * pow(r,n) * pow((math.tau / 4),math.floor(n/2))) / math.factorial(math.factorial(n))

def n_sphere_area(n, r):
    Rn = pow(r,n-1) * r
    cp = Rn / 2
    sp = cp / 2
    return derivative(n_sphere_vol(n,Rn))





