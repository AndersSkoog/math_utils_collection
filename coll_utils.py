from num_utils import  is_number
import numpy as np
import math


def adj(l): return [[l[i - 1], l[i]] for i in range(1, len(l))]

def adjacents(l): return [[l[i],l[i + 1]] for i in range(len(l) - 1)]

def is_two_tupple(x): return isinstance(x,tuple) and len(x) == 2

def is_two_tupple_numeric(x): return is_two_tupple(x) and is_number(x[0]) and is_number(x[1])

def is_two_tupple_product(x): return is_two_tupple_numeric(x) and abs(x[0]) == abs(x[1])

def find_min(arr, ind):
    """
    given a list of lists with numberic entries
    returns the element with the highest value at index ind
    :param arr: [[number]]
    :param ind: int
    :return: [number]
    """
    ind_vals = list(map(lambda p: p[ind], arr))
    ind = ind_vals.index(min(ind_vals))
    return arr[ind]

def find_max(arr, ind):
  """
  given a list of lists with numberic entries
  returns the element with the highest value at index ind
  :param arr: [[number]]
  :param ind: int
  :return: [number]
  """
  ind_vals = list(map(lambda p: p[ind], arr))
  ind = ind_vals.index(max(ind_vals))
  return arr[ind]

def find_with_pred(arr, pred):
  for el in arr:
    if pred(el):
      return el
  return None

def valid_index(ind,s): return isinstance(ind,int) and ind >= 0 or ind < s

def arr_eq(a1,a2):
    if np.shape(a1) == np.shape(a2) and np.array(a1 == a2).all(): return True
    else: return False

def det(m):
  dim = len(m)
  t1 = []
  t2 = []
  for it1 in range(dim):
    t1.append(math.prod([m[j1][(it1 + j1) % dim] for j1 in range(dim)]))
  for it2 in list(reversed(range(dim))):
    t2.append(math.prod([m[j2][(it2 - j2) % dim] for j2 in range(dim)]))
  return sum(t1) - sum(t2)


