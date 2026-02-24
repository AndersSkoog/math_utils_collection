import numpy as np
from vector_utils import normalize_vector, axis_pairs

def Rz(yaw):
  c, s = np.cos(yaw), np.sin(yaw)
  return np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]],dtype=float)

def Ry(pitch):
  c, s = np.cos(pitch), np.sin(pitch)
  return np.array([[c, 0, s],[0, 1, 0],[-s, 0, c]],dtype=float)

def Rx(roll):
  c, s = np.cos(roll), np.sin(roll)
  return np.array([[1, 0, 0],[0, c, -s],[0, s, c]],dtype=float)

def Rn(n, i, j, angle):
    R = np.identity(n)
    c, s = np.cos(angle), np.sin(angle)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R

def SO_3(yaw,pitch,roll):return Rz(yaw) @ Ry(pitch) @ Rx(roll)
def SO_3_UP(so3):  return normalize_vector(so3 @ np.array([0.0,0,1.0],dtype=float))
def SO_3_RIGHT(so3):return normalize_vector(so3 @ np.array([0.0,1.0,0.0],dtype=float))
def SO_3_FWD(so3):return normalize_vector(so3 @ np.array([1.0,0.0,0.0],dtype=float))
def SO_N(n,angles):
  assert len(angles) == (n * (n - 1)) // 2, "Incorrect number of angles"
  R = np.identity(n)
  idx_pairs = axis_pairs(n)  # All axis index pairs
  for angle, (i, j) in zip(angles, idx_pairs):
    R = Rn(n, i, j, angle) @ R  # Apply in order
  return R

def orient_basis_3(yaw,pitch,roll):
  so3 = SO_3(yaw,pitch,roll)
  return {"up":SO_3_UP(so3),"right":SO_3_RIGHT(so3),"fwd":SO_3_FWD(so3)}

def rot_matrix_around_axis(axis: np.ndarray, angle: float):
    axis = normalize_vector(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    C = 1 - c
    R = np.array([
        [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
        [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
        [z * x * C - y * s, z * y * C + x * s, c + z * z * C]
    ])
    return R
