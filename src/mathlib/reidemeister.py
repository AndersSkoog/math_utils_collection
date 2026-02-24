from vector_utils import perpendicular_vector, normalize_vector
import numpy as np

def reidemeister_I(segment, r=0.2):
  """
  Twist / Loop: Add a local loop to a single segment.
  segment: np.array of shape (2,3) -> [P1, P2]
  r: radius of loop
  Returns: 4-point Bézier control points
  """
  P1, P2 = segment
  V = np.array(P2) - np.array(P1)
  N = perpendicular_vector(V)
  C1 = P1 + V / 3 + N * r
  C2 = P1 + 2 * V / 3 - N * r
  return np.array([P1, C1, C2, P2])


def reidemeister_II(seg1, seg2, r=0.2):
  """
  Two-strand move: create an over/under crossing between two adjacent segments.
  seg1, seg2: np.array of shape (2,3)
  r: offset magnitude
  Returns: two arrays of 3-point Bézier control points each
  """
  P1, P2 = seg1
  Q1, Q2 = seg2
  # Compute midpoints
  M1 = (P1 + P2) / 2
  M2 = (Q1 + Q2) / 2
  # Offset directions
  N = normalize_vector(M2 - M1)
  # Create auxiliary control points for over/under
  C1 = P1 + (P2 - P1) / 2 + N * r
  C2 = Q1 + (Q2 - Q1) / 2 - N * r
  return np.array([P1, C1, P2]), np.array([Q1, C2, Q2])


def reidemeister_III(segA, segB, segC, r=0.2):
  """
  Sliding / triangle move: deform three segments that cross each other.
  segA, segB, segC: np.array of shape (2,3)
  r: offset magnitude for control points
  Returns: three arrays of 3-point Bézier control points each
  """
  # Midpoints
  M_A = (segA[0] + segA[1]) / 2
  M_B = (segB[0] + segB[1]) / 2
  M_C = (segC[0] + segC[1]) / 2

  # Triangle center
  center = (M_A + M_B + M_C) / 3

  def offset_segment(P1, P2):
    M = (P1 + P2) / 2
    N = normalize_vector(M - center)
    C = M + N * r
    return np.array([P1, C, P2])

  return offset_segment(*segA), offset_segment(*segB), offset_segment(*segC)
