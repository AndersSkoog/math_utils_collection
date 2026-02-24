import numpy as np
from itertools import combinations


def normal_basis_vector(unit_val, ax:int, dim:int):
    """Return unit vector along axis ax, but scaled by unit_val."""
    v = np.zeros(dim)
    v[ax] = unit_val
    return v

def rot_plane(dim, i, j, angle):
    """Rotation matrix in the (i,j) plane."""
    R = np.eye(dim)
    c, s = np.cos(angle), np.sin(angle)
    R[i, i] = c
    R[j, j] = c
    R[i, j] = -s
    R[j, i] = s
    return R

def rot_mtx(dim, angles):
    """
    General SO(n) rotation: one angle per plane (i,j), following
    lexicographic order: (0,1), (0,2), (0,3), ..., (n-2,n-1).
    """
    angles_expected = dim * (dim - 1) // 2
    assert len(angles) == angles_expected, f"Expected {angles_expected} angles, got {len(angles)}"
    # identity matrix
    R = np.eye(dim)
    axis_pairs = list(combinations(range(dim), 2))  # [(0,1),(0,2),(1,2)...]

    for angle, (i, j) in zip(angles, axis_pairs):
        R = rot_plane(dim, i, j, angle) @ R

    return R


class ConformalVectorSpace:
    """An n-dim vector space with a rotatable orthonormal basis."""

    def __init__(self, dim, basis=None):
        self.dim = dim
        self.axis_pairs = list(combinations(range(dim),2))  # [(0,1),(0,2),(1,2)...]
        # Start with canonical basis e0...en-1
        self.basis = np.eye(dim) if basis is None else basis

    def orient(self, angles):
        new_basis = rot_mtx(self.dim,angles) @ self.basis
        return ConformalVectorSpace(self.dim,basis=new_basis)

    def vector(self, scalars):
        """Create a vector expressed in the current basis."""
        scalars = np.array(scalars)
        return self.basis @ scalars

    def __repr__(self):
        return f"ConformalVectorSpace(dim={self.dim}, basis=\n{self.basis})"
