import math
import numpy as np


class Quaternion:
    def __init__(self, a, b, c, d):
        self.a = a  # Real part
        self.b = b  # Coefficient for i
        self.c = c  # Coefficient for j
        self.d = d  # Coefficient for k

    # Add two quaternions
    def __add__(self, other):
        return Quaternion(self.a + other.a, self.b + other.b, self.c + other.c, self.d + other.d)

    # Subtract two quaternions
    def __sub__(self, other):
        return Quaternion(self.a - other.a, self.b - other.b, self.c - other.c, self.d - other.d)

    # Multiply two quaternions (Hamilton product)
    def __mul__(self, other):
        a = self.a * other.a - self.b * other.b - self.c * other.c - self.d * other.d
        b = self.a * other.b + self.b * other.a + self.c * other.d - self.d * other.c
        c = self.a * other.c - self.b * other.d + self.c * other.a + self.d * other.b
        d = self.a * other.d + self.b * other.c - self.c * other.b + self.d * other.a
        return Quaternion(a, b, c, d)

    @staticmethod
    def rotate(point, axis, angle):
        ax = np.array(axis) / np.linalg.norm(axis)
        qa = np.cos(angle/2)
        qb,qc,qd = ax * np.sin(angle/2)
        q = Quaternion(qa,qb,qc,qd)
        q_inv = q.conjugate()
        q_p = Quaternion(0,point[0],point[1],point[2])
        p_rot = (q*q_p) * q_inv
        return [p_rot.b,p_rot.c,p_rot.d]

        # Quaternion conjugate (negates the imaginary parts)

    def conjugate(self):
        return Quaternion(self.a, -self.b, -self.c, -self.d)

    # Quaternion norm (magnitude)
    def norm(self):
        return math.sqrt(self.a**2 + self.b**2 + self.c**2 + self.d**2)

    # String representation of the quaternion
    def __repr__(self):
        return f"{self.a} + {self.b}i + {self.c}j + {self.d}k"


def plane_to_complex_plane(point): return point[0] + point[1] * j