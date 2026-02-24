import numpy as np

# ----------------------------- #
#  Complex → R2
# ----------------------------- #
def C_R2(p):
    pl = np.asarray(p, dtype=complex)
    return np.stack([pl.real, pl.imag], axis=-1)


# ----------------------------- #
#  S2 ↔ R3  (standard spherical coords)
# ----------------------------- #
def S2_R3(p):
    pl = np.asarray(p, dtype=float)
    r, theta, phi = pl[..., 0], pl[..., 1], pl[..., 2]

    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    return np.stack([x, y, z], axis=-1)


def R3_S2(p):
    pl = np.asarray(p, dtype=float)
    x, y, z = pl[..., 0], pl[..., 1], pl[..., 2]
    r = np.sqrt(x * x + y * y + z * z)

    theta = np.arctan2(y, x)  # longitude
    phi = np.arccos(z / r)  # colatitude

    return np.stack([r, theta, phi], axis=-1)


# ----------------------------- #
#  C2 → R4
# ----------------------------- #
def C2_R4(p):
    pl = np.asarray(p, dtype=complex)
    z1 = pl[..., 0]
    z2 = pl[..., 1]
    return np.stack([z1.real, z1.imag, z2.real, z2.imag], axis=-1)