from PIL import Image
import numpy as np
from sphere import SpherePoint
from math_utils import valid_index


class SphereImagePerspective:

  def __init__(self,side:int,image:Image):
    self.pix_org = np.array(image)
    self.s = np.shape(self.pix_org)[0]
    self.side = side
    self.c = int(self.s / 2)
    assert self.s % 2 == 0, "image size must be even"
    assert np.shape(self.pix_org)[1] == self.s, "image must be square"
    self.r = self.s / 2
    self.plane_x = np.linspace(-self.r, self.r, self.s)
    self.plane_y = np.linspace(-self.r, self.r, self.s)

  def get_shifted_pixels(self,pi:int,pj:int,pixels=None):
    assert valid_index(pi, self.s) and valid_index(pj, self.s), "pixel index not valid"
    if pixels: assert np.shape(pixels) == np.shape(self.pix_org), "passed pixeldata does not match original image"
    pix_org = pixels if pixels else np.copy(self.pix_org)
    pix_shifted = pix_org
    for v in range(self.s):
      for w in range(self.s):
        i, j = (self.c + v) % self.s, (self.c + w) % self.s
        a, b = (pi + v) % self.s, (pj + w) % self.s
        pix_shifted[i][j] = pix_org[a][b]
    return pix_shifted

  def get_shifted_plane(self,pi:int,pj:int):
    assert valid_index(pi,self.s) and valid_index(pj,self.s)
    di = self.c - pi
    dj = self.c - pj
    arr = np.array(self.plane_x,self.plane_y)
    return np.roll(arr, shift=(di, dj), axis=(0, 1))

  def get_sphere_point(self,pi:int,pj:int):
    assert valid_index(pi, self.s) and valid_index(pj, self.s), "pixel index not valid"
    px,py = self.plane_x[pi],self.plane_y[pj]
    ang = np.arctan2(py, px)  # can be the theta value for a spherical coordinate
    z = np.sqrt(pow(px, 2) + pow(py, 2))  # can be the polar angle for the spherical coordinate
    return SpherePoint(self.r, ang, z, center=[0, 0, self.r])

  def associate_pixel_index(self,pi:int,pj:int,pixels=None):
    assert valid_index(pi,self.s) and valid_index(pj,self.s)
    px,py = self.plane_x[pi],self.plane_y[pj]
    org_plane_point = [px,py]
    sphere = self.get_sphere_point(pi,pj)
    proj_plane_point = sphere.plane_coord()
    org_pixels = pixels if pixels else self.pix_org
    shifted_pixels = self.get_shifted_pixels(pi,pj,org_pixels)
    theta = np.arctan2(py,px)  # can be the theta value for a spherical coordinate
    phi = np.sqrt(pow(px, 2) + pow(py, 2))  # can be the polar angle for the spherical coordinate
    sphere_coord = [self.r,theta,phi]
    ret = dict(
      org_plane_point=org_plane_point,
      org_pixels=org_pixels,
      shifted_pixels=shifted_pixels,
      proj_plane_point=proj_plane_point,
      sphere=sphere,
      sphere_coord=sphere_coord
    )
    return ret




