from PIL import Image
import numpy as np
from sphere import SpherePoint
from math_utils import valid_index


class SphereImagePerspective:

  def __init__(self,side:int,image:Image):
    self.pix_org = np.array(image)
    self.s = np.shape(self.pix_org)[0]
    self.side = side
    print(self.s)
    self.c = int(self.s / 2)
    assert self.s % 2 == 0, "image size must be even"
    assert np.shape(self.pix_org)[1] == self.s, "image must be square"
    self.r = self.side / 2
    self.plane_x = np.linspace(-self.r, self.r, self.s)
    self.plane_y = np.linspace(-self.r, self.r, self.s)

  def get_shifted_pixels(self,pi:int,pj:int,pixels=None):
    assert valid_index(pi, self.s) and valid_index(pj, self.s), "pixel index not valid"
    #if np.shape(pixels)[ == np.shape(self.pix_org), "passed pixeldata does not match original image"
    #if pixels is not None:assert np.shape(pixels) ==  np.shape(self.pix_org), "dont match original image"
    #pix_org = pixels if pixels is not None else np.copy(self.pix_org)
    #pix_shifted = pix_org
    di,dj = self.c - pi, self.c - pj
    pix_org = np.copy(self.pix_org)
    pix_shifted = np.copy(self.pix_org)
    pix_shifted = np.roll(pix_shifted,shift=(di,dj),axis=(0,1))
    #pix_shifted = np.roll(pix_shifted,shift=(di,dj),axis=(0,1))

    #pix_shifted = np.roll(pix_shifted,shift=(-pi,-pj),axis=(1,1))

    #pix_shifted = np.roll(pix_shifted,shift=(pi,pj),axis=(0,1))

    #pix_shifted = np.roll(pix_shifted,shift=(-pi,-pj),axis=(1,0))

    #pix_shifted = np.roll(pix_shifted,shift=(0,pj),axis=(1,1))

    #pix_shifted = np.roll(pix_shifted,shift=(0,dj), axis=(0, 1))

    #pix_shifted = np.roll(pix_shifted,shift=(dj,dj), axis=(0, 1))
    #np.roll(pix_shifted, shift=(di, dj), axis=(1, 1))


    #return pix_shifted
    """"
    for v in range(self.s):
      for w in range(self.s):
        i, j = (self.c + v) % self.s, (self.c + w) % self.s
        a, b = (pi + v) % self.s, (pj + w) % self.s
        pix_shifted[i][j] = pix_org[a][b]
    """
    return pix_shifted


  def get_shifted_plane(self,pi:int,pj:int):
    assert valid_index(pi,self.s) and valid_index(pj,self.s)
    di = self.c - pi
    dj = self.c - pj
    arr = np.array(np.copy(self.plane_x),np.copy(self.plane_y))
    return np.roll(arr, shift=(di, dj), axis=(0, 1))

  def get_sphere_point(self,pi:int,pj:int):
    assert valid_index(pi, self.s) and valid_index(pj, self.s), "pixel index not valid"
    px,py = self.plane_x[pi],self.plane_y[pj]
    print(px,py)
    theta = np.arctan2(py, px)  # can be the theta value for a spherical coordinate
    phi = np.sqrt(pow(px, 2) + pow(py, 2))  # can be the polar angle for the spherical coordinate
    print(theta,phi)
    return SpherePoint(self.r, theta, phi, center=[0, 0, self.r])

  def associate_pixel_index(self,pi:int,pj:int,pixels=None):
    assert valid_index(pi,self.s) and valid_index(pj,self.s)
    px,py = self.plane_x[pi],self.plane_y[pj]
    theta = np.arctan2(py,px)  # can be the theta value for a spherical coordinate
    phi = np.sqrt(pow(px, 2) + pow(py, 2))  # can be the polar angle for the spherical coordinate
    org_plane_point = [px,py]
    sphere = SpherePoint(self.r,theta,phi,[0,0,self.r])
    proj_plane_point = sphere.plane_coord()
    org_pixels = pixels if pixels else np.copy(self.pix_org)
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

img = Image.open("images/paint.jpg")
persp = SphereImagePerspective(4,img)
persp1 = persp.associate_pixel_index(50,70)
persp1_img = Image.fromarray(persp1["shifted_pixels"])
persp1_img.save("images/shifted_paint.jpg")

