import numpy as np
from math import sin, cos
from S2 import S2_to_R3_v2

"""
This class is used to make 2d perspective renderings of geometrical data in 3 dimensions,
The cameras perspective position is controlled by a sphereical coordinate where 
r controls the distance between the target and the camera position
theta controls the angle in the xy-plane the camera has with respect to the target
phi controls the angle in the xz-plane the camera has with respet to the target.
"""

def normalize(v):
    v = np.array(v, float)
    n = np.linalg.norm(v)
    return v / n if n != 0 else v


class SphericalCamera:

    def __init__(self, r=3.0, theta=np.pi / 3, phi=np.pi / 4, target=(0, 0, 0)):
        self.r = r
        self.theta = theta
        self.phi = phi
        self.target = np.array(target, float)

    @property
    def position(self):
        return S2_to_R3_v2([self.r,self.theta,self.phi])

    @property
    def basis(self):
        cam = self.position
        target = self.target

        forward = normalize(target - cam)

        world_up = np.array([0, 0, 1], float)
        # If parallel, use different up vector
        #if abs(np.dot(forward, world_up)) > 0.99:
        #    world_up = np.array([0, 1, 0], float)

        right = normalize(np.cross(forward, world_up))
        up = np.cross(right, forward)

        return forward, right, up

    def world_to_camera(self, p):
        """Rotate + translate a world point into camera coordinates."""
        p = np.array(p, float)
        forward, right, up = self.basis
        cam = self.position
        q = p - cam
        return np.array([np.dot(q, right), np.dot(q, up), np.dot(q, forward)])

    def render_point(self,pt,f=1.0):
        cam_pt = self.world_to_camera(pt)
        proj_pt = [f * cam_pt[0]/cam_pt[-1],f*cam_pt[1]/cam_pt[-1]]
        return proj_pt

    def render_points(self,pt_list,f=1.0):
        return list(map(self.render_point,pt_list))


