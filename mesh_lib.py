import numpy as np

def sphere_mesh(radius=1.0, lat_seg=32, lon_seg=32):
    vertices = []
    normals = []
    indices = []

    for i in range(lat_seg + 1):
        theta = np.pi * i / lat_seg
        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        for j in range(lon_seg + 1):
            phi = 2 * np.pi * j / lon_seg
            sin_phi = np.sin(phi)
            cos_phi = np.cos(phi)

            x = radius * sin_theta * cos_phi
            y = radius * sin_theta * sin_phi
            z = radius * cos_theta

            vertices.append([x, y, z])
            normals.append([x/radius, y/radius, z/radius])

    # indices for triangles
    for i in range(lat_seg):
        for j in range(lon_seg):
            first = i * (lon_seg + 1) + j
            second = first + lon_seg + 1
            indices += [first, second, first + 1, second, second + 1, first + 1]

    return np.array(vertices, dtype=np.float32), np.array(normals, dtype=np.float32), np.array(indices, dtype=np.uint32)
