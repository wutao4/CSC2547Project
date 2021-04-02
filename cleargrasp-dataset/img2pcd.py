import os
import numpy as np
import yaml
import open3d
import pyexr
import matplotlib.pyplot as plt

from utils import IO


# path to image data
PATH = "./cleargrasp-dataset-test-val/real-val/d435/"
# path to point clouds to be saved
TRANSP_PATH = "./cleargrasp-dataset-pcd/transparent"
OPAQUE_PATH = "./cleargrasp-dataset-pcd/opaque"

# camera intrinsics
with open(os.path.join(PATH, "camera_intrinsics.yaml"), 'r') as stream:
    intrinsics = yaml.safe_load(stream)
K = np.array([[intrinsics['fx'],                0, intrinsics['cx']],
              [               0, intrinsics['fy'], intrinsics['cy']],
              [               0,                0,               1]])
INV_K = np.linalg.inv(K)


def deproject(depth_image):
    """Deprojects a DepthImage into a PointCloud.
    Reference: Berkeley AutoLab Core
    https://github.com/BerkeleyAutomation/perception/blob/e1c936f38a0aef97348c2d8de364807b5238e1d0/perception/camera_intrinsics.py#L335
    """
    # create homogeneous pixels
    row_indices = np.arange(depth_image.shape[0])
    col_indices = np.arange(depth_image.shape[1])
    pixel_grid = np.meshgrid(col_indices, row_indices)
    pixels = np.c_[pixel_grid[0].flatten(), pixel_grid[1].flatten()].T
    pixels_homog = np.r_[pixels, np.ones([1, pixels.shape[1]])]
    depth_arr = np.tile(depth_image.flatten(), [3, 1])

    # deproject
    points_3d = depth_arr * INV_K.dot(pixels_homog)
    points_3d = points_3d.transpose()
    return points_3d


def img2pcd(name):
    """Convert a pair of ClearGrasp images with opaque and transparent objects into point clouds.
    """
    mask = IO.get(os.path.join(PATH, "%s-mask.png" % name))

    opaque_depth = IO.get(os.path.join(PATH, "%s-opaque-depth-img.exr" % name))
    opaque_pcd = deproject(opaque_depth * mask)
    IO.put(os.path.join(OPAQUE_PATH, "%s-opaque.pcd" % name), opaque_pcd)

    transp_depth = IO.get(os.path.join(PATH, "%s-transparent-depth-img.exr" % name))
    transp_pcd = deproject(transp_depth * mask)
    IO.put(os.path.join(TRANSP_PATH, "%s-transparent.pcd" % name), transp_pcd)


if __name__ == '__main__':
    n_img = 173
    for i in range(n_img):
        name = f'{i:09d}'
        img2pcd(name)

    # opaq = pyexr.open(os.path.join(PATH, "000000000-opaque-depth-img.exr")).get("R").squeeze().astype(np.float32)
    # print(opaq.shape)
    # transp = IO.get(os.path.join(PATH, "000000000-transparent-depth-img.exr")).astype(np.float32)
    # print(transp.shape)
    # mask = IO.get(os.path.join(PATH, "000000000-mask.png"))
    # print(mask.shape)
