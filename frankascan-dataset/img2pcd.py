import os
import numpy as np
from tqdm import tqdm
import yaml
import json
import cv2
import open3d
import pyexr
import matplotlib.pyplot as plt

from utils import IO


# Paths to depth-image data
TRAIN_PATH = "./frankascan/train"
TEST_PATH = "./frankascan/test"
SAVE_TRAIN_PATH = "./frankascan/train"
SAVE_TEST_PATH = "./frankascan/test"

# Camera intrinsics (read from camera)
K = np.array([[613.96246338,            0, 324.44714355],
              [           0, 613.75634766, 239.17121887],
              [           0,            0,            1]])
INV_K = np.linalg.inv(K)


def deproject(depth_image, inv_k):
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
    points_3d = depth_arr * inv_k.dot(pixels_homog)
    points_3d = points_3d.transpose()
    return points_3d


def img2pcd(name, load_path, save_path, inv_k, plot=False, normalize=None, skip=False):
    """Convert a pair of ClearGrasp images with opaque and transparent objects into point clouds.
    """
    mask = IO.get(os.path.join(load_path, "%s/instance_segment.png" % name))

    # Separate multiple objects into multiple point clouds
    mask_vals = np.unique(mask[:, :, 2])[1:]  # each object is indicated by a distinct value in RED channel
    maxdis = []
    skip_count = 0
    for i in range(len(mask_vals)):
        mask_i = np.array(mask[:, :, 2] == mask_vals[i], dtype=np.float32)
        if plot:
            cv2.imshow("%s idx:%d" % (name, i), mask_i)
            cv2.waitKey(0)

        mask_pcd = deproject(mask_i, inv_k).sum(axis=1) > 0

        opaque_depth = IO.get(os.path.join(load_path, "%s/detph_GroundTruth.exr" % name))
        opaque_pcd = deproject(opaque_depth, inv_k)[mask_pcd]


        transp_depth = IO.get(os.path.join(load_path, "%s/depth.exr" % name))
        transp_pcd = deproject(transp_depth, inv_k)[mask_pcd]

        maxdis.append(np.max((np.max(np.abs(opaque_pcd)), np.max(np.abs(transp_pcd)))))

        if normalize and not skip:
            opaque_pcd /= maxdis[-1] * 1.01
            transp_pcd /= maxdis[-1] * 1.01
        if maxdis[-1] >= 1 and skip:
            skip_count += 1
            continue
        if not os.path.exists(os.path.join(save_path, name)):
            os.mkdir(os.path.join(save_path, name))
        if maxdis[-1] == 0:
            print((name, i))
        # save maxdis[-1] * 1.01
        with open(os.path.join(save_path, "%s/scale_factor_%d.json" % (name, i)), 'w') as outfile:
            json.dump(maxdis[-1] * 1.01, outfile)

        IO.put(os.path.join(save_path, "%s/depth2pcd_GT_%d.pcd" % (name, i)), opaque_pcd)
        IO.put(os.path.join(save_path, "%s/depth2pcd_%d.pcd" % (name, i)), transp_pcd)
        # print(mask_pcd.sum())
        # print(opaque_pcd.shape)
        # print(transp_pcd.shape)
    return np.max(maxdis), skip_count


if __name__ == '__main__':
    # Create point clouds from depth images
    maxdis = []
    skip_count = 0
    normalize = 65.5625 * 1.01
    skip = False
    print("Converting point clouds for training set...")
    for subdir, dirs, files in os.walk(TRAIN_PATH):
        for dirname in tqdm(dirs):
            max, skip = img2pcd(dirname, load_path=TRAIN_PATH, save_path=SAVE_TRAIN_PATH, inv_k=INV_K, normalize=normalize, skip=skip)
            maxdis.append(max)
            skip_count += skip
    print("Converting point clouds for test set...")
    for subdir, dirs, files in os.walk(TEST_PATH):
        for dirname in tqdm(dirs):
            max, skip = img2pcd(dirname, load_path=TEST_PATH, save_path=SAVE_TEST_PATH, inv_k=INV_K, normalize=normalize, skip=skip)
            maxdis.append(max)
            skip_count += skip

    print(np.max(maxdis))
    if skip:
        print(skip_count)

    # img2pcd('1617656782.0434363', TRAIN_PATH, TRAIN_PATH, INV_K, plot=True)
