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


# Paths to image data
TRAIN_LOAD_PATH = "./cleargrasp-dataset-test-val/real-val/d435/"
TEST_LOAD_PATH = "./cleargrasp-dataset-test-val/real-test/d415/"
# Paths to point clouds to be saved
TRAIN_SAVE_PATH = "./cleargrasp-dataset-pcd/train/"
TEST_SAVE_PATH = "./cleargrasp-dataset-pcd/test/"

# Camera intrinsics
with open(os.path.join(TRAIN_LOAD_PATH, "camera_intrinsics.yaml"), 'r') as stream1:
    intrin1 = yaml.safe_load(stream1)
K1 = np.array([[intrin1['fx'],             0, intrin1['cx']],
               [            0, intrin1['fy'], intrin1['cy']],
               [            0,             0,            1]])
INV_K1 = np.linalg.inv(K1)

with open(os.path.join(TEST_LOAD_PATH, "camera_intrinsics.yaml"), 'r') as stream2:
    intrin2 = yaml.safe_load(stream2)
K2 = np.array([[intrin2['fx'],             0, intrin2['cx']],
               [            0, intrin2['fy'], intrin2['cy']],
               [            0,             0,            1]])
INV_K2 = np.linalg.inv(K2)


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


def img2pcd(name, load_path, save_path, inv_k, normalize=False, plot=False):
    """Convert a pair of ClearGrasp images with opaque and transparent objects into point clouds.
    """
    mask = IO.get(os.path.join(load_path, "%s-mask.png" % name))

    # Separate multiple objects into multiple point clouds
    mask_copy = np.array(mask * 255, dtype=np.uint8)
    if plot:
        cv2.imshow("%s mask" % name, mask_copy)
        cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(mask_copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    obj_idx = 0
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 100:
            mask_i = np.zeros_like(mask)
            cv2.drawContours(mask_i, contours, contourIdx=i, color=255, thickness=-1)
            if plot:
                cv2.imshow("%s idx:%d" % (name, i), mask_i)
                cv2.waitKey(0)

            mask_pcd = deproject(mask_i / 255, inv_k).sum(axis=1) > 0

            opaque_depth = IO.get(os.path.join(load_path, "%s-opaque-depth-img.exr" % name))
            opaque_pcd = deproject(opaque_depth, inv_k)[mask_pcd]
            IO.put(os.path.join(save_path, "opaque/%s-%d.pcd" % (name, obj_idx)), opaque_pcd)

            transp_depth = IO.get(os.path.join(load_path, "%s-transparent-depth-img.exr" % name))
            transp_pcd = deproject(transp_depth, inv_k)[mask_pcd]
            IO.put(os.path.join(save_path, "transparent/%s-%d.pcd" % (name, obj_idx)), transp_pcd)

            if normalize:  # normalize pcd to be within [-1, 1] for gridding
                max_val = np.nanmax((np.max(np.abs(opaque_pcd)), np.max(np.abs(transp_pcd))))
                if max_val > 1:
                    print(max_val)
                opaque_pcd /= max_val * 1.01
                transp_pcd /= max_val * 1.01
                with open(os.path.join(save_path, "norm_factors/%s-%d.json" % (name, obj_idx)), 'w') as outfile:
                    json.dump(max_val * 1.01, outfile)

            obj_idx += 1

            # print(mask_pcd.sum())
            # print(opaque_pcd.shape)
            # print(transp_pcd.shape)

        else:
            if plot:
                mask_i = np.zeros_like(mask)
                cv2.drawContours(mask_i, contours, contourIdx=i, color=255, thickness=5)
                cv2.imshow("%s idx:%d" % (name, i), mask_i)
                cv2.waitKey(0)


if __name__ == '__main__':
    # Create train-set point clouds from ClearGrasp 'real-val' images
    n_train = 173
    for i in tqdm(range(n_train)):
        name = f'{i:09d}'
        img2pcd(name, TRAIN_LOAD_PATH, TRAIN_SAVE_PATH, INV_K1, normalize=True)

    # Create test-set point clouds from ClearGrasp 'real-test' images
    n_test = 90
    for i in tqdm(range(n_test)):
        name = f'{i:09d}'
        img2pcd(name, TEST_LOAD_PATH, TEST_SAVE_PATH, INV_K2, normalize=True)

    # img2pcd(f'{3:09d}', TRAIN_LOAD_PATH, TRAIN_SAVE_PATH, INV_K1, plot=True)
    # img2pcd(f'{89:09d}', TEST_LOAD_PATH, TEST_SAVE_PATH, INV_K2, plot=True)

    # opaq = pyexr.open(os.path.join(PATH, "000000000-opaque-depth-img.exr")).get("R").squeeze().astype(np.float32)
    # print(opaq.shape)
    # transp = IO.get(os.path.join(PATH, "000000000-transparent-depth-img.exr")).astype(np.float32)
    # print(transp.shape)
    # mask = IO.get(os.path.join(PATH, "000000003-mask.png"))
    # print(mask.sum() * 0.4)
