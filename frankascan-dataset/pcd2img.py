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
from img2pcd import K



TEST_PATH = "./frankascan/test"
SAVE_TEST_PATH = "./frankascan/test"
FACTOR_PATH = "./frankascan/test"


def project_to_image(k, point_cloud, round_px=True):
    width = 640
    height = 480
    points_proj = k.dot(point_cloud.transpose())
    if len(points_proj.shape) == 1:
        points_proj = points_proj[:, np.newaxis]
    point_depths = points_proj[2, :]
    point_z = np.tile(point_depths, [3, 1])
    points_proj = np.divide(points_proj, point_z)
    if round_px:
        points_proj = np.round(points_proj)
    points_proj = points_proj[:2, :].astype(np.int16)

    valid_ind = np.where((points_proj[0, :] >= 0) & \
                         (points_proj[1, :] >= 0) & \
                         (points_proj[0, :] < width) & \
                         (points_proj[1, :] < height))[0]

    depth_data = np.zeros([height, width])
    points_proj = points_proj[:, valid_ind]
    point_depths = point_depths[valid_ind]
    for i in range(point_depths.shape[0]):
        current_depth = depth_data[points_proj[1,i], points_proj[0,i]]
        point_depth = point_depths[i]
        if current_depth == 0 or point_depth < current_depth:
            depth_data[points_proj[1,i], points_proj[0,i]] = point_depth
    return depth_data


def pcd2img(name, load_path, save_path, k, factor_path=None):
    """Convert point clouds to depth images.
    """
    mask = IO.get(os.path.join(load_path, "%s/instance_segment.png" % name))

    mask_bg = np.array(mask[:, :, 2] == 0, dtype=np.float32)
    depth_in = IO.get(os.path.join(load_path, "%s/depth.exr" % name))
    depth_out = depth_in * mask_bg

    mask_vals = np.unique(mask[:, :, 2])[1:]
    for i in range(len(mask_vals)):
        pcd_i = IO.get(os.path.join(load_path, "%s/depth2pcd_pred_%d.pcd" % (name, i)))
        if factor_path is not None:
            with open(os.path.join(factor_path, "%s/scale_factor_%d.json" % (name, i))) as f:
                factors = json.loads(f.read())
                if factors['normalize']:
                    pcd_i *= float(factors['normalize_factor'])
                if factors['centering']:
                    pcd_i += np.array(factors['center_position'])
        depth_i = project_to_image(k, pcd_i)
        mask_i = np.array(mask[:, :, 2] == mask_vals[i], dtype=np.float32)
        depth_out += depth_i * mask_i

    # write predicted depth image in exr
    depth_out = np.expand_dims(depth_out, -1).repeat(3, axis=2)
    pyexr.write(os.path.join(save_path, "%s/depth_pred.exr" % name), depth_out)


if __name__ == '__main__':
    print("Converting point clouds to depth images...")
    for subdir, dirs, files in os.walk(TEST_PATH):
        for dirname in tqdm(sorted(dirs)):
            pcd2img(dirname, load_path=TEST_PATH, save_path=SAVE_TEST_PATH, k=K, factor_path=FACTOR_PATH)

    # pcd2img('1617662669.3255167', TEST_PATH, SAVE_TEST_PATH, K, factor_path=FACTOR_PATH)
