import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from utils import IO
from img2pcd import deproject, INV_K


if __name__ == '__main__':
    dset = 'test'
    name = '1617662669.3255167'
    obj_idx = 1

    opaque = o3d.io.read_point_cloud("./frankascan/%s/%s/depth2pcd_GT_%d.pcd" % (dset, name, obj_idx))
    print(np.array(opaque.points).shape)
    o3d.visualization.draw_geometries([opaque])

    transp = o3d.io.read_point_cloud("./frankascan/%s/%s/depth2pcd_%d.pcd" % (dset, name, obj_idx))
    print(np.array(transp.points).shape)
    o3d.visualization.draw_geometries([transp])

    pred_i = o3d.io.read_point_cloud("./frankascan/%s/%s/depth2pcd_pred_%d.pcd" % (dset, name, obj_idx))
    print(np.array(pred_i.points).shape)
    o3d.visualization.draw_geometries([pred_i])

    depth_raw = IO.get("./frankascan/%s/%s/depth.exr" % (dset, name))
    plt.imshow(depth_raw)
    plt.show()
    pt_raw = deproject(depth_raw, INV_K)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt_raw)
    print(np.array(pcd.points).shape)
    o3d.visualization.draw_geometries([pcd])

    # depth_gt = IO.get("./frankascan/%s/%s/depth_GroundTruth.exr" % (dset, name))
    # plt.imshow(depth_gt)
    # plt.show()
    # pt_gt = deproject(depth_gt, INV_K)
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pt_gt)
    # print(np.array(pcd.points).shape)
    # o3d.visualization.draw_geometries([pcd])

    depth_pred = IO.get("./frankascan/%s/%s/depth_pred.exr" % (dset, name))
    plt.imshow(depth_pred)
    plt.show()
    pt_pred = deproject(depth_pred, INV_K)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pt_pred)
    print(np.array(pcd.points).shape)
    o3d.visualization.draw_geometries([pcd])
