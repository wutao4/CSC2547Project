import numpy as np
import open3d as o3d
from open3d.open3d_pybind.utility import VerbosityLevel, set_verbosity_level, get_verbosity_level

from utils import IO


if __name__ == '__main__':
    # print(get_verbosity_level())
    # set_verbosity_level(VerbosityLevel.Error)  # try to disable printing "Format = .." in read_point_cloud
    # print(get_verbosity_level())

    name = f'{35:09d}'

    opaque = o3d.io.read_point_cloud("./cleargrasp-dataset-pcd/opaque/%s-opaque.pcd" % name)
    # print(opaque)
    print(np.array(opaque.points).shape)
    # o3d.visualization.draw_geometries([opaque])

    transp = o3d.io.read_point_cloud("./cleargrasp-dataset-pcd/transparent/%s-transparent.pcd" % name)
    print(np.array(transp.points).shape)
    # o3d.visualization.draw_geometries([transp])

    # ShapeNet dataset
    name2 = '6730fb4ca7c90ddccd8b2a7439d99cc3'
    complete = o3d.io.read_point_cloud("./shapenet-chair/complete/%s.pcd" % name2)
    partial = o3d.io.read_point_cloud("./shapenet-chair/partial/%s/00.pcd" % name2)
    print(np.array(complete.points).shape)
    print(np.array(partial.points).shape)
    # o3d.visualization.draw_geometries([complete])
    # o3d.visualization.draw_geometries([partial])
