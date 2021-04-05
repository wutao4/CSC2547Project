import numpy as np
import open3d as o3d
from open3d.open3d_pybind.utility import VerbosityLevel, set_verbosity_level, get_verbosity_level

from utils import IO


if __name__ == '__main__':
    # print(get_verbosity_level())
    # set_verbosity_level(VerbosityLevel.Error)  # try to disable printing "Format = .." in read_point_cloud
    # print(get_verbosity_level())

    dset = 'test'
    name = f'{41:09d}-3'

    opaque = o3d.io.read_point_cloud("./cleargrasp-dataset-pcd/%s/opaque/%s.pcd" % (dset, name))
    # print(opaque)
    print(np.array(opaque.points).shape)
    o3d.visualization.draw_geometries([opaque])

    transp = o3d.io.read_point_cloud("./cleargrasp-dataset-pcd/%s/transparent/%s.pcd" % (dset, name))
    print(np.array(transp.points).shape)
    o3d.visualization.draw_geometries([transp])

    # ShapeNet dataset
    name2 = 'edba8e42a3996f7cb1a9ec000a076431'
    # complete = o3d.io.read_point_cloud("./shapenet-chair/complete/%s.pcd" % name2)
    # partial = o3d.io.read_point_cloud("./shapenet-chair/partial/%s/00.pcd" % name2)
    # print(np.array(complete.points).shape)
    # print(np.array(partial.points).shape)
    # o3d.visualization.draw_geometries([complete])
    # o3d.visualization.draw_geometries([partial])
