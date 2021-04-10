import numpy as np
import open3d as o3d
from open3d.open3d_pybind.utility import VerbosityLevel, set_verbosity_level, get_verbosity_level

from utils import IO


if __name__ == '__main__':
    # print(get_verbosity_level())
    # set_verbosity_level(VerbosityLevel.Error)  # try to disable printing "Format = .." in read_point_cloud
    # print(get_verbosity_level())

    dset = 'train'
    name = '1617656782.0434363'
    obj_idx = 1

    opaque = o3d.io.read_point_cloud("./frankascan/%s/%s/depth2pcd_GT_%d.pcd" % (dset, name, obj_idx))
    # print(opaque)
    print(np.array(opaque.points).shape)
    o3d.visualization.draw_geometries([opaque])

    transp = o3d.io.read_point_cloud("./frankascan/%s/%s/depth2pcd_%d.pcd" % (dset, name, obj_idx))
    print(np.array(transp.points).shape)
    o3d.visualization.draw_geometries([transp])
