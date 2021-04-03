import numpy as np
import open3d as o3d
from open3d.open3d_pybind.utility import VerbosityLevel, set_verbosity_level, get_verbosity_level

from utils import IO


if __name__ == '__main__':
    name = f'{3:09d}'

    opaque = o3d.io.read_point_cloud("./cleargrasp-dataset-pcd/opaque/%s-opaque.pcd" % name)
    # print(opaque)
    # print(np.array(opaque.points).shape)
    o3d.visualization.draw_geometries([opaque])

    print(get_verbosity_level())
    set_verbosity_level(VerbosityLevel.Error)
    print(get_verbosity_level())

    transp = o3d.io.read_point_cloud("./cleargrasp-dataset-pcd/transparent/%s-transparent.pcd" % name)
    o3d.visualization.draw_geometries([transp])
