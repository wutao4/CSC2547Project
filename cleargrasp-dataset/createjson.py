import json
import os


name_list = []
for subdir, dirs, files in os.walk("./cleargrasp-dataset-pcd/opaque"):
    for filename in sorted(files):
        name, extension = os.path.splitext(filename)
        name_list.append(name)

cleargrasp = [{
    "taxonomy_id": "cleargrasp-real-val",
    "taxonomy_name": "ClearGrasp",
    "test": name_list,
    "train": [],
    "val": []
}]

if __name__ == '__main__':
    with open('ClearGrasp.json', 'w') as outfile:
        json.dump(cleargrasp, outfile)

    # with open('ClearGrasp.json') as f:
    #     print(json.loads(f.read()))
    # with open('KITTI.json') as f:
    #     print(json.loads(f.read()))
    #     # print(repr(f.read()))
