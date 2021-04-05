import json
import os


train_list = []
for subdir, dirs, files in os.walk("./cleargrasp-dataset-pcd/train/opaque"):
    for filename in sorted(files):
        name, extension = os.path.splitext(filename)
        train_list.append(name)

test_list = []
for subdir, dirs, files in os.walk("./cleargrasp-dataset-pcd/test/opaque"):
    for filename in sorted(files):
        name, extension = os.path.splitext(filename)
        test_list.append(name)

cleargrasp = [{
    "taxonomy_id": "cleargrasp-real-val",
    "taxonomy_name": "ClearGrasp",
    "test": test_list,
    "train": train_list,
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
