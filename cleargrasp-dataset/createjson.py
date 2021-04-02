import json


n_data = 173

cleargrasp = [{
    "taxonomy_id": "cleargrasp-real-val",
    "taxonomy_name": "ClearGrasp",
    "test": [f"{i:09d}" for i in range(n_data)],
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
