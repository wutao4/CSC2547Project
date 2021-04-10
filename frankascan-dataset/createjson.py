import json
import os


train_list = []
for subdir, dirs, files in os.walk("./frankascan/train"):
    for name in sorted(dirs):
        train_list.append(name)

test_list = []
for subdir, dirs, files in os.walk("./frankascan/test"):
    for name in sorted(dirs):
        test_list.append(name)

frankascan = [{
    "taxonomy_id": "frankascan",
    "taxonomy_name": "FrankaScan-Beaker",
    "test": test_list,
    "train": train_list,
    "val": []
}]

if __name__ == '__main__':
    with open('./FrankaScan.json', 'w') as outfile:
        json.dump(frankascan, outfile)
