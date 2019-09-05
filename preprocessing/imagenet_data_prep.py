import os

MAP_CLASS_LOC = "/media/4TB/datasets/ILSVRC2015/ILSVRC2015/devkit/data/map_clsloc.txt"

class_dir_map = {}
id_class_map = {}

with open(MAP_CLASS_LOC, "rb") as map_class_file:
    rows = map_class_file.readlines()
    for row in rows:
        row = row.strip()
        arr = row.decode("utf-8").split(" ")
        class_dir_map[arr[0]] = arr[2]
        id_class_map[int(arr[1])] = arr[2]

TRAIN_DATA_FOLDER = "/media/4TB/datasets/ILSVRC2015/ILSVRC2015/Data/CLS-LOC_100/train/"

# ----------------- Rename Train Dir --------------------

# update crane as crane_bird and maillot as maillot_1 and maillot_2

for key in class_dir_map.keys():
    if os.path.isdir(TRAIN_DATA_FOLDER + key):
        os.rename(TRAIN_DATA_FOLDER + key, TRAIN_DATA_FOLDER + class_dir_map[key])

# ----------------- Test/Train Split --------------------
import glob

files = glob.glob(TRAIN_DATA_FOLDER + "**/*.JPEG")
paths = []
labels = []

for file in files:
    label_str = file.split("/")[-2]
    paths.append(file)
    labels.append(label_str)

from sklearn.model_selection import train_test_split

(trainPaths, testPaths, trainLabels, testLabels) = train_test_split(paths, labels, test_size=50000, stratify=labels, random_state=42)

TEST_DATA_FOLDER = "/media/4TB/datasets/ILSVRC2015/ILSVRC2015/Data/CLS-LOC/test/"

for testPath, testLabel in zip(testPaths, testLabels):

    if not os.path.isdir(TEST_DATA_FOLDER + testLabel):
        os.mkdir(TEST_DATA_FOLDER + testLabel)

    os.rename(testPath, TEST_DATA_FOLDER + testLabel + "/" + testPath.split("/")[-1])

# ----------------- Create val folders --------------------

BLACK_LIST = "/media/4TB/datasets/ILSVRC2015/ILSVRC2015/devkit/data/ILSVRC2015_clsloc_validation_blacklist.txt"
VAL_CLASS_PATH = "/media/4TB/datasets/ILSVRC2015/ILSVRC2015/devkit/data/ILSVRC2015_clsloc_validation_ground_truth.txt"

VAL_DATA_PATH = "/media/4TB/datasets/ILSVRC2015/ILSVRC2015/Data/CLS-LOC/val/"

VAL_ORI_DATA_PATH = "/media/4TB/datasets/ILSVRC2015/ILSVRC2015/Data/CLS-LOC/val_original/*.JPEG"

black_list = []

with open(BLACK_LIST) as b_file:
    rows = b_file.readlines()
    for row in rows:
        row = int(row.strip())
        black_list.append(row)

val_class = []

with open(VAL_CLASS_PATH) as val_file:
    rows = val_file.readlines()
    for row in rows:
        row = int(row.strip())
        val_class.append(row)

val_files = glob.glob(VAL_ORI_DATA_PATH)

for file in val_files:
    seq_num = int(file.split("/")[-1].split("_")[-1].split(".")[0])
    if seq_num not in black_list:
        class_id = val_class[seq_num - 1]
        class_name = id_class_map[class_id]

        if not os.path.isdir(VAL_DATA_PATH + class_name):
            os.mkdir(VAL_DATA_PATH + class_name)

        os.rename(file, VAL_DATA_PATH + class_name + "/" + file.split("/")[-1])

# --------------------------- Create Label Map -------------------------------------

import json
import glob

label_map = {}

dirs = glob.glob(TRAIN_DATA_FOLDER + "*")
for i, dir in enumerate(dirs):
    label_map[dir.split("/")[-1]] = i

with open("label_map_100.json", "w") as file:
    file.write(json.dumps(label_map))
