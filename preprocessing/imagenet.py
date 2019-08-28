import numpy as np
import cv2
import os

MAP_CLASS_LOC = "/media/4TB/datasets/ILSVRC2015/ILSVRC2015/devkit/data/map_clsloc.txt"

class_dir_map = {}

with open(MAP_CLASS_LOC, "rb") as map_class_file:
    rows = map_class_file.readlines()
    for row in rows:
        row = row.strip()
        arr = row.decode("utf-8").split(" ")
        class_dir_map[arr[0]] = arr[2]

TRAIN_DATA_FOLDER = "/media/4TB/datasets/ILSVRC2015/ILSVRC2015/Data/CLS-LOC/train/"

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
