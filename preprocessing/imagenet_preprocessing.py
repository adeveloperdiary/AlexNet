import cv2
import glob
import tensorflow as tf
import progressbar
import multiprocessing
from itertools import repeat
import json
import random
import argparse
import os


def scale_image(image, size):
    image_height, image_width = image.shape[:2]

    if image_height <= image_width:
        ratio = image_width / image_height
        h = size
        w = int(ratio * 256)

        image = cv2.resize(image, (w, h))

    else:
        ratio = image_height / image_width
        w = size
        h = int(ratio * 256)

        image = cv2.resize(image, (w, h))

    return image


def center_crop(image, size):
    image_height, image_width = image.shape[:2]

    if image_height <= image_width and abs(image_width - size) > 1:

        dx = int((image_width - size) / 2)
        image = image[:, dx:-dx, :]
    elif abs(image_height - size) > 1:
        dy = int((image_height - size) / 2)
        image = image[dy:-dy, :, :]

    image_height, image_width = image.shape[:2]
    if image_height is not size or image_width is not size:
        image = cv2.resize(image, (size, size))

    return image


def process_image(image, size):
    image = scale_image(image, size)
    image = center_crop(image, size)

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def worker_tf_write(files, tf_record_path, label_map, size, image_quality, tf_record_options, number):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), image_quality]
    tf_record_options = tf.io.TFRecordOptions(compression_type=tf_record_options)

    widgets = [
        'Processing Images - [', str(number), ']',
        progressbar.Bar('#', '[', ']'),
        ' [', progressbar.Percentage(), '] ',
        '[', progressbar.Counter(format='%(value)02d/%(max_value)d'), '] '

    ]

    bar = progressbar.ProgressBar(maxval=len(files), widgets=widgets)
    bar.start()

    with tf.io.TFRecordWriter(tf_record_path, tf_record_options) as tf_writer:
        for i, file in enumerate(files):
            image = process_image(cv2.imread(file), size)
            is_success, im_buf_arr = cv2.imencode(".jpg", image, encode_param)

            if is_success:
                label_str = file.split("/")[-2]

                label_number = label_map[label_str]

                image_raw = im_buf_arr.tobytes()
                row = tf.train.Example(features=tf.train.Features(feature={
                    'label': _int64_feature(label_number),
                    'image_raw': _bytes_feature(image_raw)
                }))

                tf_writer.write(row.SerializeToString())
                bar.update(i + 1)
            else:
                print("Error processing " + file)
        bar.finish()


def master_tf_write(split_file_list, tf_record_paths, size, image_quality, label_map, tf_record_options):
    cpu_core = int(multiprocessing.cpu_count() / 4)

    p = multiprocessing.Pool(cpu_core)
    results = p.starmap(worker_tf_write,
                        zip(split_file_list, tf_record_paths, repeat(label_map), repeat(size), repeat(image_quality), repeat(tf_record_options),
                            list(range(len(tf_record_paths)))))
    p.close()
    p.join()


def create_tf_record(image_folder, record_path, identifier, label_map, size=256, split_number=1000, image_quality=90, tf_record_options=None):
    print("creating " + identifier + " records")

    files = glob.glob(image_folder)

    random.shuffle(files)

    split_file_list = [files[x:x + split_number] for x in range(0, len(files), split_number)]

    tf_record_paths = []

    for i in range(len(split_file_list)):
        tf_record_paths.append(record_path + identifier + "-" + str(i) + ".tfrecord")

    master_tf_write(split_file_list, tf_record_paths, size, image_quality, label_map, tf_record_options)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True, help="path to the config JSON.", )
    args = vars(ap.parse_args())

    with open(args["config"], "rb") as file:
        config = json.loads(file.read())

    with open(config["label_map"], "rb") as file:
        label_map = json.loads(file.read())

    for indentifier in config["batch"]:
        image_folder = config["image_folder"] + indentifier + "/**/*." + config["image_type"]
        record_path = config["record_path"] + indentifier + "/"

        if not os.path.isdir(record_path):
            os.makedirs(record_path)

        create_tf_record(image_folder, record_path, indentifier, label_map, config["crop_size"], config["split_number"], config["image_quality"],
                         config["tf_record_compression"])
