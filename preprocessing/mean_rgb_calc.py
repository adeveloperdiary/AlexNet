import cv2
import glob
import progressbar
import numpy as np
import multiprocessing
import json
import argparse
import tensorflow as tf


def __worker_calculate_mean(files, cpu):
    (R, G, B) = ([], [], [])

    widgets = [
        'Calculating Mean - [', str(cpu), ']',
        progressbar.Bar('#', '[', ']'),
        ' [', progressbar.Percentage(), '] ',
        '[', progressbar.Counter(format='%(value)02d/%(max_value)d'), '] '

    ]

    bar = progressbar.ProgressBar(maxval=len(files), widgets=widgets)
    bar.start()

    for i, file in enumerate(files):
        image = cv2.imread(file)
        (b, g, r) = cv2.mean(image)[:3]
        R.append(r)
        G.append(g)
        B.append(b)
        bar.update(i + 1)
    bar.finish()
    return np.mean(R), np.mean(G), np.mean(B)


def __master_get_mean_rgb(files):
    cpu_core = multiprocessing.cpu_count()

    (R, G, B) = ([], [], [])
    item_per_thread = int(len(files) / cpu_core) + 1

    split_file_list = [files[x:x + item_per_thread] for x in range(0, len(files), item_per_thread)]
    p = multiprocessing.Pool(cpu_core)
    results = p.starmap(__worker_calculate_mean, zip(split_file_list, list(range(len(split_file_list)))))
    p.close()
    p.join()

    for val in results:
        R.append(val[0])
        G.append(val[1])
        B.append(val[2])

    return np.mean(R), np.mean(G), np.mean(B)


def __master_get_mean_rgb_from_tfrecord(files):
    def parse_image(record):
        features = {
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string)
        }
        parsed_record = tf.io.parse_single_example(record, features)
        image = tf.io.decode_jpeg(parsed_record['image_raw'], channels=3)
        label = tf.cast(parsed_record['label'], tf.int32)
        return image, label

    record_files = tf.data.Dataset.list_files(files)

    dataset = tf.data.TFRecordDataset(filenames=record_files, compression_type="GZIP")

    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .repeat(1) \
        .batch(1024) \
        .prefetch(tf.data.experimental.AUTOTUNE)

    rgb_mean_arr = []

    for i, (image, label) in enumerate(dataset.take(-1)):
        rgb_mean_arr.append(tf.reduce_mean(tf.cast(image, tf.float64), axis=(0, 1, 2)))

    return np.mean(np.array(rgb_mean_arr), axis=0)


def get_mean_rgb(image_dir, output_file, useTFRecord=False):
    files = glob.glob(image_dir)

    if useTFRecord:
        rgb_mean = __master_get_mean_rgb_from_tfrecord(files)
        R, G, B = rgb_mean[0], rgb_mean[1], rgb_mean[2]
    else:
        R, G, B = __master_get_mean_rgb(files)

    with open(output_file, "w+") as f:
        f.write(json.dumps({"R": R, "G": G, "B": B}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image_dir", required=True, help="path to the input image dir")
    ap.add_argument("-o", "--output_file_name", required=True, help="path to the json output")
    ap.add_argument("-tf", "--use_tfrecord", required=False, type=bool, default=False, help="path to the json output")
    args = vars(ap.parse_args())

    get_mean_rgb(args["image_dir"], args["output_file_name"], args["use_tfrecord"])
