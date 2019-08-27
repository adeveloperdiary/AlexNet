import cv2
import glob
import tensorflow as tf
import progressbar
import numpy as np
import multiprocessing
from itertools import repeat


def worker_calculate_mean(files, cpu):
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


def get_mean_rgb(files):
    cpu_core = multiprocessing.cpu_count()

    (R, G, B) = ([], [], [])
    item_per_thread = int(len(files) / cpu_core) + 1

    split_file_list = [files[x:x + item_per_thread] for x in range(0, len(files), item_per_thread)]
    p = multiprocessing.Pool(cpu_core)
    results = p.starmap(worker_calculate_mean, zip(split_file_list, list(range(cpu_core))))
    p.close()
    p.join()

    for val in results:
        R.append(val[0])
        G.append(val[1])
        B.append(val[2])

    return np.mean(R), np.mean(G), np.mean(B)


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
    if image_height is not size and image_width is not size:
        image = cv2.resize(image, (size, size))

    return image


def process_image(image, size):
    '''
    (B, G, R) = cv2.split(image.astype("float32"))
    R -= self.mean_rgb["R"]
    G -= self.mean_rgb["G"]
    B -= self.mean_rgb["B"]

    image = cv2.merge([B, G, R])
    '''

    image = scale_image(image, size)
    image = center_crop(image, size)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def worker_tf_write(files, tf_record_path, label_map, size, image_quality, number):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), image_quality]
    tf_record_options = tf.io.TFRecordOptions(compression_type="GZIP")

    widgets = [
        'Processing Images - [', str(number), ']',
        progressbar.Bar('#', '[', ']'),
        ' [', progressbar.Percentage(), '] ',
        '[', progressbar.Counter(format='%(value)02d/%(max_value)d'), '] '

    ]

    bar = progressbar.ProgressBar(maxval=len(files), widgets=widgets)
    bar.start()

    with tf.io.TFRecordWriter(tf_record_path,tf_record_options) as tf_writer:
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


def master_tf_write(split_file_list, tf_record_paths, size, image_quality, label_map):
    cpu_core = multiprocessing.cpu_count()

    p = multiprocessing.Pool(cpu_core)
    results = p.starmap(worker_tf_write, zip(split_file_list, tf_record_paths, repeat(label_map), repeat(size), repeat(image_quality),
                                             list(range(len(tf_record_paths)))))
    p.close()
    p.join()


class ImagePreprocess:

    def __init__(self, image_folder, record_path, identifier, size=256, split_number=1000, image_quality=70):
        self.image_folder = image_folder
        self.record_path = record_path
        self.size = size
        self.image_quality = image_quality
        self.split_number = split_number
        self.label_map = {}
        self.tf_record_options = tf.io.TFRecordOptions(compression_type="GZIP")
        # identifier=train/test/val
        self.identifier = identifier
        self.label_sequence = 0
        self.mean_rgb = {}

    def create_tf_record(self):
        files = glob.glob(self.image_folder)

        data_len = len(files)
        if data_len / self.split_number is not 0:
            stop_at = int(data_len / self.split_number) * self.split_number
        else:
            stop_at = data_len

        files = files[:stop_at]

        self.mean_rgb["R"], self.mean_rgb["G"], self.mean_rgb["B"] = get_mean_rgb(files)

        split_file_list = [files[x:x + self.split_number] for x in range(0, len(files), self.split_number)]

        tf_record_paths = []

        for i in range(len(split_file_list)):
            tf_record_paths.append(self.record_path + self.identifier + "-" + str(i) + ".tfrecord")

        label_map = {
            "cat": 0,
            "dog": 1
        }

        master_tf_write(split_file_list, tf_record_paths, self.size, self.image_quality, label_map)

        return self.mean_rgb


if __name__ == '__main__':
    pre_process = ImagePreprocess(image_folder="/media/4TB/datasets/cats_vs_dogs/train/**/*.jpg",
                                  record_path="/media/4TB/datasets/cats_vs_dogs/tf_record/train/", identifier="train")
    print(pre_process.create_tf_record())
