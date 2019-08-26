import numpy as np
import cv2
import os
from multiprocessing import Pool, TimeoutError
import glob
import tensorflow as tf


def scale_image(image, size=256):
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


def center_crop(image, size=256):
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


def resize_center_crop(image, size):
    image = scale_image(image, size)
    image = center_crop(image, size)

    return image


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


if __name__ == '__main__':
    '''
    path = "/media/4TB/datasets/cats_vs_dogs/train/*.jpg"
    files = glob.glob(path)

    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]
    options = tf.io.TFRecordOptions(compression_type="GZIP")
    with tf.io.TFRecordWriter("/media/4TB/datasets/cats_vs_dogs/train.tfrecord", options) as writer:
        for i, file in enumerate(files):

            image = resize_center_crop(cv2.imread(file), 256)
            is_success, im_buf_arr = cv2.imencode(".jpg", image, encode_param)
            image_raw = im_buf_arr.tobytes()
            # img_bytes = open(file, 'rb').read()

            row = tf.train.Example(features=tf.train.Features(feature={
                'label': _int64_feature(1),
                'image_raw': _bytes_feature(image_raw)
            }))
            writer.write(row.SerializeToString())
            if i % 1000 == 0:
                print(i)
    '''

    raw_image_dataset = tf.data.TFRecordDataset('/media/4TB/datasets/cats_vs_dogs/train.tfrecord')
    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }

    for serialized_example in tf.data.TFRecordDataset(["/media/4TB/datasets/cats_vs_dogs/train.tfrecord"],compression_type="GZIP"):
        parsed_example = tf.io.parse_single_example(serialized_example,features)
        image_raw=parsed_example['image_raw']

        img=tf.io.decode_jpeg(image_raw).numpy()
        cv2.imwrite("/media/4TB/datasets/cats_vs_dogs/test.jpg",cv2.cvtColor(img,cv2.COLOR_BGR2RGB))



