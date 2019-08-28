import numpy as np
import cv2
import tensorflow as tf
import json

record_files = tf.data.Dataset.list_files("/media/4TB/datasets/cats_vs_dogs/pre_processed_data/train/*.tfrecord", seed=42)

dataset = tf.data.TFRecordDataset(filenames=record_files, compression_type="GZIP")

with open("rgb.json", "r") as f:
    rgb_mean_dict = json.loads(f.read())

mean = np.zeros((3))
mean[0] = rgb_mean_dict["R"]
mean[1] = rgb_mean_dict["G"]
mean[2] = rgb_mean_dict["B"]

rgb_mean = tf.constant(mean)
rgb_mean = tf.reshape(rgb_mean, [1, 1, 3])
rgb_mean = tf.image.convert_image_dtype(rgb_mean, tf.uint8)


def parse_image(record):
    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_record = tf.io.parse_single_example(record, features)
    image = tf.io.decode_jpeg(parsed_record['image_raw'], channels=3)
    image = tf.image.flip_left_right(image)
    label = tf.cast(parsed_record['label'], tf.int32)
    image = image - rgb_mean
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


dataset = dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .map(lambda image, label: (tf.image.random_flip_left_right(image), label), num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .map(lambda image, label: (tf.image.random_crop(image, size=[227, 227, 3]), label), num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000) \
    .repeat() \
    .batch(2) \
    .prefetch(tf.data.experimental.AUTOTUNE)

for image, label in dataset.take(1):
    print(np.mean((tf.squeeze(image[0, :, :, :])).numpy()))
    print(np.min((tf.squeeze(image[0, :, :, :])).numpy()))
    print(np.max((tf.squeeze(image[0, :, :, :])).numpy()))
    cv2.imwrite("test0.jpg", (tf.squeeze(image[0, :, :, :])).numpy()*255)
    cv2.imwrite("test1.jpg", (tf.squeeze(image[1, :, :, :])).numpy()*255)
    print(label)
