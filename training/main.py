import numpy as np
import cv2
import tensorflow as tf

record_files = tf.data.Dataset.list_files("/media/4TB/datasets/cats_vs_dogs/pre_processed_data/train/*.tfrecord", seed=42)

dataset = tf.data.TFRecordDataset(filenames=record_files, compression_type="GZIP")


def parse_image(record):
    features = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string)
    }
    parsed_record = tf.io.parse_single_example(record, features)
    image = tf.io.decode_jpeg(parsed_record['image_raw'], channels=3)
    image = tf.image.flip_left_right(image)
    # image = tf.image.convert_image_dtype(image, tf.float32)
    label = tf.cast(parsed_record['label'], tf.int32)

    return image, label


dataset = dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .map(lambda image, label: (tf.image.random_flip_up_down(image), label), num_parallel_calls=tf.data.experimental.AUTOTUNE) \
    .shuffle(buffer_size=1000) \
    .repeat() \
    .batch(2) \
    .prefetch(tf.data.experimental.AUTOTUNE)

for image, label in dataset.take(1):
    cv2.imwrite("test0.jpg", (tf.squeeze(image[0, :, :, :])).numpy())
    cv2.imwrite("test1.jpg", (tf.squeeze(image[1, :, :, :])).numpy())
    print(label)
