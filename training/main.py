import numpy as np
import cv2
import tensorflow as tf
import json

with open("../preprocessing/imagenet_mean_rgb_tf.json", "r") as f:
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
    # image = tf.image.flip_left_right(image)
    label = tf.cast(parsed_record['label'], tf.int32)

    # onehot_label = tf.Variable(lambda:tf.zeros(159, dtype=tf.dtypes.float32, name=None))
    # onehot_label = onehot_label[label].assign(1.0)

    image = image - rgb_mean
    image = tf.image.convert_image_dtype(image, tf.float32)
    return image, label


def get_dataset(path, batch_size=1024):
    record_files = tf.data.Dataset.list_files(path, seed=42)

    dataset = tf.data.TFRecordDataset(filenames=record_files, compression_type="GZIP")

    # dataset = record_files.interleave(tf.data.TFRecordDataset, cycle_length=8, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(lambda image, label: (tf.image.random_flip_left_right(image), label), num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .map(lambda image, label: (tf.image.random_crop(image, size=[227, 227, 3]), label), num_parallel_calls=tf.data.experimental.AUTOTUNE) \
        .repeat() \
        .batch(batch_size, drop_remainder=True) \
        .prefetch(buffer_size=10)
    # .shuffle(buffer_size=1024) \

    return dataset


'''
train_dataset = get_dataset("/media/4TB/datasets/ILSVRC2015/ILSVRC2015/tf_records_100/train/*.tfrecord")

for image, label in train_dataset.take(1):
    print(np.mean((tf.squeeze(image[0, :, :, :])).numpy()))
    print(np.min((tf.squeeze(image[0, :, :, :])).numpy()))
    print(np.max((tf.squeeze(image[0, :, :, :])).numpy()))
    cv2.imwrite("test0.jpg", (tf.squeeze(image[0, :, :, :])).numpy() * 255)
    cv2.imwrite("test1.jpg", (tf.squeeze(image[1, :, :, :])).numpy() * 255)
    print(label)

'''
train_dataset = get_dataset("/media/home/1TB_Disk/data/tf_records_100/train/*.tfrecord")
val_dataset = get_dataset("/media/home/1TB_Disk/data/tf_records_100/val/*.tfrecord")
# test_dataset = get_dataset("/media/4TB/datasets/ILSVRC2015/ILSVRC2015/tf_records_100/test/*.tfrecord",batch_size=8000)

# train_dataset = get_dataset("/media/4TB/datasets/cats_vs_dogs/pre_processed_data/train/*.tfrecord")
# val_dataset = get_dataset("/media/4TB/datasets/cats_vs_dogs/pre_processed_data/val/*.tfrecord")
# train_dataset = get_dataset("/media/4TB/datasets/cats_vs_dogs/pre_processed_data/test/*.tfrecord", batch_size=256)

from AlexNetModel import AlexNetModel
import datetime


def scheduler(epoch):
    if epoch <= 30:
        return 0.01
    if 60 >= epoch > 30:
        return 0.001
    else:
        return 0.0001


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.04, momentum=0.9, decay=0.0005)

mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    model = AlexNetModel(159)

    model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrices=["accuracy"])

    history = model.fit(train_dataset, epochs=2, steps_per_epoch=191, validation_data=val_dataset, validation_steps=7,
                        callbacks=[tensorboard_callback, lr_callback])

    # model.save("model_159_1.h5")

'''
new_model = tf.keras.models.load_model('training/model_159_1.h5')

from sklearn.metrics import classification_report

for image, label in test_dataset.take(1):
    prediction = new_model.predict(image)
    print(classification_report(label, prediction.argmax(axis=1), target_names=j.keys()))
'''
