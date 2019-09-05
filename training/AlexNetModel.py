import numpy as np
import cv2
import tensorflow as tf


def AlexNetModel(classes):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), padding="valid", input_shape=[227, 227, 3],
                                     kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.compat.v2.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5), padding="valid", kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.compat.v2.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding="valid", kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.compat.v2.keras.layers.BatchNormalization(axis=-1))

    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3), padding="valid", kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.compat.v2.keras.layers.BatchNormalization(axis=-1))

    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="valid", kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.compat.v2.keras.layers.BatchNormalization(axis=-1))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(4096, kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.compat.v2.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(4096, kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Activation("relu"))
    model.add(tf.compat.v2.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(0.5))

    model.add(tf.keras.layers.Dense(classes, kernel_initializer="he_normal"))
    model.add(tf.keras.layers.Activation("softmax"))

    return model
