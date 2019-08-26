import cv2
import glob
import tensorflow as tf
import progressbar
import numpy as np
import multiprocessing
import datetime


def worker_calculate_mean(files):
    (R, G, B) = ([], [], [])

    widgets = [
        'Calculating Mean - ',
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
    results = p.map(worker_calculate_mean, split_file_list)
    p.close()
    p.join()

    for val in results:
        R.append(val[0])
        G.append(val[1])
        B.append(val[2])

    return np.mean(R), np.mean(G), np.mean(B)


class ImagePreprocess:

    def __init__(self, image_folder, record_path, identifier, size=256, split_number=1000, image_quality=70):
        self.image_folder = image_folder
        self.record_path = record_path
        self.size = size
        self.encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), image_quality]
        self.split_number = split_number
        self.label_map = {}
        self.tf_record_options = tf.io.TFRecordOptions(compression_type="GZIP")
        # identifier=train/test/val
        self.identifier = identifier
        self.label_sequence = 0
        self.mean_rgb = {}

    def __scale_image(self, image):
        image_height, image_width = image.shape[:2]

        if image_height <= image_width:
            ratio = image_width / image_height
            h = self.size
            w = int(ratio * 256)

            image = cv2.resize(image, (w, h))

        else:
            ratio = image_height / image_width
            w = self.size
            h = int(ratio * 256)

            image = cv2.resize(image, (w, h))

        return image

    def __center_crop(self, image):
        image_height, image_width = image.shape[:2]

        if image_height <= image_width and abs(image_width - self.size) > 1:

            dx = int((image_width - self.size) / 2)
            image = image[:, dx:-dx, :]
        elif abs(image_height - self.size) > 1:
            dy = int((image_height - self.size) / 2)
            image = image[dy:-dy, :, :]

        image_height, image_width = image.shape[:2]
        if image_height is not self.size and image_width is not self.size:
            image = cv2.resize(image, (self.size, self.size))

        return image

    def __process_image(self, image):

        '''
        (B, G, R) = cv2.split(image.astype("float32"))
        R -= self.mean_rgb["R"]
        G -= self.mean_rgb["G"]
        B -= self.mean_rgb["B"]

        image = cv2.merge([B, G, R])
        '''

        image = self.__scale_image(image)
        image = self.__center_crop(image)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        return image

    def __get_mean_rgb(self, files, stop_at):
        startTime = datetime.datetime.now().replace(microsecond=0)
        self.mean_rgb["R"], self.mean_rgb["G"], self.mean_rgb["B"] = get_mean_rgb(files[:stop_at])
        endTime = datetime.datetime.now().replace(microsecond=0)
        print(datetime.time(0, 0, (endTime - startTime).seconds).strftime("%M:%S"))

    def __int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def __bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def __get_label_number(self, label_str):
        if label_str in self.label_map:
            return self.label_map[label_str]
        else:
            self.label_map[label_str] = self.label_sequence
            self.label_sequence += 1
            return self.label_map[label_str]

    def create_tf_record(self):
        files = glob.glob(self.image_folder)

        tf_writer = tf.io.TFRecordWriter(self.record_path + self.identifier + "-0.tfrecord", self.tf_record_options)

        data_len = len(files)
        if data_len / self.split_number is not 0:
            stop_at = int(data_len / self.split_number) * self.split_number
        else:
            stop_at = data_len

        self.__get_mean_rgb(files, stop_at)

        widgets = [
            'Processing Images - ',
            progressbar.Bar('#', '[', ']'),
            ' [', progressbar.Percentage(), '] ',
            '[', progressbar.Counter(format='%(value)02d/%(max_value)d'), '] '

        ]

        bar = progressbar.ProgressBar(maxval=stop_at, widgets=widgets)
        bar.start()

        for i, file in enumerate(files):

            if i < stop_at:
                image = self.__process_image(cv2.imread(file))
                is_success, im_buf_arr = cv2.imencode(".jpg", image, self.encode_param)
                if is_success:

                    label_str = file.split("/")[-2]
                    label_number = self.__get_label_number(label_str)

                    image_raw = im_buf_arr.tobytes()

                    row = tf.train.Example(features=tf.train.Features(feature={
                        'label': self.__int64_feature(label_number),
                        'image_raw': self.__bytes_feature(image_raw)
                    }))

                    if i % self.split_number == 0:
                        seq = int(i / self.split_number)
                        tf_writer.close()
                        tf_writer = tf.io.TFRecordWriter(self.record_path + self.identifier + "-" + str(seq) + ".tfrecord", self.tf_record_options)

                    tf_writer.write(row.SerializeToString())
                    bar.update(i + 1)
                else:
                    print("Error processing " + file)
            else:
                break

        tf_writer.close()
        bar.finish()

        return self.label_map, self.mean_rgb


if __name__ == '__main__':
    pre_process = ImagePreprocess(image_folder="/media/4TB/datasets/cats_vs_dogs/train/**/*.jpg",
                                  record_path="/media/4TB/datasets/cats_vs_dogs/tf_record/train/", identifier="train")
    print(pre_process.create_tf_record())
