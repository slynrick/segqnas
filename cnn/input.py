""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Input function and dataset info classes.

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/cifar10.py

"""

import os

import numpy as np
import tensorflow as tf
from PIL import Image

tf.compat.v1.disable_v2_behavior()

def load_pascalvoc12_sample_names(dataset_path, dataset_type):
    if dataset_type == "train":
        data_descriptor_file_path = os.path.join(
            dataset_path,
            "VOCdevkit",
            "VOC2012",
            "ImageSets",
            "Segmentation",
            "train.txt",
        )
    elif dataset_type == "val":
        data_descriptor_file_path = os.path.join(
            dataset_path, "VOCdevkit", "VOC2012", "ImageSets", "Segmentation", "val.txt"
        )
    elif dataset_type == "test":
        data_descriptor_file_path = os.path.join(
            dataset_path,
            "VOCdevkit",
            "VOC2012",
            "ImageSets",
            "Segmentation",
            "test.txt",
        )

    with open(data_descriptor_file_path, "r") as data_descriptor_file:
        return data_descriptor_file.read().splitlines()


class PascalVOC2012DataGenerator(tf.keras.utils.Sequence):
    def __init__(
        self,
        sample_names,
        img_path,
        mask_path,
        height=448,
        width=448,
        num_channels=3,
        num_classes=21,
        batch_size=32,
        is_train=True,
        shuffle=False,
        transforms=False,
    ):
        self.height = height
        self.width = width
        self.num_channels = num_channels
        self.num_classes = num_classes
        self.sample_names = sample_names
        self.img_path = img_path
        self.mask_path = mask_path
        self.batch_size = batch_size
        self.is_train = is_train
        self.shuffle = shuffle
        self.augment = transforms
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.sample_names) / self.batch_size))

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]

        batch_sample_names = [self.sample_names[k] for k in indexes]

        X = self._generate_imgs(batch_sample_names)

        if self.is_train:
            y = self._generate_masks(batch_sample_names)
            return X, y
        else:
            return X

    def _generate_imgs(self, indexes):
        X = np.empty((self.batch_size, self.width, self.height, self.num_channels))

        for i, sample_name in enumerate(indexes):
            img_path = os.path.join(self.img_path, sample_name + ".jpg")
            img = self._load_img(img_path)
            X[
                i,
            ] = img

        return X

    def _load_img(self, img_path):
        img = Image.open(img_path)
        img = img.resize((self.width, self.height), Image.ANTIALIAS)
        img = np.array(img)
        img = img / 255

        # img = cv2.imread(img_path)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.resize(img, (self.width, self.height), interpolation = cv2.INTER_AREA)
        # img = img / 255
        return img

    def _generate_masks(self, indexes):
        y = np.empty((self.batch_size, self.width, self.height, self.num_classes))

        for i, sample_name in enumerate(indexes):
            mask_path = os.path.join(self.mask_path, sample_name + ".png")
            y[
                i,
            ] = self._load_mask(mask_path)

        return y

    def _load_mask(self, mask_path):
        mask = Image.open(mask_path)
        mask = mask.resize((self.width, self.height), Image.ANTIALIAS)
        mask = np.array(mask)
        mask = self._one_hot_encode_mask(mask)
        return mask

    def _one_hot_encode_mask(self, mask):
        # create channel for mask
        # (height, width) => (height, width, 1)
        mask = mask[..., np.newaxis]

        # create a binary mask for each channel (class)
        one_hot_mask = []
        for _class in range(0, self.num_classes):
            class_mask = np.all(np.equal(mask, _class), axis=-1)
            one_hot_mask.append(class_mask)
        one_hot_mask = np.stack(one_hot_mask, axis=-1)
        one_hot_mask = one_hot_mask.astype("int32")

        return one_hot_mask

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.sample_names))
        if self.shuffle:
            np.random.shuffle(self.sample_names)

    def augment(self):
        pass
