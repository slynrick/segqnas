""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Input function and dataset info classes.

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/cifar10.py

"""

import json
import os
from math import ceil

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras


# helper function for data visualization
def visualize(**images):
    """PLot images in one row."""
    cols = 6
    rows = ceil(len(images) / 6)
    plt.figure(figsize=(32, 10))
    for i, (name, image) in enumerate(images.items()):
        plt.subplot(rows, cols, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.title(" ".join(name.split("_")).title())
        plt.imshow(image)
    plt.show()


# helper function for data visualization
def denormalize(x):
    """Scale image to range 0..1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x


def round_clip_0_1(x, **kwargs):
    return x.round().clip(0, 1)


# define heavy augmentations
def get_training_augmentation(image_height, image_width):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=0.75),
        # A.RandomBrightness(limit=0.001, p=0.5)
        # A.PadIfNeeded(min_height=image_height, min_width=image_height, always_apply=True, border_mode=0),
        # A.CropNonEmptyMaskIfExists(height=image_height, width=image_width, always_apply=True, ignore_channels=[0]),
        # A.GaussianBlur(p=0.5),
        # A.IAAAdditiveGaussianNoise(p=0.2),
        # A.IAAPerspective(p=0.5),
        # A.OneOf(
        #     [
        #         A.CLAHE(p=1),
        #         A.RandomBrightness(p=1),
        #         A.RandomGamma(p=1),
        #     ],
        #     p=0.9,
        # ),
        # A.OneOf(
        #     [
        #         A.IAASharpen(p=1),
        #         A.Blur(blur_limit=3, p=1),
        #         A.MotionBlur(blur_limit=3, p=1),
        #     ],
        #     p=0.9,
        # ),
        # A.OneOf(
        #     [
        #         A.RandomContrast(p=1),
        #         A.HueSaturationValue(p=1),
        #     ],
        #     p=0.9,
        # ),
        # A.Lambda(mask=round_clip_0_1)
    ]
    return A.Compose(train_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        A.Lambda(image=preprocessing_fn),
    ]
    return A.Compose(_transform)


def get_validation_augmentation(image_height, image_width):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        A.Resize(image_height, image_width),
        # A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=48, pad_width_divisor=48, always_apply=True, border_mode=0),
    ]
    return A.Compose(test_transform)


class Dataloader(tf.keras.utils.Sequence):
    """Load data from dataset and form batches

    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """

    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]

        return tuple(batch)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


class SpleenDataset:
    def __init__(
        self,
        images_filepaths,
        labels_filepaths,
        augmentation=None,
    ):
        self.images_filepaths = images_filepaths
        self.labels_filepaths = labels_filepaths
        self.augmentation = augmentation

    def __getitem__(self, i):
        image = np.load(self.images_filepaths[i])
        label = np.load(self.labels_filepaths[i])

        image = image[..., np.newaxis]
        label = label[..., np.newaxis]

        image = image.astype("float32")
        mean = np.mean(image)
        std = np.std(image)
        image -= mean
        image /= std

        label = label.astype("float32")

        if self.augmentation:
            sample = self.augmentation(image=image, mask=label)
            image, label = sample["image"], sample["mask"]

        return image, label

    def __len__(self):
        return len(self.images_filepaths)


def get_train_val_filenames(data_path):
    descriptor_filepath = os.path.join(data_path, "dataset.json")
    with open(descriptor_filepath, "r") as fp:
        descriptor_dict = json.load(fp)

    patients = list(descriptor_dict.keys())

    train_patients, val_patients = train_test_split(patients, test_size=0.2)

    train_images_filepaths = []
    train_labels_filepaths = []
    val_images_filepaths = []
    val_labels_filepaths = []

    for patient in train_patients:
        for slice_ in descriptor_dict[patient]:
            train_images_filepaths.append(slice_[0])
            train_labels_filepaths.append(slice_[1])

    for patient in val_patients:
        for slice_ in descriptor_dict[patient]:
            val_images_filepaths.append(slice_[0])
            val_labels_filepaths.append(slice_[1])

    return (
        train_images_filepaths,
        val_images_filepaths,
        train_labels_filepaths,
        val_labels_filepaths,
    )
