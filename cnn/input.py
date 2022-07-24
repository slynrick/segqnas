""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Input function and dataset info classes.

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/cifar10.py

"""

import os
from math import ceil

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image


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

        A.ShiftScaleRotate(scale_limit=(0.5, 2), rotate_limit=(-10,10), shift_limit=0.1, p=1, border_mode=0),

        A.PadIfNeeded(min_height=image_height, min_width=image_height, always_apply=True, border_mode=0),
        A.CropNonEmptyMaskIfExists(height=image_height, width=image_width, always_apply=True, ignore_channels=[0]),

        A.GaussianBlur(p=0.5),

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
        A.Lambda(mask=round_clip_0_1)
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
        #A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=48, pad_width_divisor=48, always_apply=True, border_mode=0),
    ]
    return A.Compose(test_transform)


class PascalVOC2012Dataset:
    """Pascal VOC 2012 Dataset. Read images, apply augmentation and preprocessing transformations.

    Args:
        image_paths (str): path to images folder
        mask_paths (str): path to segmentation masks folder
        classes (list): values of classes to extract from segmentation mask
        augmentation (albumentations.Compose): data transfromation pipeline
            (e.g. flip, scale, etc.)
        preprocessing (albumentations.Compose): data preprocessing
            (e.g. noralization, shape manipulation, etc.)

    """

    CLASSES = [
        "background",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "potted plant",
        "sheep",
        "sofa",
        "train",
        "tv/monitor",
    ]

    VOC_COLORMAP = [
        [0, 0, 0],
        [128, 0, 0],
        [0, 128, 0],
        [128, 128, 0],
        [0, 0, 128],
        [128, 0, 128],
        [0, 128, 128],
        [128, 128, 128],
        [64, 0, 0],
        [192, 0, 0],
        [64, 128, 0],
        [192, 128, 0],
        [64, 0, 128],
        [192, 0, 128],
        [64, 128, 128],
        [192, 128, 128],
        [0, 64, 0],
        [128, 64, 0],
        [0, 192, 0],
        [128, 192, 0],
        [0, 64, 128],
    ]

    def __init__(
        self,
        dataset_descriptor_filepath,
        images_path,
        masks_path,
        classes=None,
        image_height=448,
        image_width=448,
        augmentation=None,
        preprocessing=None,
    ):
        with open(dataset_descriptor_filepath, "r") as file:
            self.ids = file.read().splitlines()

        self.image_filepaths = [
            os.path.join(images_path, img_id + ".jpg") for img_id in self.ids
        ]
        self.mask_filepaths = [
            os.path.join(masks_path, img_id + ".png") for img_id in self.ids
        ]

        # convert str names to class values on masks
        if classes:
            self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        else:
            self.class_values = [
                self.CLASSES.index(cls.lower()) for cls in self.CLASSES
            ]

        self.image_height = image_height
        self.image_width = image_width
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image = cv2.imread(self.image_filepaths[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(self.mask_filepaths[i])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = self._convert_to_segmentation_mask(mask)

        # extract certain classes from mask (e.g. cars)
        #print(image.shape, mask.shape)

        #masks = [(mask == v) for v in self.class_values]
        #mask = np.stack(masks, axis=-1).astype('float')
        
        # add background if mask is not binary
        #if mask.shape[-1] != 1:
        #    background = 1 - mask.sum(axis=-1, keepdims=True)
        #    mask = np.concatenate((mask, background), axis=-1)
        
        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        
        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        return image, mask
        # # read data
        # image = Image.open(self.image_filepaths[i])
        # image = image.resize((self.image_width, self.image_height), Image.ANTIALIAS)
        # image = np.array(image)

        # mask = Image.open(self.mask_filepaths[i])
        # mask = mask.resize((self.image_width, self.image_height), Image.ANTIALIAS)
        # mask = np.array(mask, dtype=np.uint8)
        # mask = self._one_hot_encode_mask(mask)

        # # apply augmentations
        # if self.augmentation:
        #     sample = self.augmentation(image=image, mask=mask)
        #     image, mask = sample["image"], sample["mask"]

        # # apply preprocessing
        # if self.preprocessing:
        #     sample = self.preprocessing(image=image, mask=mask)
        #     image, mask = sample["image"], sample["mask"]

        # image = image.astype("float32")  # / 255
        # mask = mask.astype("float32")

        # return image, mask

    def __len__(self):
        return len(self.ids)

    # def _one_hot_encode_mask(self, mask):
    #     # create channel for mask
    #     # (height, width) => (height, width, 1)
    #     mask = mask[..., np.newaxis]

    #     # create a binary mask for each channel (class)
    #     one_hot_mask = []
    #     for _class in range(0, len(self.class_values)):
    #         class_mask = np.all(np.equal(mask, _class), axis=-1)
    #         one_hot_mask.append(class_mask)
    #     one_hot_mask = np.stack(one_hot_mask, axis=-1)
    #     one_hot_mask = one_hot_mask.astype("uint8")

    #     return one_hot_mask

    def _convert_to_segmentation_mask(self, mask):
        # This function converts a mask from the Pascal VOC format to the format required by AutoAlbument.
        #
        # Pascal VOC uses an RGB image to encode the segmentation mask for that image. RGB values of a pixel
        # encode the pixel's class.
        #
        # AutoAlbument requires a segmentation mask to be a NumPy array with the shape [height, width, num_classes].
        # Each channel in this mask should encode values for a single class. Pixel in a mask channel should have
        # a value of 1.0 if the pixel of the image belongs to this class and 0.0 otherwise.
        height, width = mask.shape[:2]
        segmentation_mask = np.zeros((height, width, len(self.VOC_COLORMAP)), dtype=np.float32)
        #segmentation_mask = np.zeros((height, width, 2), dtype=np.float32)
        for label_index, label in enumerate(self.VOC_COLORMAP):
            segmentation_mask[:, :, label_index] = np.all(mask == label, axis=-1).astype(float)
        #segmentation_mask[:, :, 0] = np.all(mask == self.VOC_COLORMAP[0], axis=-1).astype(float)
        #segmentation_mask[:, :, 1] = np.ones((height, width), dtype=np.float32) - np.all(mask == self.VOC_COLORMAP[0], axis=-1).astype(float)
        return segmentation_mask


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
