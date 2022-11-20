import os
import pickle
import random
from time import time

import numpy as np
from albumentations import (Compose, Flip, HorizontalFlip, RandomBrightness,
                            Resize, ShiftScaleRotate)
from tensorflow.keras.utils import Sequence

from prostate_dataset.config import (dataset_folder, num_threads,
                                   preprocessed_folder)
from prostate_dataset.utils import get_list_of_patients, subfiles

random.seed(0)


def get_training_augmentation(patch_size):
    train_transform = [
        # HorizontalFlip(p=0.5),
        ShiftScaleRotate(
            p=1.0
        ),  # (shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
        #RandomBrightness(p=1.0, limit=(-0.1, 0.1)),
        Resize(*patch_size),
    ]
    return Compose(train_transform)


def get_validation_augmentation(patch_size):
    test_transform = [
        Resize(*patch_size),
    ]
    return Compose(test_transform)


class ProstateDataloader(Sequence):
    def __init__(self, dataset, batch_size=1, augmentation=None, shuffle=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        batch_data = []
        for j in range(start, stop):
            image, label = self.dataset[j]

            if self.augmentation:
                sample = self.augmentation(image=image, mask=label)
                image, label = sample["image"], sample["mask"]

            batch_data.append([image, label])

        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*batch_data)]

        return tuple(batch)

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


class ProstateDataset:
    def __init__(self, patients, only_non_empty_slices=False, skip_slices=0):
        self.only_non_empty_slices = only_non_empty_slices
        self.skip_slices=skip_slices
        self._get_list_of_files(patients)

    def _get_list_of_files(self, patients):
        files = []
        for patient in patients:
            files_for_patient = subfiles(
                preprocessed_folder, prefix=patient + "_", suffix="npy"
            )

            if self.only_non_empty_slices:
                non_empty_files_for_patient = []

                for file_for_patient in files_for_patient:
                    data = np.load(file_for_patient, allow_pickle=True)
                    mask = data[2]
                    labels = np.unique(mask)
                    if len(labels) == 1 and labels[0] == 0:
                        continue
                    else:
                        non_empty_files_for_patient.append(file_for_patient)

                files_for_patient = non_empty_files_for_patient

            files.extend(files_for_patient)

        self.files = files[::self.skip_slices+1]

    def __getitem__(self, i):
        data = np.load(self.files[i])

        image = np.moveaxis(data[0:2], 0, -1)        
        mask = data[2]

        return image, mask

    def __len__(self):
        return len(self.files)
