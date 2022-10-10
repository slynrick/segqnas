import os
import pickle
from time import time

import albumentations as A
import numpy as np
from tensorflow.keras.utils import Sequence

from spleen_dataset.config import dataset_folder, num_threads, preprocessed_folder
from spleen_dataset.utils import get_list_of_patients, subfiles


def get_training_augmentation(patch_size):
    train_transform = [
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(p=1),
        A.RandomBrightness(p=1, limit=(-0.1, 0.1)),
        A.Resize(*patch_size),
    ]
    return A.Compose(train_transform)


def get_validation_augmentation(patch_size):
    test_transform = [
        A.Resize(*patch_size),
    ]
    return A.Compose(test_transform)


class SpleenDataloader(Sequence):
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


class SpleenDataset:
    def __init__(self, patients, only_non_empty_slices=False):
        self.only_non_empty_slices = only_non_empty_slices
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
                    data = np.load(file_for_patient)
                    mask = data[1]
                    labels = np.unique(mask)
                    if len(labels) == 1 and labels[0] == 0:
                        continue
                    else:
                        non_empty_files_for_patient.append(file_for_patient)

                files_for_patient = non_empty_files_for_patient

            files.extend(files_for_patient)

        self.files = files

    def __getitem__(self, i):
        data = np.load(self.files[i])

        image = data[0]
        mask = data[1]

        return image, mask

    def __len__(self):
        return len(self.files)
