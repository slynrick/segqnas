import random
import os
import pickle
import numpy as np
from sklearn.model_selection import KFold

from albumentations import (
    Compose,
    HorizontalFlip,
    RandomBrightness,
    Resize,
    RandomBrightnessContrast,
    ShiftScaleRotate,
)
from tensorflow.keras.utils import Sequence

random.seed(0)


def get_list_of_patients(data_path):
    filenames = os.listdir(data_path)

    def get_patient_from_filename(filename):
        return (filename.split(".")[0]).split("_")[0]

    patients = list(set(map(get_patient_from_filename, filenames)))
    patients.sort(key=float)
    return patients


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join
    else:
        l = lambda x, y: y
    res = [
        l(folder, i)
        for i in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, i))
        and (prefix is None or i.startswith(prefix))
        and (suffix is None or i.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def load_pickle(file, mode="rb"):
    with open(file, mode) as f:
        a = pickle.load(f)
    return a


def get_split_deterministic(all_keys, fold=0, num_splits=5, random_state=12345):
    """
    Splits a list of patient identifiers (or numbers) into num_splits folds and returns the split for fold fold.
    :param all_keys:
    :param fold:
    :param num_splits:
    :param random_state:
    :return:
    """
    all_keys_sorted = np.sort(list(all_keys))
    splits = KFold(n_splits=num_splits, shuffle=True, random_state=random_state)
    for i, (train_idx, test_idx) in enumerate(splits.split(all_keys_sorted)):
        if i == fold:
            train_keys = np.array(all_keys_sorted)[train_idx]
            test_keys = np.array(all_keys_sorted)[test_idx]
            break
    return train_keys, test_keys


def get_training_augmentation(patch_size):
    train_transform = [
        #HorizontalFlip(p=0.5),
        ShiftScaleRotate(
            p=1.0
        ),  # (shift_limit=0.0625, scale_limit=0.1, rotate_limit=45),
        RandomBrightness(p=1.0, limit=(-0.01, 0.01)),
        Resize(*patch_size),
    ]
    return Compose(train_transform)


def get_validation_augmentation(patch_size):
    test_transform = [
        Resize(*patch_size),
    ]
    return Compose(test_transform)


class Dataloader(Sequence):
    def __init__(
        self, dataset, batch_size=1, skip_slices=0, augmentation=None, shuffle=True
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.skip_slices = skip_slices
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.indexes = np.arange(len(dataset))
        self.on_epoch_end()

    def __getitem__(self, i):

        # collect batch data
        start = (i * (self.skip_slices + 1)) * self.batch_size
        stop = (i * (self.skip_slices + 1) + 1) * self.batch_size
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
        return len(self.indexes) // (self.batch_size * (self.skip_slices + 1))

    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)


class Dataset:
    def __init__(self, data_path, patients, only_non_empty_slices=False):
        self.only_non_empty_slices = only_non_empty_slices
        self.data_path = data_path
        self._get_list_of_files(patients)

    def _get_list_of_files(self, patients):
        files = []
        for patient in patients:
            files_for_patient = subfiles(
                self.data_path, prefix=patient + "_", suffix="npy"
            )

            if self.only_non_empty_slices:
                non_empty_files_for_patient = []

                for file_for_patient in files_for_patient:
                    data = np.load(file_for_patient, allow_pickle=True)
                    mask = data[-1]
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

        image = np.moveaxis(data[0:-1], 0, -1)
        mask = data[-1]

        return image, mask

    def __len__(self):
        return len(self.files)
