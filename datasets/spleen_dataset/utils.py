import os
import pickle

import numpy as np
from sklearn.model_selection import KFold


def get_list_of_patients(base_dir):
    ct_directory = os.path.join(base_dir, "imagesTr")
    ct_files = subfiles(ct_directory, prefix="spleen", join=False)
    patients = list(map(get_patient_from_filename, ct_files))
    return patients


def get_patient_from_filename(filename):
    return (filename.split(".")[0]).split("_")[-1]


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
