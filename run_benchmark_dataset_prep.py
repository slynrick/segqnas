import argparse
import os
import shutil
from multiprocessing import Pool

import datasets.prostate_dataset.config as prostate_config
import datasets.spleen_dataset.config as spleen_config
import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import (
    join, maybe_mkdir_p, subfiles)
from monai.apps import download_and_extract
from sklearn.model_selection import train_test_split


def get_list_of_patients(base_dir, patient_filename_prefix):
    ct_directory = join(base_dir, "imagesTr")
    ct_files = subfiles(ct_directory, prefix=patient_filename_prefix, join=False)
    patients = list(map(get_patient_from_filename, ct_files))
    return patients


def get_patient_from_filename(filename):
    return (filename.split(".")[0]).split("_")[-1]


def get_list_of_files(base_dir, patient_filename_prefix, patient_filename_suffix):
    """
    returns a list of lists containing the filenames. The outer list contains all training examples. Each entry in the
    outer list is again a list pointing to the files of that training example in the following order:
    CT, segmentation
    :param base_dir:
    :return:
    """
    list_of_lists = []

    ct_directory = join(base_dir, "imagesTr")
    segmentation_directory = join(base_dir, "labelsTr")
    patients = get_list_of_patients(base_dir, patient_filename_prefix)

    for patient in patients:
        patient_filename = patient_filename_prefix + "_" + patient + patient_filename_suffix
        ct_file = join(ct_directory, patient_filename)
        segmentation_file = join(segmentation_directory, patient_filename)
        this_case = [ct_file, segmentation_file]
        list_of_lists.append(this_case)

    return list_of_lists


def load_and_preprocess_prostate(case, patient_name, output_folder):
    """
    loads, preprocesses and saves a case
    This is what happens here:
    1) load all images and stack them to a 4d array
    2) crop to nonzero region, this removes unnecessary zero-valued regions and reduces computation time
    3) normalize the nonzero region with its mean and standard deviation
    4) save 4d tensor as numpy array. Also save metadata required to create niftis again (required for export
    of predictions)
    :param case:
    :param patient_name:
    :return:
    """
    # load SimpleITK Images
    imgs_sitk = [sitk.ReadImage(i) for i in case]

    # get pixel arrays from SimpleITK images
    imgs_npy = [sitk.GetArrayFromImage(i) for i in imgs_sitk]

    imgs_npy = [i[np.newaxis, ...] if len(i.shape) == 3 else i for i in imgs_npy]

    # now stack the images into one 4d array, cast to float because we will get rounding problems if we don't
    imgs_npy = np.concatenate(imgs_npy).astype(np.float32)

    for slice_idx in range(imgs_npy.shape[1]):
        slice_npy = imgs_npy[:, slice_idx, :, :]

        mean = slice_npy[0].mean()
        std = slice_npy[0].std()
        slice_npy[0] = (slice_npy[0] - mean) / (std + 1e-8)

        mean = slice_npy[1].mean()
        std = slice_npy[1].std()
        slice_npy[1] = (slice_npy[1] - mean) / (std + 1e-8)

        np.save(
            join(output_folder, patient_name + "_" + str(slice_idx) + ".npy"), slice_npy
        )

def load_and_preprocess_spleen(case, patient_name, output_folder):
    """
    loads, preprocesses and saves a case
    This is what happens here:
    1) load all images and stack them to a 4d array
    2) crop to nonzero region, this removes unnecessary zero-valued regions and reduces computation time
    3) normalize the nonzero region with its mean and standard deviation
    4) save 4d tensor as numpy array. Also save metadata required to create niftis again (required for export
    of predictions)
    :param case:
    :param patient_name:
    :return:
    """
    # load SimpleITK Images
    imgs_sitk = [sitk.ReadImage(i) for i in case]

    # get pixel arrays from SimpleITK images
    imgs_npy = [sitk.GetArrayFromImage(i) for i in imgs_sitk]

    imgs_npy = np.concatenate([i[None] for i in imgs_npy]).astype(np.float32)

    for slice_idx in range(imgs_npy.shape[1]):
        slice_npy = imgs_npy[:, slice_idx, :, :]

        slice_npy[0] = np.clip(slice_npy[0], -57, 164)
        slice_npy[0] = (slice_npy[0] - np.min(slice_npy[0])) / np.ptp(slice_npy[0])

        np.save(
            join(output_folder, patient_name + "_" + str(slice_idx) + ".npy"), slice_npy
        )

def split_by_train_val_test(preprocessed_folder, mode='image'):
    if mode not in ['image', 'patiente']:
        print("Not splitted because the mode selected is not implemented")
    files = list((f, f.replace(".npy", ""), f.replace(".npy", "").split('_')[0]) for f in os.listdir(preprocessed_folder))
    patients = np.unique([f[-1] for f in files])
    images = np.unique([f[0] for f in files])

    maybe_mkdir_p(join(preprocessed_folder, 'train'))
    maybe_mkdir_p(join(preprocessed_folder, 'test'))
    if mode == 'image':
        train, test = train_test_split(images, test_size=0.2)
        print(f"Images in train/val: {train}")
        print(f"Images in test: {test}")
        print("Starting split")
        for fname, _, _ in files:
            if fname in train:
                shutil.copy(join(preprocessed_folder, fname), join(preprocessed_folder, 'train'))
            if fname in test:
                shutil.copy(join(preprocessed_folder, fname), join(preprocessed_folder, 'test'))
    elif mode == 'patiente':
        train, test = train_test_split(patients, test_size=0.2)
        print(f"Patients in train/val: {train}")
        print(f"Patients in test: {test}")
        print("Starting split")
        for fname, _, patient in files:
            if patient in train:
                shutil.copy(join(preprocessed_folder, fname), join(preprocessed_folder, 'train'))
            if patient in test:
                shutil.copy(join(preprocessed_folder, fname), join(preprocessed_folder, 'test'))
    

def download_dataset(root_folder, dataset_folder, output_tar, resource, md5):
    compressed_file = os.path.join(root_folder, output_tar)

    if not os.path.exists(dataset_folder):
        download_and_extract(resource, compressed_file, root_folder, md5)


def main(dataset, root_folder, dataset_folder, preprocessed_folder, num_threads, resource, md5, output_tar, patient_filename_prefix, patient_filename_suffix, split_mode):
    download_dataset(root_folder, dataset_folder, output_tar, resource, md5)

    list_of_lists = get_list_of_files(dataset_folder, patient_filename_prefix, patient_filename_suffix)
    list_of_patient_names = get_list_of_patients(dataset_folder, patient_filename_prefix)

    maybe_mkdir_p(preprocessed_folder)

    p = Pool(processes=num_threads)
    p.starmap(
        load_and_preprocess_prostate if dataset == 'prostate' else load_and_preprocess_spleen,
        zip(
            list_of_lists,
            list_of_patient_names,
            [preprocessed_folder] * len(list_of_lists),
        ),
    )
    p.close()
    p.join()

    split_by_train_val_test(preprocessed_folder, split_mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download datasets')
    parser.add_argument('-d','--dataset', type=str, required=True, help="Select a dataset to be downloaded: 'prostate' or 'spleen'")
    parser.add_argument('-t','--threads', type=int, default=4, help="Number os threads to download the dataset")
    parser.add_argument('-s','--split_mode', type=str, required=True, help="Select a split mode: 'image' or 'patiente'")


    args = parser.parse_args()

    num_threads = args.threads
    root_folder = ""
    dataset_folder = ""
    preprocessed_folder = ""
    resource = ""
    md5 = ""
    output_tar = ""
    patient_filename_prefix = ""
    patient_filename_suffix = ""
    split_mode = args.split_mode
    if args.dataset == 'prostate':
        root_folder = prostate_config.root_folder
        dataset_folder = prostate_config.dataset_folder
        preprocessed_folder = prostate_config.preprocessed_folder
        resource = prostate_config.resource_url
        md5 = prostate_config.resource_md5
        output_tar = prostate_config.output_tar
        patient_filename_prefix = prostate_config.patient_filename_prefix
        patient_filename_suffix = prostate_config.patient_filename_suffix
    elif args.dataset == 'spleen':
        root_folder = spleen_config.root_folder
        dataset_folder = spleen_config.dataset_folder
        preprocessed_folder = spleen_config.preprocessed_folder
        resource = spleen_config.resource_url
        md5 = spleen_config.resource_md5
        output_tar = spleen_config.output_tar
        patient_filename_prefix = spleen_config.patient_filename_prefix
        patient_filename_suffix = spleen_config.patient_filename_suffix

    main(args.dataset, root_folder, dataset_folder, preprocessed_folder, num_threads, resource, md5, 
         output_tar, patient_filename_prefix, patient_filename_suffix, split_mode)
