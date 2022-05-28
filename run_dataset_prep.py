""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Read PascalVOC2012 data and writes into TFRecords files.
"""

import argparse
import os
import shutil
import tarfile
from PIL import Image
import numpy as np
import random

VALID_DATA_RATIO = 0.1
TRAIN_EX = 50000
TEST_EX = 10000
NUM_BINS = 5
NUM_CLASSES = 21  # 20 classes + 1 background

import util


def load_pascalvoc12(data_path):
    """Download PASCAL VOC 2012 and load the images for training, validationand test sets and masks for training and validation sets.
    The masks are not provided for the test dataset and evaluation must be done on their evaluation server.

    Args:
        data_path: (str) path to the directory where PASCAL VOC 2012 is, or where to download it to.

    Returns:
        train_imgs, train_masks, val_imgs, val_masks, test_imgs.
    """
    pascalvoc12_train_val_url = (
        "http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"
    )
    pascalvoc12_test_url = "http://pjreddie.com/media/files/VOC2012test.tar"

    source_urls = [pascalvoc12_train_val_url, pascalvoc12_test_url]

    file_names = [source_url.split("/")[-1] for source_url in source_urls]

    file_paths = []

    for source_url, file_name in zip(source_urls, file_names):
        file_path = util.download_file(data_path, file_name, source_url)
        file_paths.append(file_path)

    # Download PASCAL VOC 2012 dataset (2 links are used, the first contains train and val sets and the second contains test set)
    # downloaded_file_paths = util.download_pascalvoc12(data_path)

    # Extract the .tar files downloaded to the data_path path
    for file_path in file_paths:
        print(f"Extracting {file_path}")
        tarfile.open(file_path, "r").extractall(data_path)
        os.remove(file_path)

    # Clean unused folders in dataset
    shutil.rmtree(
        os.path.join(data_path, "VOCdevkit", "VOC2012", "Annotations"),
        ignore_errors=True,
    )
    shutil.rmtree(
        os.path.join(data_path, "VOCdevkit", "VOC2012", "ImageSets", "Action"),
        ignore_errors=True,
    )
    shutil.rmtree(
        os.path.join(data_path, "VOCdevkit", "VOC2012", "ImageSets", "Layout"),
        ignore_errors=True,
    )
    shutil.rmtree(
        os.path.join(data_path, "VOCdevkit", "VOC2012", "ImageSets", "Main"),
        ignore_errors=True,
    )
    shutil.rmtree(
        os.path.join(data_path, "VOCdevkit", "VOC2012", "SegmentationObject"),
        ignore_errors=True,
    )

    # Relevant folders for the data
    # Directory where train.txt and val.txt files are. These files contain the name of the images in the training and validation datasets
    descriptor_files_folder = os.path.join(
        data_path, "VOCdevkit", "VOC2012", "ImageSets", "Segmentation"
    )
    # Directory where the images are
    img_files_folder = os.path.join(data_path, "VOCdevkit", "VOC2012", "JPEGImages")
    # Directory where the masks are
    mask_files_folder = os.path.join(
        data_path, "VOCdevkit", "VOC2012", "SegmentationClass"
    )

    # Iterate through the folders to populate the dataset dictionary with imgs and masks from train and validation
    dataset = {}
    for split in ["train", "val"]:
        print(f"Loading {split} dataset")

        dataset[split] = {"imgs": [], "masks": []}

        dataset_descriptor_file = open(
            os.path.join(descriptor_files_folder, split + ".txt")
        )
        for data_file_name in dataset_descriptor_file.read().splitlines():
            img = np.array(
                Image.open(os.path.join(img_files_folder, data_file_name + ".jpg"))
            )
            mask = np.array(
                Image.open(os.path.join(mask_files_folder, data_file_name + ".png"))
            )
            dataset[split]["imgs"].append(img)
            mask = mask[..., np.newaxis]  # (height, width) => (height, width, 1)
            dataset[split]["masks"].append(mask)

    print(f"Loading test dataset")

    # test dataset doesnt have masks
    dataset["test"] = {
        "imgs": [],
    }

    dataset_descriptor_file = open(os.path.join(descriptor_files_folder, "test.txt"))
    for data_file_name in dataset_descriptor_file.read().splitlines():
        img = np.array(
            Image.open(os.path.join(img_files_folder, data_file_name + ".jpg"))
        )
        dataset["test"]["imgs"].append(img)

    return (
        dataset["train"]["imgs"],
        dataset["train"]["masks"],
        dataset["val"]["imgs"],
        dataset["val"]["masks"],
        dataset["test"]["imgs"],
    )


def main(data_path, output_folder, limit_data, random_seed):

    info_dict = {"dataset": f"PascalVOC12"}

    # Download dataset and load train, val and test dataset
    train_imgs, train_masks, val_imgs, val_masks, test_imgs = load_pascalvoc12(
        data_path
    )

    if limit_data:
        train_limit_data = int(limit_data / 2)
        val_limit_data = int(limit_data / 2)

        train_data = list(zip(train_imgs, train_masks))
        val_data = list(zip(val_imgs, val_masks))

        train_data = random.sample(train_data, train_limit_data)
        val_data = random.sample(val_data, val_limit_data)

        train_imgs, train_masks = zip(*train_data)
        val_imgs, val_masks = zip(*val_data)

    # Calculate mean of training dataset (does not include validation!)
    train_img_mean, train_img_std = util.calculate_stats(train_imgs)

    output_path = os.path.join(data_path, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        raise OSError("Directory already exists!")

    # Save it as a numpy array
    np.savez_compressed(
        os.path.join(output_path, "pascalvoc12_train_mean"),
        train_img_mean=train_img_mean,
    )

    np.savez_compressed(
        os.path.join(output_path, "pascalvoc12_train_std"), train_img_std=train_img_std
    )

    # Convert to tf.train. Example and write the to TFRecords
    output_file = os.path.join(output_path, "train_1.tfrecords")
    util.convert_to_tfrecords(train_imgs, train_masks, output_file)
    output_file = os.path.join(output_path, "valid_1.tfrecords")
    util.convert_to_tfrecords(val_imgs, val_masks, output_file)
    output_file = os.path.join(output_path, "test_1.tfrecords")
    util.convert_to_tfrecords(test_imgs, None, output_file)

    info_dict["train_records"] = len(train_imgs)
    info_dict["valid_records"] = len(val_imgs)
    info_dict["test_records"] = len(test_imgs)

    util.create_info_file(out_path=output_path, info_dict=info_dict)

    shutil.rmtree(os.path.join(data_path, "VOCdevkit"), ignore_errors=True)

    print("Done! =)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Directory where PASCAL VOC 2012 is, or where to download it to.",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="pascalvoc12_tfr",
        help="Name of the folder that will contain the tfrecords files; it is "
        "saved inside *data_path*.",
    )
    parser.add_argument(
        "--limit_data",
        type=int,
        default=0,
        help="If zero, all training data is used to generate train and "
        "validation datasets. Otherwise, the train and validation "
        "sets will be generated from a subset of *limit_data* examples.",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed to be used. It affects the train/validation splitting"
        " and the data limitation example selection. If None, the random "
        "seed will be the current time.",
    )

    args = parser.parse_args()
    main(**vars(args))
