""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Read CIFAR-10/100 data from pickled numpy arrays and writes into TFRecords files.

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py

"""

import argparse
import os
import pickle
import tarfile
from time import time
from PIL import Image
import numpy as np

import util

VALID_DATA_RATIO = 0.1
TRAIN_EX = 50000
TEST_EX = 10000
NUM_BINS = 5
NUM_CLASSES = 21 # 20 classes + 1 background

def load_pascalvoc12(data_path):
    """ Download PASCAL VOC 12 and load the images and labels for training and test sets.

    Args:
        data_path: (str) path to the directory where PASCAL VOC 12 is, or where to download it to.

    Returns:
        train_imgs, train_labels, test_imgs, test_labels.
    """
    pascalvoc12_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"

    # Check if dataset exists and download it if does not
    name = pascalvoc12_url.split('/')[-1]
    file_path = util.download_file(data_path, name, pascalvoc12_url)
    tarfile.open(file_path, 'r').extractall(data_path)

    # Relevant folders for the data
    # Directory where train.txt and val.txt files are. These files contain the name of the images in the training and validation datasets
    descriptor_files_folder = os.path.join(data_path, 'VOCdevkit', 'VOC2012', 'ImageSets', 'Segmentation')
    # Directory where the images are
    img_files_folder = os.path.join(data_path, 'VOCdevkit', 'VOC2012', 'JPEGImages')
    # Directory where the masks are
    mask_files_folder = os.path.join(data_path, 'VOCdevkit', 'VOC2012', 'SegmentationClass')

    dataset = {}
    for split in ['train', 'val']:

        dataset[split] = {
            'imgs': [],
            'masks': []
        }

        dataset_descriptor_file = open(os.path.join(descriptor_files_folder, split + '.txt'))
        for data_file_name in dataset_descriptor_file.read().splitlines():

            img = np.array(Image.open(os.path.join(img_files_folder, data_file_name + '.jpg')))
            mask = np.array(Image.open(os.path.join(mask_files_folder, data_file_name + '.png')))

            dataset[split]['imgs'] = img
            dataset[split]['masks'] = mask

    return dataset['train']['imgs'], dataset['train']['masks'], dataset['val']['imgs'], dataset['val']['masks']

def main(data_path, output_folder, limit_data, random_seed):
    
    info_dict = {'dataset': f'PascalVOC12'}

    train_imgs, train_masks, test_imgs, test_masks = load_pascalvoc12(data_path)

    # split train set into train and validation

    if limit_data:
        size = limit_data
    else:
        size = len(train_masks)

    if random_seed is None:
        random_seed = int(time())

    np.random.seed(random_seed)  # Choose random seed
    info_dict['seed'] = random_seed

    train_imgs, train_masks, valid_imgs, valid_masks = util.split_dataset(
        images=train_imgs, labels=train_masks, num_classes=NUM_CLASSES,
        valid_ratio=VALID_DATA_RATIO, limit=size)

    # Calculate mean of training dataset (does not include validation!)
    train_img_mean = util.calculate_mean(train_imgs)

    output_path = os.path.join(data_path, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        raise OSError('Directory already exists!')

    # Save it as a numpy array
    np.savez_compressed(os.path.join(output_path, 'pascalvoc12_train_mean'),
                        train_img_mean=train_img_mean)


def main2(data_path, output_folder, limit_data, num_classes, random_seed, label_mode):

    info_dict = {'dataset': f'CIFAR{num_classes}'}

    train_imgs, train_labels, test_imgs, test_labels = load_cifar10(data_path)

    # Split into train and validation ##########################################################

    if limit_data:
        size = limit_data
    else:
        size = len(train_labels)

    if random_seed is None:
        random_seed = int(time())

    np.random.seed(random_seed)  # Choose random seed
    info_dict['seed'] = random_seed

    train_imgs, train_labels, valid_imgs, valid_labels = util.split_dataset(
        images=train_imgs, labels=train_labels, num_classes=num_classes,
        valid_ratio=VALID_DATA_RATIO, limit=size)

    # Calculate mean of training dataset (does not include validation!)
    train_img_mean = util.calculate_mean(train_imgs)

    output_path = os.path.join(data_path, output_folder)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        raise OSError('Directory already exists!')

    # Save it as a numpy array
    np.savez_compressed(os.path.join(output_path, 'cifar_train_mean'),
                        train_img_mean=train_img_mean)

    # Convert to tf.train.Example and write the to TFRecords ###################################
    output_file = os.path.join(output_path, 'train_1.tfrecords')
    util.convert_to_tfrecords(train_imgs, train_labels, output_file)
    output_file = os.path.join(output_path, 'valid_1.tfrecords')
    util.convert_to_tfrecords(valid_imgs, valid_labels, output_file)
    output_file = os.path.join(output_path, 'test_1.tfrecords')
    util.convert_to_tfrecords(test_imgs, test_labels, output_file)

    info_dict['train_records'] = len(train_labels)
    info_dict['valid_records'] = len(valid_labels)
    info_dict['test_records'] = len(test_labels)
    info_dict['shape'] = list(train_imgs.shape[1:])

    util.create_info_file(out_path=output_path, info_dict=info_dict)

    print('Done! =)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                        help='Directory where CIFAR is, or where to download it to.')
    parser.add_argument('--output_folder', type=str, default='cifar_tfr',
                        help='Name of the folder that will contain the tfrecords files; it is '
                             'saved inside *data_path*.')
    parser.add_argument('--limit_data', type=int, default=0,
                        help='If zero, all training data is used to generate train and '
                             'validation datasets. Otherwise, the train and validation '
                             'sets will be generated from a subset of *limit_data* examples.')
    parser.add_argument('--random_seed', type=int, default=None,
                        help='Random seed to be used. It affects the train/validation splitting'
                             ' and the data limitation example selection. If None, the random '
                             'seed will be the current time.')

    args = parser.parse_args()
    main(**vars(args))
