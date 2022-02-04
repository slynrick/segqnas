""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Utility functions and classes.

    References:
    https://github.com/tensorflow/models/blob/r1.10.0/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.py
    https://github.com/tensorflow/models/blob/r1.10.0/official/resnet/cifar10_download_and_extract.py

"""

import logging
import os
import pickle as pkl
import re
import tarfile
import sys
from shutil import rmtree
from run_dataset_prep import VALID_DATA_RATIO
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import yaml
from six.moves import urllib


class ExtractData(object):
    """ Class to extract data from an events.out.tfevents file. Uses an EventMultiplexer to
        access this data.
    """

    def __init__(self, input_dir, output_dir, run_tag_dict=None):
        """ Initialize ExtractData.

        Args:
            input_dir: (str) path to the directory containing the Tensorflow training files.
            output_dir: (str) path to the directory where to save the csv files.
            run_tag_dict: dict containing the runs from which data will be extracted. Example:
                {'run_dir_name1': ['tag1', 'tag2'], 'run_dir_name2': ['tag2']}. Default is to
                get *train_loss* from the *input_dir* and *accuracy* from the eval folder in
                *input_dir*.
        """

        self.dir_path = input_dir
        self.event_files = {}
        if run_tag_dict:
            self.run_tag_dict = run_tag_dict
        else:
            self.run_tag_dict = {'': ['train_loss'], 'eval': ['accuracy']}

        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_event_files(self):
        """ List event files in *self.input_dir*. """

        for run, tags in self.run_tag_dict.items():
            run_dir = os.path.join(self.dir_path, run)
            files = [os.path.join(run_dir, f) for f in os.listdir(run_dir)
                     if f.startswith('events')]
            # If more than one file, get the most recent one.
            if len(files) > 1:
                files.sort(key=lambda x: os.path.getmtime(x))
                files = files[-1:]
            self.event_files[run] = {'file': files[0], 'tags': tags}

    def export_to_csv(self, event_file, tag, write_headers=True):
        """ Extract tabular data from the scalars at a given run and tag. The result is a
            list of 3-tuples (wall_time, step, value).

        Args:
            event_file: (str) path to an event file.
            tag: (str) name of the tensor to be extracted from *event_file* (ex.: 'train_loss').
            write_headers: (bool) True if csv file should contain headers.
        """

        out_file = f'{os.path.split(self.dir_path)[-1]}_{tag}.csv'
        out_path = os.path.join(self.output_dir, out_file)
        iterator = tf.compat.v1.train.summary_iterator(event_file)

        with open(out_path, 'w') as f:
            if write_headers:
                print(f'wall_time,step,value', file=f)
            for e in iterator:
                for v in e.summary.value:
                    if v.tag == tag:
                        print(f'{e.wall_time},{e.step},{v.simple_value}', file=f)

    def extract(self):
        """ Extract data from each run and tag in *self.run_tag_dict* and export it to a csv
            file.
        """

        self.get_event_files()

        for run, v in self.event_files.items():
            for tag_name in v['tags']:
                self.export_to_csv(v['file'], tag_name)


def natural_key(string):
    """ Key to use with sort() in order to sort string lists in natural order.
        Example: [1_1, 1_2, 1_5, 1_10, 1_13].
    """

    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string)]


def delete_old_dirs(path, keep_best=False, best_id=''):
    """ Delete directories with old training files (models, checkpoints...). Assumes the
        directories' names start with digits.

    Args:
        path: (str) path to the experiment folder.
        keep_best: (bool) True if user wants to keep files from the best individual.
        best_id: (str) id of the best individual.
    """

    folders = [os.path.join(path, d) for d in os.listdir(path)
               if os.path.isdir(os.path.join(path, d)) and d[0].isdigit()]
    folders.sort(key=natural_key)

    if keep_best and best_id:
        folders = [d for d in folders if os.path.basename(d) != best_id]

    for f in folders:
        rmtree(f)


def check_files(exp_path):
    """ Check if exp_path exists and if it does, check if log_file is valid.

    Args:
        exp_path: (str) path to the experiment folder.
    """

    if not os.path.exists(exp_path):
        raise OSError('User must provide a valid \"--experiment_path\" to continue '
                      'evolution or to retrain a model.')

    file_path = os.path.join(exp_path, 'data_QNAS.pkl')

    if os.path.exists(file_path):
        if os.stat(file_path).st_size == 0:
            raise OSError('User must provide an \"--experiment_path\" with a valid data file to '
                          'continue evolution or to retrain a model.')
    else:
        raise OSError('log_file not found!')

    file_path = os.path.join(exp_path, 'log_params_evolution.txt')

    if os.path.exists(file_path):
        if os.stat(file_path).st_size == 0:
            raise OSError('User must provide an \"--experiment_path\" with a valid config_file '
                          'to continue evolution or to retrain a model.')
    else:
        raise OSError('log_params_evolution.txt not found!')


def init_log(log_level, name, file_path=None):
    """ Initialize a logging.Logger with level *log_level* and name *name*.

    Args:
        log_level: (str) one of 'NONE', 'INFO' or 'DEBUG'.
        name: (str) name of the module initiating the logger (will be the logger name).
        file_path: (str) path to the log file. If None, stdout is used.

    Returns:
        logging.Logger object.
    """

    logger = logging.getLogger(name)

    if file_path is None:
        handler = logging.StreamHandler()
    else:
        handler = logging.FileHandler(file_path)

    formatter = logging.Formatter('%(levelname)s: %(module)s: %(asctime)s.%(msecs)03d '
                                  '- %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_level == 'INFO':
        logger.setLevel(logging.INFO)
    elif log_level == 'DEBUG':
        logger.setLevel(logging.DEBUG)

    return logger


def load_yaml(file_path):
    """ Wrapper to load a yaml file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        dict with loaded parameters.
    """

    with open(file_path, 'r') as f:
        file = yaml.load(f)

    return file


def load_pkl(file_path):
    """ Load a pickle file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        loaded data.
    """

    with open(file_path, 'rb') as f:
        file = pkl.load(f)

    return file


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def calculate_stats(images):
    """ Calculate the image mean and std of the rescaled images array. Rescale the uint8 values to
        floats in the range [0, 1], and calculate the mean and std over all examples.

    Args:
        images: list of uint8 numpy arrays

    Returns:
        mean: np.ndarray of mean image with shape = [channels].
        std: np.ndarray of mean image with shape = [channels]
    """
    pixel_sum = np.zeros((1,3))
    pixel_square_sum = np.zeros((1,3))
    pixel_count = 0

    for image in images:

        rescaled_image = image / 255
        flattened_image = np.reshape(rescaled_image, (-1, image.shape[2]))

        pixel_sum += np.sum(flattened_image, axis=0)
        pixel_square_sum += np.sum(flattened_image ** 2, axis=0)

        pixel_count += image.shape[0] * image.shape[1]

    total_mean = pixel_sum / pixel_count
    total_var = (pixel_square_sum / pixel_count) - (total_mean ** 2)
    total_std = np.sqrt(total_var)

    return total_mean[0], total_std[0]


def convert_to_tfrecords(images, masks, output_file):
    """ Convert images and masks (numpy arrays) to tfrecords files.

    Args:
        images: list of uint8 numpy array of images (shape = [height, width, channels]).
        masks: list of uint8 numpy array of images (shape = [height, width]).
        output_file: (str) path to output file.
    """

    print(f'Generating {output_file}')

    with tf.compat.v1.python_io.TFRecordWriter(output_file) as record_writer:
        for i in range(len(images)):
            image = images[i]
            image_raw = image.flatten().tostring()
            if(masks):
                mask_raw = masks[i].flatten().tostring()
                example = tf.train.Example(features=tf.train.Features(
                    feature={'height': _int64_feature(image.shape[0]),
                            'width': _int64_feature(image.shape[1]),
                            'depth': _int64_feature(image.shape[2]),
                            'mask_raw': _bytes_feature(mask_raw),
                            'image_raw': _bytes_feature(image_raw)}))
            else:
                example = tf.train.Example(features=tf.train.Features(
                    feature={'height': _int64_feature(image.shape[0]),
                            'width': _int64_feature(image.shape[1]),
                            'depth': _int64_feature(image.shape[2]),
                            'image_raw': _bytes_feature(image_raw)}))

            record_writer.write(example.SerializeToString())

def download_pascalvoc12(data_path):   
    """ Download the train, validation and test datasets to the *data_path* if it is not already there.

    Args:
        data_path: (str) path to the directory where the dataset is or where to download it to.

    Returns:
        paths to the downloaded files.
    """
    pascalvoc12_train_val_url = "http://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar"
    pascalvoc12_test_url = "http://pjreddie.com/media/files/VOC2012test.tar"

    source_urls = [pascalvoc12_train_val_url, pascalvoc12_test_url]

    file_names = [source_url.split('/')[-1] for source_url in source_urls]

    file_paths = []

    for source_url, file_name in zip(source_urls, file_names):

        file_path = os.path.join(data_path, file_name)

        if not os.path.exists(file_path):

            def _progress(count, block_size, total_size):
                sys.stdout.write(
                    f'\r>> Downloading {file_name} {100.0 * count * block_size / total_size:.1f}%')
                sys.stdout.flush()

            if not os.path.exists(data_path):
                os.makedirs(data_path)

            print(f'Download from {source_url}.')
            file_path, _ = urllib.request.urlretrieve(source_url, file_path, _progress)
            stat_info = os.stat(file_path)
            print(f'\nSuccessfully downloaded {file_name} {stat_info.st_size} bytes! :)')
        else:
            print(f'Dataset from {source_url} already downloaded, skipping download...')

        file_paths.append(file_path)

    return file_paths


def create_info_file(out_path, info_dict):
    """ Saves info in *info_dict* in a txt file.

    Args:
        out_path: (str) path to the directory where to save info file.
        info_dict: dict with all relevant info the user wants to save in the info file.
    """

    with open(os.path.join(out_path, 'data_info.txt'), 'w') as f:
        yaml.dump(info_dict, f)