import json
import logging
import os
import pickle as pkl
import re
import sys
import tarfile
from shutil import rmtree

import nibabel
import numpy as np
import tensorflow as tf
import yaml
from six.moves import urllib
from skimage.transform import resize


def listdir_nohidden(path):
    files = []
    for f in os.listdir(path):
        if not f.startswith("."):
            files.append(f)
    return files


def prepare_spleen_data(data_path, target_path, image_size):

    images_path = os.path.join(data_path, "imagesTr")
    labels_path = os.path.join(data_path, "labelsTr")
    target_images_path = os.path.join(target_path, "imagesTr")
    target_labels_path = os.path.join(target_path, "labelsTr")

    if not os.path.exists(images_path):
        os.makedirs(images_path)

    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    if not os.path.exists(target_images_path):
        os.makedirs(target_images_path)

    if not os.path.exists(target_labels_path):
        os.makedirs(target_labels_path)

    images_filenames = listdir_nohidden(images_path)
    labels_filenames = listdir_nohidden(labels_path)

    images_filenames.sort()
    labels_filenames.sort()

    descriptor_dict = {}

    for image_filename, label_filename in zip(images_filenames, labels_filenames):

        image = nibabel.load(os.path.join(images_path, image_filename))
        label = nibabel.load(os.path.join(labels_path, label_filename))
        patient = image_filename.split("_")[1].split(".")[0]
        descriptor_dict[f"{patient}"] = []

        for slice_num in range(label.shape[2]):
            slice_label = label.get_fdata()[:, :, slice_num]
            slice_label = resize(
                slice_label, (image_size, image_size), preserve_range=True
            )
            slice_image = image.get_fdata()[:, :, slice_num]
            slice_image = resize(
                slice_image, (image_size, image_size), preserve_range=True
            )

            if len(np.unique(slice_label)) != 1:
                slice_image_filename = os.path.join(
                    target_images_path,
                    f"{image_filename.split('.')[0]}_{slice_num}.npy",
                )
                slice_label_filename = os.path.join(
                    target_labels_path,
                    f"{label_filename.split('.')[0]}_{slice_num}.npy",
                )

                np.save(slice_image_filename, slice_image)
                np.save(slice_label_filename, slice_label)
                descriptor_dict[f"{patient}"].append(
                    (slice_image_filename, slice_label_filename)
                )

    with open(os.path.join(target_path, "dataset.json"), "w") as fp:
        json.dump(descriptor_dict, fp)


class ExtractData(object):
    """Class to extract data from an events.out.tfevents file. Uses an EventMultiplexer to
    access this data.
    """

    def __init__(self, input_dir, output_dir, run_tag_dict=None):
        """Initialize ExtractData.

        Args:
            input_dir: (str) path to the directory containing the Tensorflow training files.
            output_dir: (str) path to the directory where to save the csv files.
            run_tag_dict: dict containing the runs from which data will be extracted. Example:
                {'run_dir_name1': ['tag1', 'tag2'], 'run_dir_name2': ['tag2']}. Default is to
                get *train_loss* from the *input_dir* and *mean iou* from the eval folder in
                *input_dir*.
        """

        self.dir_path = input_dir
        self.event_files = {}
        if run_tag_dict:
            self.run_tag_dict = run_tag_dict
        else:
            self.run_tag_dict = {"": ["train_loss"], "eval": ["mean_iou"]}

        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def get_event_files(self):
        """List event files in *self.input_dir*."""

        for run, tags in self.run_tag_dict.items():
            run_dir = os.path.join(self.dir_path, run)
            files = [
                os.path.join(run_dir, f)
                for f in os.listdir(run_dir)
                if f.startswith("events")
            ]
            # If more than one file, get the most recent one.
            if len(files) > 1:
                files.sort(key=lambda x: os.path.getmtime(x))
                files = files[-1:]
            self.event_files[run] = {"file": files[0], "tags": tags}

    def export_to_csv(self, event_file, tag, write_headers=True):
        """Extract tabular data from the scalars at a given run and tag. The result is a
            list of 3-tuples (wall_time, step, value).

        Args:
            event_file: (str) path to an event file.
            tag: (str) name of the tensor to be extracted from *event_file* (ex.: 'train_loss').
            write_headers: (bool) True if csv file should contain headers.
        """

        out_file = f"{os.path.split(self.dir_path)[-1]}_{tag}.csv"
        out_path = os.path.join(self.output_dir, out_file)
        iterator = tf.compat.v1.train.summary_iterator(event_file)

        with open(out_path, "w") as f:
            if write_headers:
                print(f"wall_time,step,value", file=f)
            for e in iterator:
                for v in e.summary.value:
                    if v.tag == tag:
                        print(f"{e.wall_time},{e.step},{v.simple_value}", file=f)

    def extract(self):
        """Extract data from each run and tag in *self.run_tag_dict* and export it to a csv
        file.
        """

        self.get_event_files()

        for run, v in self.event_files.items():
            for tag_name in v["tags"]:
                self.export_to_csv(v["file"], tag_name)


def natural_key(string):
    """Key to use with sort() in order to sort string lists in natural order.
    Example: [1_1, 1_2, 1_5, 1_10, 1_13].
    """

    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string)]


def delete_old_dirs(path, keep_best=False, best_id=""):
    """Delete directories with old training files (models, checkpoints...). Assumes the
        directories' names start with digits.

    Args:
        path: (str) path to the experiment folder.
        keep_best: (bool) True if user wants to keep files from the best individual.
        best_id: (str) id of the best individual.
    """

    folders = [
        os.path.join(path, d)
        for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d)) and d[0].isdigit()
    ]
    folders.sort(key=natural_key)

    if keep_best and best_id:
        folders = [d for d in folders if os.path.basename(d) != best_id]

    for f in folders:
        rmtree(f)


def check_files(exp_path):
    """Check if exp_path exists and if it does, check if log_file is valid.

    Args:
        exp_path: (str) path to the experiment folder.
    """

    if not os.path.exists(exp_path):
        raise OSError(
            'User must provide a valid "--experiment_path" to continue '
            "evolution or to retrain a model."
        )

    file_path = os.path.join(exp_path, "net_list.pkl")

    if os.path.exists(file_path):
        if os.stat(file_path).st_size == 0:
            raise OSError(
                'User must provide an "--experiment_path" with a valid data file to '
                "continue evolution or to retrain a model."
            )
    else:
        raise OSError("log_file not found!")

    file_path = os.path.join(exp_path, "log_params_evolution.txt")

    if os.path.exists(file_path):
        if os.stat(file_path).st_size == 0:
            raise OSError(
                'User must provide an "--experiment_path" with a valid config_file '
                "to continue evolution or to retrain a model."
            )
    else:
        raise OSError("log_params_evolution.txt not found!")


def init_log(log_level, name, file_path=None):
    """Initialize a logging.Logger with level *log_level* and name *name*.

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

    formatter = logging.Formatter(
        "%(levelname)s: %(module)s: %(asctime)s.%(msecs)03d " "- %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    if log_level == "INFO":
        logger.setLevel(logging.INFO)
    elif log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)

    return logger


def load_yaml(file_path):
    """Wrapper to load a yaml file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        dict with loaded parameters.
    """

    with open(file_path, "r") as f:
        file = yaml.full_load(f)

    return file


def load_pkl(file_path):
    """Load a pickle file.

    Args:
        file_path: (str) path to the file to load.

    Returns:
        loaded data.
    """

    with open(file_path, "rb") as f:
        file = pkl.load(f)

    return file


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def calculate_stats(images):
    """Calculate the image mean and std of the rescaled images array. Rescale the uint8 values to
        floats in the range [0, 1], and calculate the mean and std over all examples.

    Args:
        images: list of uint8 numpy arrays

    Returns:
        mean: np.ndarray of mean image with shape = [channels].
        std: np.ndarray of mean image with shape = [channels]
    """
    pixel_sum = np.zeros((1, 3))
    pixel_square_sum = np.zeros((1, 3))
    pixel_count = 0

    for image in images:

        rescaled_image = image / 255
        flattened_image = np.reshape(rescaled_image, (-1, image.shape[2]))

        pixel_sum += np.sum(flattened_image, axis=0)
        pixel_square_sum += np.sum(flattened_image**2, axis=0)

        pixel_count += image.shape[0] * image.shape[1]

    total_mean = pixel_sum / pixel_count
    total_var = (pixel_square_sum / pixel_count) - (total_mean**2)
    total_std = np.sqrt(total_var)

    return total_mean[0], total_std[0]


def convert_to_tfrecords(images, masks, output_file):
    """Convert images and masks (numpy arrays) to tfrecords files.

    Args:
        images: list of uint8 numpy array of images (shape = [height, width, channels]).
        masks: list of uint8 numpy array of images (shape = [height, width, 1]).
        output_file: (str) path to output file.
    """

    print(f"Generating {output_file}")

    with tf.compat.v1.python_io.TFRecordWriter(output_file) as record_writer:
        for i in range(len(images)):
            image = images[i]
            image_raw = image.flatten().tostring()
            if masks:
                mask = masks[i]
                mask_raw = mask.flatten().tostring()
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "height": _int64_feature(image.shape[0]),
                            "width": _int64_feature(image.shape[1]),
                            "channels": _int64_feature(image.shape[2]),
                            "mask_raw": _bytes_feature(mask_raw),
                            "image_raw": _bytes_feature(image_raw),
                        }
                    )
                )
            else:
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            "height": _int64_feature(image.shape[0]),
                            "width": _int64_feature(image.shape[1]),
                            "channels": _int64_feature(image.shape[2]),
                            "image_raw": _bytes_feature(image_raw),
                        }
                    )
                )

            record_writer.write(example.SerializeToString())


def download_file(data_path, file_name, source_url):
    """Download *file_name* in *source_url* if it is not in *data_path*.

    Args:
        data_path: (str) path to the directory where the dataset is or where to download it to.
        file_name: (str) name of the file to download.
        source_url: (str) URL source from where the file should be downloaded.

    Returns:
        path to the downloaded file.
    """

    file_path = os.path.join(data_path, file_name)

    if not os.path.exists(file_path):

        def _progress(count, block_size, total_size):
            sys.stdout.write(
                f"\r>> Downloading {file_name} {100.0 * count * block_size / total_size:.1f}%"
            )
            sys.stdout.flush()

        if not os.path.exists(data_path):
            os.makedirs(data_path)

        print(f"Download from {source_url}.")
        file_path, _ = urllib.request.urlretrieve(source_url, file_path, _progress)
        stat_info = os.stat(file_path)
        print(f"\nSuccessfully downloaded {file_name} {stat_info.st_size} bytes! :)")
    else:
        print(f"Dataset already downloaded, skipping download...")

    return file_path


def extract_file(compressed_file_path, output_data_path):
    """Extract the file in *compressed_file_path* in the *output_data_path*

    Args:
        output_data_path: (str) path to the file to be extracted.
        compressed_file_path: (str) path where the file will be extracted.
    """
    print(f"Extracting {compressed_file_path}")
    tarfile.open(compressed_file_path, "r").extractall(output_data_path)


def create_info_file(out_path, info_dict):
    """Saves info in *info_dict* in a txt file.

    Args:
        out_path: (str) path to the directory where to save info file.
        info_dict: dict with all relevant info the user wants to save in the info file.
    """

    with open(os.path.join(out_path, "data_info.txt"), "w") as f:
        yaml.dump(info_dict, f)
