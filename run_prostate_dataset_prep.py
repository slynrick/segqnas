from multiprocessing import Pool

import numpy as np
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *
from monai.apps import download_and_extract

from spleen_dataset.config import (
    dataset_folder,
    num_threads,
    preprocessed_folder,
    root_folder,
)


def get_list_of_patients(base_dir):
    ct_directory = join(base_dir, "imagesTr")
    ct_files = subfiles(ct_directory, prefix="spleen", join=False)
    patients = list(map(get_patient_from_filename, ct_files))
    return patients


def get_patient_from_filename(filename):
    return (filename.split(".")[0]).split("_")[-1]


def get_list_of_files(base_dir):
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
    patients = get_list_of_patients(base_dir)

    for patient in patients:
        patient_filename = "spleen_" + patient + ".nii.gz"
        ct_file = join(ct_directory, patient_filename)
        segmentation_file = join(segmentation_directory, patient_filename)
        this_case = [ct_file, segmentation_file]
        list_of_lists.append(this_case)

    return list_of_lists


def load_and_preprocess(case, patient_name, output_folder):
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

    # get some metadata
    spacing = imgs_sitk[0].GetSpacing()
    # the spacing returned by SimpleITK is in inverse order relative to the numpy array we receive. If we wanted to
    # resample the data and if the spacing was not isotropic (in BraTS all cases have already been resampled to 1x1x1mm
    # by the organizers) then we need to pay attention here. Therefore we bring the spacing into the correct order so
    # that spacing[0] actually corresponds to the spacing of the first axis of the numpy array
    spacing = np.array(spacing)[::-1]

    direction = imgs_sitk[0].GetDirection()
    origin = imgs_sitk[0].GetOrigin()

    original_shape = imgs_npy[0].shape

    # now stack the images into one 4d array, cast to float because we will get rounding problems if we don't
    imgs_npy = np.concatenate([i[None] for i in imgs_npy]).astype(np.float32)

    # # now find the nonzero region and crop to that
    # nonzero = [np.array(np.where(i != 0)) for i in imgs_npy[1:]]
    # nonzero = [[np.min(i, 1), np.max(i, 1)] for i in nonzero]
    # nonzero = np.array(
    #     [np.min([i[0] for i in nonzero], 0), np.max([i[1] for i in nonzero], 0)]
    # ).T
    # # nonzero now has shape 3, 2. It contains the (min, max) coordinate of nonzero voxels for each axis

    # # now crop to nonzero
    # imgs_npy = imgs_npy[
    #     :,
    #     nonzero[0, 0] : nonzero[0, 1] + 1,
    #     :,#nonzero[1, 0] : nonzero[1, 1] + 1,
    #     :,#nonzero[2, 0] : nonzero[2, 1] + 1,
    # ]

    # # now we create a mask that we use for normalization
    # nonzero_masks = [i != 0 for i in imgs_npy[:-1]]
    # mask = np.zeros(imgs_npy.shape[1:], dtype=bool)
    # for i in range(len(nonzero_masks)):
    #     mask = mask | nonzero_masks[i]

    # # now normalize each modality with its mean and standard deviation (computed within the mask)
    # for i in range(len(imgs_npy) - 1):
    #     mean = imgs_npy[i][mask].mean()
    #     std = imgs_npy[i][mask].std()
    #     imgs_npy[i] = (imgs_npy[i] - mean) / (std + 1e-8)
    #     imgs_npy[i][mask == 0] = 0

    # Z normalisation
    # mean = imgs_npy[0].mean()
    # std = imgs_npy[0].std()
    # imgs_npy[0] = (imgs_npy[0] - mean) / (std + 1e-8)

    for slice_idx in range(imgs_npy.shape[1]):
        slice_npy = imgs_npy[:, slice_idx, :, :]

        slice_npy[0] = np.clip(slice_npy[0], -57, 164)
        slice_npy[0] = (slice_npy[0] - np.min(slice_npy[0])) / np.ptp(slice_npy[0])

        # mean = slice_npy[0].mean()
        # std = slice_npy[0].std()
        # slice_npy[0] = (slice_npy[0] - mean) / (std + 1e-8)

        np.save(
            join(output_folder, patient_name + "_" + str(slice_idx) + ".npy"), slice_npy
        )

    # now save as npz
    # np.save(join(output_folder, patient_name + ".npy"), imgs_npy)

    metadata = {
        "spacing": spacing,
        "direction": direction,
        "origin": origin,
        "original_shape": original_shape,
        # "nonzero_region": nonzero,
    }

    save_pickle(metadata, join(output_folder, patient_name + ".pkl"))


# def save_segmentation_as_nifti(segmentation, metadata, output_file):
#     original_shape = metadata["original_shape"]
#     seg_original_shape = np.zeros(original_shape, dtype=np.uint8)
#     nonzero = metadata["nonzero_region"]
#     seg_original_shape[
#         nonzero[0, 0] : nonzero[0, 1] + 1,
#         nonzero[1, 0] : nonzero[1, 1] + 1,
#         nonzero[2, 0] : nonzero[2, 1] + 1,
#     ] = segmentation
#     sitk_image = sitk.GetImageFromArray(seg_original_shape)
#     sitk_image.SetDirection(metadata["direction"])
#     sitk_image.SetOrigin(metadata["origin"])
#     # remember to revert spacing back to sitk order again
#     sitk_image.SetSpacing(tuple(metadata["spacing"][[2, 1, 0]]))
#     sitk.WriteImage(sitk_image, output_file)


def download_dataset():

    resource = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task05_Prostate.tar"
    md5 = "35138f08b1efaef89d7424d2bcc928db"

    compressed_file = os.path.join(root_folder, "Task05_Prostate.tar")

    if not os.path.exists(dataset_folder):
        download_and_extract(resource, compressed_file, root_folder, md5)


if __name__ == "__main__":

    download_dataset()

    # list_of_lists = get_list_of_files(dataset_folder)
    # list_of_patient_names = get_list_of_patients(dataset_folder)

    # maybe_mkdir_p(preprocessed_folder)

    # p = Pool(processes=num_threads)
    # p.starmap(
    #     load_and_preprocess,
    #     zip(
    #         list_of_lists,
    #         list_of_patient_names,
    #         [preprocessed_folder] * len(list_of_lists),
    #     ),
    # )
    # p.close()
    # p.join()
