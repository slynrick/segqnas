import argparse

from util import download_file, extract_file


def main(data_path):

    pascal_voc_source = "http://pjreddie.com/media/files/"
    train_val_file_name = "VOCtrainval_11-May-2012.tar"
    test_file_name = "VOC2012test.tar"

    for file_name in [train_val_file_name, test_file_name]:
        source_url = pascal_voc_source + file_name
        file_path = download_file(data_path, file_name, source_url)
        extract_file(file_path, data_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="pascalvoc12",
        help="Directory where PASCAL VOC 2012 is, or where to download it to.",
    )

    args = parser.parse_args()
    main(**vars(args))
