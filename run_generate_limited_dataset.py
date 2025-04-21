import argparse
import datetime
import os
import random
import shutil

def main(input_path, output_path, num_images):
    if not os.path.exists(input_path):
        print(f"path {input_path} does not exist")
        exit()

    if not os.path.isdir(input_path):
        print(f"path {input_path} is not a directory")
        exit()
    
    if not os.path.exists(output_path):
        print(f"path {output_path} does not exist")
        os.mkdir(output_path)
        print(f"path {output_path} created")

    if not os.path.isdir(output_path):
        print(f"path {output_path} is not a directory")
        exit()

    train_size = int(num_images * 0.8)
    test_size = num_images - train_size

    for subfolder, size in [("train", train_size), ("test", test_size)]:
        input_subfolder = os.path.join(input_path, subfolder)
        output_subfolder = os.path.join(output_path, subfolder)

        if not os.path.exists(output_subfolder):
            os.mkdir(output_subfolder)
            print(f"path {output_subfolder} created")

        npy_files = [f for f in os.listdir(input_subfolder) if f.endswith(".npy")]
        random.seed(datetime.datetime.now().timestamp())
        random.shuffle(npy_files)

        for i, file in enumerate(npy_files[:size]):
            print(f"select file {file} on folder {subfolder} - {i}/{size}")
            shutil.copy(os.path.join(input_subfolder, file), os.path.join(output_subfolder, file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Limit datasets')
    parser.add_argument('-i','--input_path', type=str, required=True, help="Select a dataset to be downloaded: 'prostate' or 'spleen'")
    parser.add_argument('-o','--output_path', type=str, required=True, help="Number os threads to download the dataset")
    parser.add_argument('-n','--num_images', type=int, required=True, help="Select a split mode: 'image' or 'patiente'")


    args = parser.parse_args()

    main(args.input_path, args.output_path, args.num_images)
