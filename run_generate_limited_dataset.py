import argparse
import os
import shutil

def main(input_path, output_path, num_image):
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

    print(len(os.listdir(input_path)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Limit datasets')
    parser.add_argument('-i','--input_path', type=str, required=True, help="Select a dataset to be downloaded: 'prostate' or 'spleen'")
    parser.add_argument('-o','--output_path', type=str, required=True, help="Number os threads to download the dataset")
    parser.add_argument('-n','--num_image', type=int, required=True, help="Select a split mode: 'image' or 'patiente'")


    args = parser.parse_args()

    main(args.input_path, args.output_path, args.num_image)
