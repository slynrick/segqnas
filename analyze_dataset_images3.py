""" Copyright (c) 2020, Daniela Szwarcman and IBM Research
    * Licensed under The MIT License [see LICENSE for details]

    - Retrain CNN model generated by Q-NAS.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from cnn.input import Dataloader, Dataset, get_validation_augmentation


def main(**args):

    patch_size = (args['image_size'], args['image_size'], args['num_channels'])
    val_augmentation = get_validation_augmentation(patch_size)
    
    val_dataset = Dataset(
        data_path=os.path.join(args['data_path'], 'train'),
        selected=None,
        only_non_empty_slices=True,
    )
    
    val_dataloader = Dataloader(
        dataset=val_dataset,
        augmentation=val_augmentation,
        shuffle=True,
    )

    num_images = 6
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))  # 2 rows, 3 columns
    i = 0
    class_colors = np.linspace(0.0, 1.0, 10) # Colors for each class
    for image, mask in val_dataloader:
        row = i // 3  # Calculate the row index
        col = i % 3   # Calculate the column index
        print(row, col)
        grayscale_channel = image[0][:, : , 0]
        
        axes[row, col].imshow(grayscale_channel, cmap='gray')
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Image {i+1}")

        masked_image = mask[0]
        unique_classes = np.unique(mask[0])
        print(unique_classes)
        for k, cls in enumerate(unique_classes):
            if cls == 0:
                continue
            mask_indices = np.nonzero(mask[0] == cls)
            masked_image[mask_indices] = np.array(class_colors[k]) * 255
            print(k, cls, class_colors[k]*255)
        
        axes[row, col].imshow(masked_image, alpha=0.3, cmap='tab10')
        axes[row, col].axis('off')
        axes[row, col].set_title(f"Image {i+1}")

        num_images -= 1
        i += 1

        if num_images == 0:
            break
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to experiment_path."
    )

    parser.add_argument(
        "--image_size", type=int, required=True, help="Size of the image."
    )

    parser.add_argument(
        "--num_channels", type=int, required=True, help="Channels of the image."
    )

    parser.add_argument(
        "--num_classes", type=int, required=True, help="Classes of the image."
    )

    arguments = parser.parse_args()

    main(**vars(arguments))