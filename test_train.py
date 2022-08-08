import random

import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)

from cnn.input import Dataloader, SpleenDataset, get_train_val_filenames
from cnn.model import build_net

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
  except RuntimeError as e:
    print(e)

data_path = './data/Task09_Spleen_2D'

image_size = 128
batch_size = 32
num_classes = 2

train_images_filepaths, val_images_filepaths, train_labels_filepaths, val_labels_filepaths = get_train_val_filenames(data_path)

train_dataset = SpleenDataset(train_images_filepaths, train_labels_filepaths)
val_dataset = SpleenDataset(val_images_filepaths, val_labels_filepaths)

train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = Dataloader(val_dataset, batch_size=batch_size, shuffle=False)

model = build_net((image_size, image_size, 1), num_classes, fn_dict=None, net_list=None)

history = model.fit(
    train_dataloader,
    validation_data=val_dataloader,
    epochs=100,
    callbacks=[
        EarlyStopping(
            monitor="val_loss", mode="min", verbose=1, patience=3
        ),
        ModelCheckpoint(
            "best_model.h5",
            save_best_only=True,
            monitor="val_loss",
            mode="min",
        ),
        ReduceLROnPlateau(),
    ],
)
