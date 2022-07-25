import segmentation_models as sm
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import (Callback, EarlyStopping,
                                        ModelCheckpoint, ReduceLROnPlateau,
                                        TensorBoard)

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)

sm.set_framework('tf.keras')

import os
import random

import matplotlib.pyplot as plt
import numpy as np

from cnn.input import (Dataloader, PascalVOC2012Dataset, denormalize,
                       get_preprocessing, get_training_augmentation,
                       get_validation_augmentation, visualize)

random.seed(0)

train_dataset_descriptor_filepath = os.path.join(
    'pascalvoc12',
    'VOCdevkit',
    'VOC2012',
    'ImageSets',
    'Segmentation',
    'train.txt',
)

val_dataset_descriptor_filepath = os.path.join(
    'pascalvoc12',
    'VOCdevkit',
    'VOC2012',
    'ImageSets',
    'Segmentation',
    'val.txt',
)

images_path = os.path.join(
    'pascalvoc12', 
    'VOCdevkit', 
    'VOC2012', 
    'JPEGImages'
)

masks_path = os.path.join(
    'pascalvoc12', 
    'VOCdevkit', 
    'VOC2012', 
    'SegmentationClass'
) 

image_height = 384
image_width = 384
backbone = 'resnet50'
batch_size = 16
num_classes = 21
num_channels = 3

train_dataset = PascalVOC2012Dataset(
    train_dataset_descriptor_filepath,
    images_path=images_path,
    masks_path=masks_path,
    image_height=image_height,
    image_width=image_width,
    #augmentation=get_training_augmentation(image_height, image_width),
    augmentation=get_validation_augmentation(image_height, image_width),
    preprocessing=get_preprocessing(sm.get_preprocessing(backbone)),
)

val_dataset = PascalVOC2012Dataset(
    val_dataset_descriptor_filepath,
    images_path=images_path,
    masks_path=masks_path,
    image_height=image_height,
    image_width=image_width,
    augmentation=get_validation_augmentation(image_height, image_width),
    preprocessing=get_preprocessing(sm.get_preprocessing(backbone)),
)

train_dataloader = Dataloader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = Dataloader(val_dataset, batch_size=batch_size, shuffle=False)

# check shapes for errors
assert train_dataloader[0][0].shape == (batch_size, image_height, image_width, num_channels)
assert train_dataloader[0][1].shape == (batch_size, image_height, image_width, num_classes)

# check shapes for errors
assert val_dataloader[0][0].shape == (batch_size, image_height, image_width, num_channels)
assert val_dataloader[0][1].shape == (batch_size, image_height, image_width, num_classes)

net = sm.PSPNet(backbone, classes=num_classes, activation='softmax', encoder_weights='imagenet', encoder_freeze=False)

net.summary()


class_indexes = list(range(1, num_classes))

dice_loss = sm.losses.DiceLoss(class_indexes=class_indexes) 
cce_loss = sm.losses.CategoricalCELoss()
total_loss = dice_loss + cce_loss

metrics = [
    tf.keras.metrics.OneHotIoU(num_classes=num_classes, target_class_ids=class_indexes, name='mean_iou_20'), 
    # dice_coef_20cat,
    # jaccard_coef,
    #sm.metrics.IOUScore(class_indexes=class_indexes),
    #sm.metrics.FScore(class_indexes=class_indexes),
    'accuracy'
]
sgd = tf.keras.optimizers.SGD(learning_rate=1e-3, momentum=0.9, nesterov=True)

# compile keras model with defined optimozer, loss and metrics
net.compile(sgd, total_loss, metrics)

class LrReducer(Callback):
    def __init__(self, base_lr = 0.01, max_epoch = 150, power=0.9, verbose=1):
        super(Callback, self).__init__()
        self.max_epoch = max_epoch
        self.power = power
        self.verbose = verbose
        self.base_lr = base_lr

    def on_epoch_end(self, epoch, logs={}):
        lr_now = K.get_value(self.model.optimizer.lr)
        new_lr = max(0.00001, min(self.base_lr * (1 - epoch / float(self.max_epoch))**self.power, lr_now))
        K.set_value(self.model.optimizer.lr, new_lr)
        if self.verbose:
            print(" - learning rate: %10f" % (new_lr))

model_checkpoint = ModelCheckpoint('./best_model.h5', monitor='val_loss', save_weights_only=True, save_best_only=True, mode='min')
tensorboard_callback = TensorBoard(log_dir='log', write_graph=True, write_images=True, histogram_freq=1)
plateau_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.99, verbose=1, patience=0, min_lr=0.00001)
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
lr_scheduler =  LrReducer()

callbacks = [
    model_checkpoint,
    tensorboard_callback,
    plateau_callback,
    lr_scheduler,
    early_stopping
]

history = net.fit(train_dataloader, 
                validation_data=val_dataloader,
                epochs=50,
                callbacks=callbacks
            )
