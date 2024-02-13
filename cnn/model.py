from dataclasses import dataclass

import tensorflow as tf

from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

from cnn.layer import Layer
from cnn.loss import gen_dice_coef_loss
from cnn.metric import gen_dice_coef, soft_gen_dice_coef


def calculate_number_of_filters(depth, stem_filters):
    return stem_filters * (2**depth)


def get_skip_connection(previous_cell, feature_maps):
    if len(feature_maps) > 1:

        current_feature_map = feature_maps[-1]
        previous_feature_maps = feature_maps[:-1]

        if previous_cell == "UpscalingCell":
            for feature_map in previous_feature_maps[::-1]:
                if current_feature_map.shape.as_list() == feature_map.shape.as_list():
                    return feature_map

        elif previous_cell == "NonscalingCell":
            if (
                previous_feature_maps[-1].shape.as_list()
                == current_feature_map.shape.as_list()
            ):
                return previous_feature_maps[-1]

    return None


def fix_cell_for_feasibility(cell, depth, num_layers, layer_num, min_depth, max_depth):
    if num_layers - layer_num == depth:
        cell = "UpscalingCell"

    if num_layers - layer_num == depth + 1 and cell == "DownscalingCell":
        cell = "NonscalingCell"

    if depth == min_depth and cell == "UpscalingCell":
        cell = "NonscalingCell"

    if depth == max_depth and cell == "DownscalingCell":
        cell = "NonscalingCell"

    return cell


def calculate_number_of_filters(cell, filters):
    if cell == "DownscalingCell":
        filters *= 2
    elif cell == "UpscalingCell":
        filters /= 2
    return filters


def build_net_mirror(side, net_list, cell_list, layer_dict):
    feature_maps = []
    for layer_num, layer in enumerate(net_list):
        previous_cell = cell

        if cell_list:
            cell = cell_list[layer_num]
        else:
            cell = layer_dict[layer].get("cell")
        
        if side == 'decoder':
            # mirror logic
            if cell == "DownscalingCell":
                cell = "UpscalingCell"
        
        block = layer_dict[layer].get("block", None)
        kernel = layer_dict[layer].get("kernel", None)

        # cell = fix_cell_for_feasibility(
        #     cell, depth, num_layers, layer_num, min_depth, max_depth
        # )

        filters = calculate_number_of_filters(cell, filters)

        skip = get_skip_connection(
            previous_cell=previous_cell, feature_maps=feature_maps
        )

        if skip is not None:
            x = [x, skip]
        x = Layer(cell, block, kernel, filters)(
            x, name=f"Layer_{layer_num}_{cell}_{block}"
        )

        if cell == "DownscalingCell":
            depth += 1
        elif cell == "UpscalingCell":
            depth -= 1

        feature_maps.append(x)
    return feature_maps


def build_net(
    input_shape,
    stem_filters,
    max_depth,
    num_classes,
    layer_dict,
    net_list,
    cell_list=None,
    is_train=True,
):
    depth = 0
    min_depth = 0
    feature_maps = []
    num_layers = len(net_list)

    inputs = Input(input_shape, name="input")

    cell = "NonscalingCell"
    block = "StemConvolution"
    kernel = 3
    filters = stem_filters
    x = Layer(cell, block, kernel, filters)(inputs, name=f"{block}")

    # encoder 
    feature_maps += build_net_mirror('encoder', net_list, cell_list, layer_dict)
    # decoder
    feature_maps += build_net_mirror('decoder', net_list, cell_list, layer_dict)

    cell = "NonscalingCell"
    block = "OutputConvolution"
    kernel = num_classes
    filters = num_classes
    prediction_mask = Layer(cell, block, kernel, filters)(x, name=f"{block}")

    model = Model(inputs=[inputs], outputs=[prediction_mask], name="net")

    optimizer = optimizer = Adam()

    model.compile(
        optimizer=optimizer,
        loss=gen_dice_coef_loss,
        metrics=[gen_dice_coef, soft_gen_dice_coef],
    )

    return model
