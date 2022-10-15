from dataclasses import dataclass

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay

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

    for layer_num, layer in enumerate(net_list):
        previous_cell = cell

        if cell_list:
            cell = cell_list[layer_num]
        else:
            cell = layer_dict[layer].get("cell")

        block = layer_dict[layer].get("block", None)
        kernel = layer_dict[layer].get("kernel", None)

        cell = fix_cell_for_feasibility(
            cell, depth, num_layers, layer_num, min_depth, max_depth
        )

        # if cell == "DownscalingCell":
        #     filters *= 2
        # elif cell == "UpscalingCell":
        #     filters /= 2

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

    cell = "NonscalingCell"
    block = "OutputConvolution"
    kernel = 1
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


# def build_net(
#     input_shape,
#     stem_filters,
#     max_depth,
#     num_classes,
#     layer_dict,
#     net_list,
#     cell_list=None,
#     is_train=True,
# ):
#     depth = 0
#     min_depth = 0
#     previous_feature_maps = []
#     num_layers = len(net_list)

#     inputs = Input(input_shape, name="input")

#     x = Layer("NonscalingCell", "StemConvolution", 3, stem_filters)(
#         inputs, name=f"Stem"
#     )

#     for layer_num, layer in enumerate(net_list):

#         if cell_list:
#             cell = cell_list[layer_num]
#         else:
#             cell = layer_dict[layer].get("cell", None)

#         block = layer_dict[layer].get("block", None)
#         kernel = layer_dict[layer].get("kernel", None)

#         cell = fix_cell_for_feasibility(
#             cell, depth, num_layers, layer_num, min_depth, max_depth
#         )

#         skip = get_skip_connection(cells_info=cells_info, previous_feature_maps=previous_feature_maps)

#         if cell == "DownscalingCell":
#             depth += 1
#         elif cell == "UpscalingCell":
#             depth -= 1

#         cells_info.append(CellInfo(cell_type=cell, depth=depth, layer_num=layer_num))

#         filters = calculate_number_of_filters(depth=depth, stem_filters=stem_filters)

#         if skip is not None:
#             x = [x, skip]

#         x = Layer(cell, block, kernel, filters)(
#             x, name=f"Layer_{layer_num}_{cell}_{block}"
#         )

#         print(depth, x.shape.as_list())

#         previous_feature_maps.append(x)


#     prediction_mask = Layer("NonscalingCell", "OutputConvolution", 1, num_classes)(
#         x, name=f"Final"
#     )

#     model = Model(inputs=[inputs], outputs=[prediction_mask], name="net")

#     optimizer = optimizer = Adam()

#     model.compile(
#         optimizer=optimizer,
#         loss=gen_dice_coef_loss,
#         metrics=[gen_dice_coef, soft_gen_dice_coef],
#     )

#     return model
