from typing import Any, List

import tensorflow as tf
from cnn.layer import Layer
from cnn.loss import gen_dice_coef_loss, gen_dice_coef_weight_avg_loss, gen_dice_coef_soft_loss
from cnn.metric import gen_dice_coef_avg, gen_dice_coef_weight_avg, soft_gen_dice_coef
from keras.layers import Input
from keras.models import Model
from keras.optimizers import Adam

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


def build_net_mirror(side, net_list, cell_list, layer_dict, cell, filters, x, last_feature_maps = []):
    feature_maps: List[Any] = last_feature_maps
    depth = 0
    for layer_num, layer in enumerate(net_list):
        real_layer_num = layer_num + len(last_feature_maps)
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

        filters = calculate_number_of_filters(cell, filters)

        skip = get_skip_connection(
            previous_cell=previous_cell, feature_maps=feature_maps
        )

        if skip is not None:
            x = [x, skip]
        x = Layer(cell, block, kernel, filters)(
            x, name=f"Layer_{real_layer_num}_{cell}_{block}"
        )

        if cell == "DownscalingCell":
            depth += 1
        elif cell == "UpscalingCell":
            depth -= 1
        feature_maps.append(x)
    return feature_maps, x, cell, filters, depth


def build_net(
    input_shape,
    stem_filters,
    max_depth,
    num_classes,
    layer_dict,
    net_list,
    cell_list=None,
    is_train=True,
    loss_class_weights=None
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
    # x = Layer(cell, block, kernel, filters)(inputs, name=f"{block}")
    x = inputs

    real_layer_num = 0
    
    # encoder 
    for layer_num, layer in enumerate(net_list):
        real_layer_num += 1
        previous_cell = cell

        if cell_list:
            cell = cell_list[layer_num]
        else:
            cell = layer_dict[layer].get("cell")
        
        block = layer_dict[layer].get("block", None)
        kernel = layer_dict[layer].get("kernel", None)

        filters = calculate_number_of_filters(cell, filters)

        skip = get_skip_connection(
            previous_cell=previous_cell, feature_maps=feature_maps
        )

        if skip is not None:
            x = [x, skip]
        x = Layer(cell, block, kernel, filters)(
            x, name=f"Layer_{real_layer_num}_{cell}_{block}"
        )

        if cell == "DownscalingCell":
            depth += 1
        elif cell == "UpscalingCell":
            depth -= 1
        feature_maps.append(x)
    
    #decoder
    for layer_num, layer in enumerate(net_list[::-1]):
        real_layer_num += 1
        previous_cell = cell

        if cell_list:
            cell = cell_list[layer_num]
        else:
            cell = layer_dict[layer].get("cell")
        
        if cell == "DownscalingCell":
            cell = "UpscalingCell"
        
        block = layer_dict[layer].get("block", None)
        kernel = layer_dict[layer].get("kernel", None)

        filters = calculate_number_of_filters(cell, filters)

        skip = get_skip_connection(
            previous_cell=previous_cell, feature_maps=feature_maps
        )

        if skip is not None:
            x = [x, skip]
        x = Layer(cell, block, kernel, filters)(
            x, name=f"Layer_{real_layer_num}_{cell}_{block}"
        )

        if cell == "DownscalingCell":
            depth += 1
        elif cell == "UpscalingCell":
            depth -= 1
        feature_maps.append(x)

    cell = "NonscalingCell"
    block = "OutputConvolution"
    kernel = num_classes
    filters = num_classes
    prediction_mask = Layer(cell, block, kernel, filters)(x, name=f"{block}")

    model = Model(inputs=[inputs], outputs=[prediction_mask], name="net")

    optimizer = Adam()

    if loss_class_weights is not None:
        def custom_gen_dice_coef_weight_avg_loss(y_true, y_pred):
            return gen_dice_coef_weight_avg_loss(y_true, y_pred, loss_class_weights)
        
        def custom_gen_dice_coef_weight_avg(y_true, y_pred):
            return gen_dice_coef_weight_avg(y_true, y_pred, loss_class_weights)

        model.compile(
            optimizer=optimizer,
            loss=custom_gen_dice_coef_weight_avg_loss,
            metrics=[custom_gen_dice_coef_weight_avg],
        )
    else:
        def custom_gen_dice_coef_loss(y_true, y_pred):
            return gen_dice_coef_loss(y_true, y_pred)
        
        def custom_gen_dice_coef(y_true, y_pred):
            return gen_dice_coef_avg(y_true, y_pred)
    
        model.compile(
            optimizer=optimizer,
            loss=custom_gen_dice_coef_loss,
            metrics=[custom_gen_dice_coef],
        )
    
    #print(model.summary())

    return model
