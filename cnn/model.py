import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay

from cnn.layer import Layer
from cnn.loss import gen_dice_coef_loss
from cnn.metric import gen_dice_coef, soft_gen_dice_coef

STEM_FILTERS = 32


def calculate_number_of_filters(depth):
    return STEM_FILTERS * (2**depth)


def get_skip_connection(previous_feature_maps, current_feature_map):
    for feature_map in previous_feature_maps[::-1]:
        if current_feature_map.shape.as_list() == feature_map.shape.as_list():
            return feature_map

    return None

def clip_cell_min_max_depth(cell, depth, min_depth=0, max_depth=4):
    if depth < min_depth:
        depth = min_depth
        cell = "NonscalingCell"

    if depth > max_depth:
        depth = max_depth
        cell = "NonscalingCell"

    return cell


def build_net(input_shape, num_classes, fn_dict, layer_list, is_train=True):
    depth = 0
    min_depth = 0
    max_depth = 4
    previous_feature_maps = []
    num_layers = len(layer_list)

    inputs = Input(input_shape, name="input")

    x = Layer("NonscalingCell", "StemConvolution", 3, STEM_FILTERS)(
        inputs, name=f"StemConvolution"
    )

    for layer_num, layer in enumerate(layer_list):

        cell = fn_dict[layer].get("cell", None)
        block = fn_dict[layer].get("block", None)
        kernel = fn_dict[layer].get("kernel", None)

        if num_layers - layer_num <= depth:
            cell = "UpscalingCell"

        if cell == "DownscalingCell":
            depth += 1
        elif cell == "UpscalingCell":
            depth -= 1

        cell = clip_cell_min_max_depth(cell, depth, min_depth, max_depth)

        filters = calculate_number_of_filters(depth)

        skip = get_skip_connection(previous_feature_maps[:-1], x)

        if skip is not None:
            x = [x, skip]

        x = Layer(cell, block, kernel, filters)(
            x, name=f"Layer_{layer_num}_{cell}_{block}"
        )

        previous_feature_maps.append(x)

    prediction_mask = Layer("NonscalingCell", "OutputConvolution", 3, num_classes)(
        x, name=f"OutputConvolution"
    )

    model = Model(inputs=[inputs], outputs=[prediction_mask], name="net")

    optimizer = optimizer = Adam()

    model.compile(
        optimizer=optimizer,
        loss=gen_dice_coef_loss,
        metrics=[gen_dice_coef, soft_gen_dice_coef],
    )

    return model
