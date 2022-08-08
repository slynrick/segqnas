from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from cnn.cells import DownscalingCell, NonscalingCell, UpscalingCell
from cnn.loss import gen_dice_coef_loss
from cnn.metric import gen_dice_coef


def build_net(input_shape, num_classes, fn_dict, net_list, is_train=True):

    layer_dict = {}
    for name, definition in fn_dict.items():
        if definition["function"] in ["Conv2xBlock"]:
            definition["params"]["mu"] = mu
            definition["params"]["epsilon"] = epsilon
        layer_dict[name] = globals()[definition["function"]](**definition["params"])

    print(layer_dict)
    print(net_list)
    raise Exception()

    inputs = Input(input_shape, name="input")    
    x = inputs

    filters = [32, 64, 128, 256, 512, 256, 128, 64, 32]
    kernel_size = 3
    block = 'VGGBlock'

    x = DownscalingCell(block, kernel_size, filters[0])(x)
    skip1 = x

    x = DownscalingCell(block, kernel_size, filters[1])(x)
    skip2 = x

    x= DownscalingCell(block, kernel_size, filters[2])(x)
    skip3 = x

    x = DownscalingCell(block, kernel_size, filters[3])(x)
    skip4 = x

    x = NonscalingCell(block, kernel_size, filters[4])(x)
    
    x = UpscalingCell(block, kernel_size, filters[5])([x, skip4])
    
    x = UpscalingCell(block, kernel_size, filters[6])([x, skip3])
    
    x = UpscalingCell(block, kernel_size, filters[7])([x, skip2])
    
    x = UpscalingCell(block, kernel_size, filters[8])([x, skip1])
    
    # final conv
    prediction = Conv2D(name="prediction_mask",
                                filters=num_classes, kernel_size=(1, 1),
                                activation="sigmoid")(x)
    
    model = Model(inputs=[inputs], outputs=[
                               prediction], name="net")

    model.compile(optimizer=Adam(learning_rate=1e-4), loss=gen_dice_coef_loss, metrics=[gen_dice_coef])

    return model
