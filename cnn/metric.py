from keras import backend as K
import tensorflow as tf

def gen_dice_coef_weight_avg(y_true, y_pred, loss_class_weights, smooth=1e-7):
    """
    Weighted average of Dice coefficients for each class. Ignores background pixel label 0
    Pass to model as metric during compile statement
    """
    num_classes = y_pred.shape[-1]
    dices = []
    weights = K._to_tensor(loss_class_weights[1:], dtype='float32')  # Ignore background class
    for idx in range(1, num_classes):
        dice = gen_dice_coef_single_class(y_true, y_pred, idx, smooth)
        dices.append(dice)
    dices = K._to_tensor(dices, dtype='float32')
    weighted_dices = dices * weights
    return K.sum(weighted_dices)

def gen_dice_coef_avg(y_true, y_pred, smooth=1e-7):
    """
    Dice average per class. Ignores background pixel label 0
    Pass to model as metric during compile statement
    """
    return K.mean(gen_dice_coef(y_true, y_pred, smooth))

def gen_dice_coef(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient for each class. Ignores background pixel label 0
    Pass to model as metric during compile statement
    """
    num_classes = y_pred.shape[-1]
    dices = []
    for idx in list(range(num_classes))[1:]:
        dices.append(gen_dice_coef_single_class(y_true, y_pred, idx, smooth))
    return K._to_tensor(dices, dtype='float32')

def gen_dice_coef_single_class(y_true, y_pred, class_index, smooth=1e-7):
    """
    Dice coefficient for num_classes labels (classes). Ignores background pixel label 0
    Pass to model as metric during compile statement
    """
    num_classes = y_pred.shape[-1]
    y_true_f = K.flatten(
        K.one_hot(K.cast(y_true, "int32"), num_classes=num_classes)[..., class_index]
    )
    y_pred_f = K.flatten(y_pred[..., class_index])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2.0 * intersect / (denom + smooth)))

def soft_gen_dice_coef(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient for num_classes labels (classes). Ignores background pixel label 0
    Pass to model as metric during compile statement
    """
    num_classes = y_pred.shape[-1]
    y_true_f = K.flatten(
        K.one_hot(K.cast(y_true, "int32"), num_classes=num_classes)[..., 1:]
    )
    y_pred_f = K.flatten(y_pred[..., 1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2.0 * intersect / (denom + smooth)))
