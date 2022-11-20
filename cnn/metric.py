from tensorflow.keras import backend as K


def soft_gen_dice_coef(y_true, y_pred, smooth=1e-7):
    """
    Dice coefficient for num_classes labels (classes). Ignores background pixel label 0
    Pass to model as metric during compile statement
    """
    num_classes = y_pred.shape[-1]
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, "int32"), num_classes=num_classes)[..., 1:])
    y_pred_f = K.flatten(y_pred[..., 1:])
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2.0 * intersect / (denom + smooth)))


def gen_dice_coef(y_true, y_pred, smooth=1e-7):
    num_classes = y_pred.shape[-1]
    y_true_f = K.flatten(K.one_hot(K.cast(y_true, "int32"), num_classes=num_classes)[..., 1:])
    y_pred_f = K.flatten(K.round(y_pred[..., 1:]))
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2.0 * intersect / (denom + smooth)))
