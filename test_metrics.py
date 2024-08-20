import numpy as np
from keras import backend as K
import tensorflow as tf

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
    print(y_pred_f)
    intersect = K.sum(y_true_f * y_pred_f, axis=-1)
    denom = K.sum(y_true_f + y_pred_f, axis=-1)
    return K.mean((2.0 * intersect / (denom + smooth)))

def test_dice_coef_functions():
    # Sample data
    num_classes = 3  # Adjust as needed
    batch_size = 2
    height, width = 32, 32

    # Create random tensors
    y_true = np.random.randint(0, num_classes, size=(batch_size, height, width))
    y_pred = np.random.random((batch_size, height, width, num_classes))

    # Convert to tensors
    y_true = tf.convert_to_tensor(y_true, dtype=tf.int32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # Test gen_dice_coef_avg
    dice_avg = gen_dice_coef_avg(y_true, y_pred)
    assert isinstance(dice_avg, tf.Tensor)
    print("Dice average:", dice_avg)

    # Test gen_dice_coef
    dice_coeffs = list(gen_dice_coef(y_true, y_pred))
    assert len(dice_coeffs) == num_classes - 1  # Excluding background class
    for dice_coeff in dice_coeffs:
        assert isinstance(dice_coeff, tf.Tensor)
    print("Dice coefficients:", dice_coeffs)

    # Test gen_dice_coef_single_class
    for class_index in range(1, num_classes):
        dice_single = gen_dice_coef_single_class(y_true, y_pred, class_index)
        assert isinstance(dice_single, tf.Tensor)
        print("Dice for class", class_index, ":", dice_single)

if __name__ == "__main__":
    test_dice_coef_functions()