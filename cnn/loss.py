from cnn.metric import gen_dice_coef


def gen_dice_coef_loss(y_true, y_pred):
    '''
    Dice loss to minimize. Pass to model as loss during compile statement
    '''
    return 1 - gen_dice_coef(y_true, y_pred)
