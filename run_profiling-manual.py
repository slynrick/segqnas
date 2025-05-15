import tensorflow as tf
from cnn.model import build_net
from cnn.train_detailed import get_model

# Layer dictionary remains the same
layer_dict = {
        'den_d_3': {'cell': 'DownscalingCell', 'block': 'DenseBlock',               'kernel': 3, 'prob':1/6/8},
        'den_d_5': {'cell': 'DownscalingCell', 'block': 'DenseBlock',               'kernel': 5, 'prob':1/6/8},
        'den_d_7': {'cell': 'DownscalingCell', 'block': 'DenseBlock',               'kernel': 7, 'prob':1/6/8},
        'inc_d_3': {'cell': 'DownscalingCell', 'block': 'InceptionBlock',           'kernel': 3, 'prob':1/6/8},
        'inc_d_5': {'cell': 'DownscalingCell', 'block': 'InceptionBlock',           'kernel': 5, 'prob':1/6/8},
        'inc_d_7': {'cell': 'DownscalingCell', 'block': 'InceptionBlock',           'kernel': 7, 'prob':1/6/8},
        'res_d_3': {'cell': 'DownscalingCell', 'block': 'ResNetBlock',              'kernel': 3, 'prob':1/6/8},
        'res_d_5': {'cell': 'DownscalingCell', 'block': 'ResNetBlock',              'kernel': 5, 'prob':1/6/8},
        'res_d_7': {'cell': 'DownscalingCell', 'block': 'ResNetBlock',              'kernel': 7, 'prob':1/6/8},
        'vgg_d_3': {'cell': 'DownscalingCell', 'block': 'VGGBlock',                 'kernel': 3, 'prob':1/6/8},
        'vgg_d_5': {'cell': 'DownscalingCell', 'block': 'VGGBlock',                 'kernel': 5, 'prob':1/6/8},
        'vgg_d_7': {'cell': 'DownscalingCell', 'block': 'VGGBlock',                 'kernel': 7, 'prob':1/6/8},

        'den_n_3': {'cell': 'NonscalingCell',  'block': 'DenseBlock',               'kernel': 3, 'prob':1/6/8},
        'den_n_5': {'cell': 'NonscalingCell',  'block': 'DenseBlock',               'kernel': 5, 'prob':1/6/8},
        'den_n_7': {'cell': 'NonscalingCell',  'block': 'DenseBlock',               'kernel': 7, 'prob':1/6/8},
        'inc_n_3': {'cell': 'NonscalingCell',  'block': 'InceptionBlock',           'kernel': 3, 'prob':1/6/8},
        'inc_n_5': {'cell': 'NonscalingCell',  'block': 'InceptionBlock',           'kernel': 5, 'prob':1/6/8},
        'inc_n_7': {'cell': 'NonscalingCell',  'block': 'InceptionBlock',           'kernel': 7, 'prob':1/6/8},
        'res_n_3': {'cell': 'NonscalingCell',  'block': 'ResNetBlock',              'kernel': 3, 'prob':1/6/8},
        'res_n_5': {'cell': 'NonscalingCell',  'block': 'ResNetBlock',              'kernel': 5, 'prob':1/6/8},
        'res_n_7': {'cell': 'NonscalingCell',  'block': 'ResNetBlock',              'kernel': 7, 'prob':1/6/8},
        'vgg_n_3': {'cell': 'NonscalingCell',  'block': 'VGGBlock',                 'kernel': 3, 'prob':1/6/8},
        'vgg_n_5': {'cell': 'NonscalingCell',  'block': 'VGGBlock',                 'kernel': 5, 'prob':1/6/8},
        'vgg_n_7': {'cell': 'NonscalingCell',  'block': 'VGGBlock',                 'kernel': 7, 'prob':1/6/8},

        'mobile_n_v1': {'cell': 'NonscalingCell',   'block': 'MobileNetBlock',      'kernel': 3, 'prob':1/2/8},
        'mobile_d_v1': {'cell': 'DownscalingCell',  'block': 'MobileNetBlock',      'kernel': 3, 'prob':1/2/8},
        'eff_n':       {'cell': 'NonscalingCell',   'block': 'EfficientNetBlock',   'kernel': 3, 'prob':1/2/8},
        'eff_d':       {'cell': 'DownscalingCell',  'block': 'EfficientNetBlock',   'kernel': 3, 'prob':1/2/8},

        'selfatt': {'cell': 'NonscalingCell',  'block': 'SelfAttentionBlock',                    'prob':1/8},

        'ide':     {'cell': 'NonscalingCell',  'block': 'IdentityBlock',                         'prob':1/8},
    }


def get_model(train_params, layer_dict, cell_list=None):
    num_classes = train_params["num_classes"]
    num_channels = train_params["num_channels"]
    image_size = train_params["image_size"]
    stem_filters = train_params["stem_filters"]
    max_depth = train_params["max_depth"]
    loss_class_weights = train_params["loss_class_weights"]
    use_loss_class_weights = train_params["use_loss_class_weights"]
    net_list = train_params["net_list"]

    patch_size = (image_size, image_size, num_channels)

    return build_net(
        input_shape=patch_size,
        num_classes=num_classes,
        stem_filters=stem_filters,
        max_depth=max_depth,
        layer_dict=layer_dict,
        net_list=net_list,
        cell_list=cell_list,
        loss_class_weights=loss_class_weights if use_loss_class_weights else None,
    )

def profile_model(model_name):
    """Profile the model already constructed in the Tensorflow default graph.

    Args:
        model_name: (str) some model identifying name.
    """
    profile_opts = tf.compat.v1.profiler.ProfileOptionBuilder

    param_stats = tf.compat.v1.profiler.profile(
        tf.compat.v1.get_default_graph(),
        cmd="graph",
        options=profile_opts.trainable_variables_parameter(),
    )
    total_params = param_stats.total_parameters

    param_stats = tf.compat.v1.profiler.profile(
        tf.compat.v1.get_default_graph(),
        cmd="op",
        options=profile_opts.float_operation(),
    )
    total_float_ops = param_stats.total_float_ops

    print(f"Model: {model_name}")
    print(f"total_parameters (Millions): {total_params/1e6:.2f}")
    print(f"total_float_ops (MFLOPS): {total_float_ops/1e6:.2f}")

# Function to perform random search for a given train configuration
def profiling_random_search(train_params, layer_dict):
    with tf.Graph().as_default():
        with tf.compat.v1.variable_scope("q_net"):
            # Load the best model
            net: tf.keras.Model = get_model(train_params, layer_dict)
            profile_model("Best Random Search Model")

# Train parameter configurations
train_configs = [
    {
        "batch_size": 32,
        "data_augmentation": True,
        "data_path": "spleen_dataset/data/Task09_Spleen_preprocessed/",
        "early_stopping_patience": 5,
        "epochs": 100,
        "eval_epochs": 6,
        "experiment_path": "experiments/config_experiment_spleen_1_rs",
        "folds": 5,
        "gpu_selected": 2,
        "image_size": 128,
        "initializations": 1,
        "log_level": "DEBUG",
        "loss_class_weights": [0.0, 1.0],
        "max_depth": 4,
        "num_channels": 1,
        "num_classes": 2,
        "phase": "continue_evolution",
        "skip_slices": 1,
        "stem_filters": 16,
        "threads": 3,
        "use_early_stopping_patience": False,
        "use_loss_class_weights": True,
        "net_list": [
                "inc_n_5",
                "eff_d",
                "den_d_3",
                "eff_n",
                "vgg_d_5"
            ]
    },
    {
        "batch_size": 32,
        "data_augmentation": True,
        "data_path": "prostate_dataset/data/Task05_Prostate_preprocessed/",
        "early_stopping_patience": 5,
        "epochs": 100,
        "eval_epochs": 6,
        "experiment_path": "experiments/config_experiment_prostate_1_rs",
        "folds": 5,
        "gpu_selected": 2,
        "image_size": 128,
        "initializations": 1,
        "log_level": "DEBUG",
        "loss_class_weights": [0.0, 0.5, 0.5],
        "max_depth": 4,
        "num_channels": 1,
        "num_classes": 3,
        "phase": "evolution",
        "skip_slices": 1,
        "stem_filters": 16,
        "threads": 3,
        "use_early_stopping_patience": False,
        "use_loss_class_weights": True,
        "net_list": [
                "vgg_n_5",
                "mobile_d_v1",
                "res_d_7",
                "den_n_5",
                "inc_n_3"
            ]
    },
    {
        "batch_size": 32,
        "data_augmentation": True,
        "data_path": "liver_dataset/data/Task03_Liver_preprocessed_limited",
        "early_stopping_patience": 5,
        "epochs": 100,
        "eval_epochs": 6,
        "experiment_path": "experiments/config_experiment_liver_1_rs",
        "folds": 5,
        "gpu_selected": 2,
        "image_size": 128,
        "initializations": 1,
        "log_level": "DEBUG",
        "loss_class_weights": [0.0, 0.5, 0.5],
        "max_depth": 4,
        "num_channels": 1,
        "num_classes": 3,
        "phase": "continue_evolution",
        "skip_slices": 1,
        "stem_filters": 16,
        "threads": 3,
        "use_early_stopping_patience": False,
        "use_loss_class_weights": True,
        "net_list": [
                "inc_n_5",
                "inc_n_3",
                "inc_n_5",
                "selfatt",
                "vgg_n_7"
            ]
    },
    {
        "batch_size": 32,
        "data_augmentation": True,
        "data_path": "bcss_dataset/data/BCSS_512_preprocessed_1k",
        "early_stopping_patience": 5,
        "epochs": 100,
        "eval_epochs": 6,
        "experiment_path": "experiments/config_experiment_bcss_2_rs",
        "folds": 5,
        "gpu_selected": 2,
        "image_size": 128,
        "initializations": 1,
        "log_level": "DEBUG",
        "loss_class_weights": [
            0.0, 0.04761904762, 0.04761904762, 0.04761904762, 0.04761904762,
            0.04761904762, 0.04761904762, 0.04761904762, 0.04761904762, 0.04761904762,
            0.04761904762, 0.04761904762, 0.04761904762, 0.04761904762, 0.04761904762,
            0.04761904762, 0.04761904762, 0.04761904762, 0.04761904762, 0.04761904762,
            0.04761904762, 0.04761904762,
        ],
        "max_depth": 4,
        "num_channels": 1,
        "num_classes": 22,
        "phase": "continue_evolution",
        "skip_slices": 1,
        "stem_filters": 16,
        "threads": 3,
        "use_early_stopping_patience": False,
        "use_loss_class_weights": True,
        "net_list": [
            "vgg_d_5",
            "inc_n_3",
            "res_d_7",
            "res_n_7",
            "res_n_3"
        ]
    },
]


train_configs = [
    {
        "name": "brats",
        "batch_size": 32,
        "data_augmentation": True,
        "data_path": "brain_tumour_dataset/data/Task01_BrainTumour_preprocessed_4k",
        "early_stopping_patience": 5,
        "epochs": 100,
        "eval_epochs": 6,
        "experiment_path": "experiments/config_experiment_brats_2_rs",
        "folds": 5,
        "gpu_selected": 2,
        "image_size": 128,
        "initializations": 1,
        "log_level": "DEBUG",
        "loss_class_weights": [0.0, 0.33, 0.33, 0.33],
        "max_depth": 4,
        "num_channels": 1,
        "num_classes": 4,
        "phase": "continue_evolution",
        "skip_slices": 1,
        "stem_filters": 16,
        "threads": 3,
        "use_early_stopping_patience": False,
        "use_loss_class_weights": True,
        "net_list": [
                "vgg_n_3",
                "res_n_3",
                "ide",
                "mobile_d_v1",
                "res_d_5"
            ]
    },
]
# Perform random search for each configuration
for train_params in train_configs:
    print(f"Profiling {train_params['name']}")
    profiling_random_search(train_params, layer_dict)