import random
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

# Function to perform random search for a given train configuration
def perform_random_search(train_params, layer_dict, random_search_iterations=50):
    num_nodes = 5
    best_mean_dsc = 0
    best_net_list = None

    for i in range(random_search_iterations):
        # Randomly sample nodes from layer_dict to create a net_list
        net_list = random.choices(list(layer_dict.keys()), k=num_nodes)
        print(f"Iteration {i + 1}: Testing net_list: {net_list}")

        # Train and evaluate the network
        mean_dsc, std_dsc, test_dice = get_model(train_params, layer_dict, net_list)

        print(f"Iteration {i + 1}: {mean_dsc} +- {std_dsc}, Test dice: {test_dice}")

        # Update the best configuration if the current one is better
        if mean_dsc > best_mean_dsc:
            best_mean_dsc = mean_dsc
            best_net_list = net_list

    print(f"Best configuration for {train_params['experiment_path']}: {best_net_list}")
    print(f"Best mean DSC: {best_mean_dsc}")
    print("-" * 50)

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
    },
]


# Perform random search for each configuration
for train_params in train_configs:
    print(f"Starting random search for {train_params['name']}")
    perform_random_search(train_params, layer_dict)