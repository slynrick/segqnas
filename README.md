# SegQNAS

## Neural Architecture Search for Semantic Segmentation using the QNAS algorithm

This repository contains code for the works presented in the following papers:

### Requirements

Before setting up the conda environment make sure you have `openmpi` (https://www.open-mpi.org/)

The follow these steps to use GPU (https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)

After that run the following to setup the enviroment:

```console
git clone https://github.com/GuilhermeBaldo/segqnas.git
cd segqnas/

python -m venv .venv
source .venv/bin/activate
pip install tensorflow-gpu



conda create -n segqnas python=3.8
conda activate segqnas
pip3 install --upgrade pip
pip3 install tensorflow==2.4
conda install pyyaml=5.3.1
conda install psutil
conda install mpi4py=3.0.3
conda install pandas
conda install Pillow
conda install -c simpleitk simpleitk
conda install -c conda-forge scikit-learn
conda install -c conda-forge monai
conda install -c conda-forge nibabel
conda install -c conda-forge tqdm
conda install -c conda-forge imgaug
conda install -c conda-forge albumentations
```

---
### Running Q-NAS

The entire process is divided in 3 steps (1 python script for each):
1. Dataset preparation
2. Run architecture search
3. Retrain final architecture

Optionally, the user can run the script `run_profiling.py` to get the number of parameters and FLOPs of one of the discovered architectures.

#### 1. Dataset Preparation

The user can choose to work with one of these datasets from the http://medicaldecathlon.com/:
- Prostate segmentation (Task 05)
- Spleen segmentation (Task 09)

The script `run_prostate_dataset_prep.py` and `run_spleen_dataset_prep.py` prepares the dataset for Task 05 and 09 respectively.

Here's an example of how to prepare the Task 09:

```console
python run_spleen_dataset_prep.py 
```

The dataset is going to be saved in the `spleen_dataset/`

#### 2. Run architecture search

All the configurable parameters to run the architecture search with Q-NAS are set in a _yaml configuration file_.   
This file sets 2 groups of parameters, namely: `QNAS` (parameters related to the evolution itself) and `train`   
(parameters related to the training session conducted to evaluate the architectures). The following
 template shows the type and meaning of each parameter in the configuration file: 

```yaml
QNAS:
    crossover_rate:      (float) crossover rate [0.0, 1.0]
    max_generations:     (int) maximum number of generations to run the algorithm
    max_num_nodes:       (int) maximum number of nodes in the network
    num_quantum_ind:     (int) number of quantum individuals
    repetition:          (int) number of classical individuals each quantum individual will generate
    replace_method:      (str) selection mechanism; 'best' or 'elitism'
    update_quantum_gen:  (int) generation frequency to update quantum genes
    update_quantum_rate: (float) rate for the quantum update (similar to crossover rate)
    save_data_freq:      (int) generation frequency to save train data of best model of current 

    layer_dict: {
      'function_name': {'block': 'function_class', 'kernel': 'kernel_size', 'prob': probability_value}
    }
    
    cell_list: [
      'cell_type'
    ]

train:
    batch_size:          (int) number of examples in a batch to train the networks.
    epochs:              (int) training epochs
    eval_epochs:         (int) last epochs used to average the dice coeficient
    initializations:     (int) number of initializations for the cross validation
    folds:               (int) number of folds for the cross validation
    stem_filters:        (int) number of filters in the depth 0
    max_depth:           (int) max depth

    # Dataset
    data_path:           (str) spleen_dataset/data/Task09_Spleen_preprocessed/ or spleen_dataset/data/Task05_Prostate_preprocessed/
    image_size:          (int) image input size
    num_channels:        (int) number of input channels
    skip_slices:         (int) skip every other n slice in the dataset
    num_classes:         (int) number of output channels
    data_augmentation:   (bool) True if data augmentation methods should be applied
```

We provide 3 configuration file examples in the folder `config_files`; one can use them as-is, or modify as
 needed.   
In summary, the files are:
- `config1.txt` evolves both the architecture and some hyperparameters of the network
- `config2.txt` evolves only the architecture and adopts penalization
- `config3.txt` evolves only the architecture with residual blocks and adopts penalization


This is an example of how to run architecture search for dataset `cifar10/cifar_tfr_10000` with `config1.txt`:

```shell script
nohup mpirun -n 9 python run_evolution.py --experiment_path experiment_1 --data_path spleen_dataset/data/Task09_Spleen_preprocessed/ --config_file config_files/config_experiment_1.txt --log_level DEBUG &
```

The number of workers in the MPI execution must be equal to the number of classical individuals. In `config1.txt`,   
this number is 20 (_num_quantum_ind_ (=5) x _repetition_ (=4) = 20). The output folder `my_exp_config1`   
looks like this:

>12_7   
csv_data   
data_QNAS.pkl   
log_params_evolution.txt   
log_QNAS.txt

The folder `12_7` has the Tensorflow files for the best network in the evolution; in this case, is the
 individual number `7` found in generation `12`. The folder `csv_data` has csv files with training
   information of the individuals (loss and mean iou for the best individuals in some generations). Both of
    these directories are not used in later steps, they are just information that one might want to inspect.

The file `data_QNAS.pkl` keeps all the evolution data (chromosomes, fitness values, number of evaluations, 
 best  individual ID ...). All the parameters (configuration file and command-line) are saved in
  `log_params_evolution.txt`, and `log_QNAS.txt` logs the evolution progression.

It is also possible to continue a finished evolution process. Note that all the parameters will be set as in   
`log_params_evolution.txt`, ignoring the values in the file indicated by `--config_file`. The only parameter 
  that can be overwritten is `max_generations`, so that one can set for how many generations the evolution
  will continue. To continue the above experiment for another 100 generations, the user can run:

```console
  nohup mpirun -n 9 python run_evolution.py \
  --experiment_path experiment_1_8 \
  --config_file config_files/config_experiment_1_spleen.txt \
  --log_level DEBUG &
```

Run `python run_evolution.py --help` for additional parameter details.


#### 3. Retrain network

After the evolution is complete, the final network can be retrained on the entire dataset (see papers for
 details). Here's an example of how to retrain the best network of the experiment saved in `my_exp_config1` 
  for 300 epochs with the dataset in `cifar10/cifar_tfr`, using the scheme (optimizer and hyperparameters) of
  the evolution:

```console
  nohup python run_retrain.py \
  --experiment_path experiment_1/ \
  --data_path spleen_dataset/data/Task09_Spleen_preprocessed/ \
  --retrain_folder retrained/ \
  --id_num 24_5 \
  --batch_size 32 \
  --max_epochs 100 \
  --eval_epochs 10 \
  --initializations 5 \
  --folds 5 \
  --log_level DEBUG &
```

After the training is complete, the directory `my_exp_config1/retrain` will contain the following files:

>best  
eval  
eval_test  
eval_train_eval  
checkpoint  
events.out.tfevents  
graph.pbtxt  
log_params_retrain.txt  
model.ckpt-52500.data-00000-of-00001  
model.ckpt-52500.index  
model.ckpt-52500.meta  

In the folder `best`, we have the best validation model saved. The file `log_params_retrain.txt` summarizes
 the training parameters. The other files and folders are generated by Tensorflow, including the last model
  saved, the graph and events for Tensorboard.


It is also possible to retrain the network with training schemes defined in the literature (check the help
 message for the `--lr_schedule` parameter). For example, to retrain the best network of experiment
  `my_exp_config2` using the `cosine` scheme, one can run:

```shell script
python run_retrain.py \
    --experiment_path my_exp_config2 \
    --data_path cifar10/cifar_tfr \
    --log_level INFO \
    --batch_size 256 \
    --eval_batch_size 1000 \
    --retrain_folder train_cosine \
    --threads 8 \
    --lr_schedule cosine \
    --run_train_eval
```

The script `run_retrain.py` also supports retraining any individual saved in `data_QNAS.pkl`: use the
 parameters `--generation` and `--individual` to indicate which one you want to train. Run `python
  run_retrain.py --help` for additional parameter details.


#### Profile architecture

If one wants to get the number of weights and the MFLOPs in a specific individual he/she can run
 `run_profiling.py`. For example, to get the values for individual `1` of generation `50` of the experiment
  saved in `my_exp_config3`, run:

```shell script
python run_profiling.py \
    --exp_path my_exp_config3 \
    --generation 50 \
    --individual 1
```