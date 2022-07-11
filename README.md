# TactileLocNeurons
This package is a PyTorch implementation of the paper "Event-Driven Tactile Learning with Location Spiking Neurons".

## Citation ##
Kang, Peng and Banerjee, Srutarshi and Chopp, Henry and Katsaggelos, Aggelos and Cossairt, Oliver. "Event-Driven Tactile Learning with Location Spiking Neurons". 
In _2022 International Joint Conference on Neural Networks (IJCNN 2022)_.

```bibtex
@inproceedings{kangTactile,
        title={Event-Driven Tactile Learning with Location Spiking Neurons},
        author={Kang, Peng and Banerjee, Srutarshi and Chopp, Henry and Katsaggelos, Aggelos and Cossairt, Oliver},
        booktitle={2022 International Joint Conference on Neural Networks (IJCNN)},
        pages={1--8},
        year={2022},
        organization={IEEE}
}
```

## Requirements
Python 3 with `slayerPytorch` and the packages in the `requirements.txt`:

The project has been tested with one RTX 3090 on Ubuntu 20.04 / Ubuntu 22.04. The training and testing time should be in minutes.

## Installation
1. Follow the requirements and installation of `slayerPytorch` to install it, see `slayerPytorch/README.md`.
2. Install any necessary packages in the `requirements.txt`.

## Datasets

1. Donwload the `preprocessed` data [here](https://clear-nus.github.io/visuotactile/download.html).
2. Save the preprocessed data for Objects, Containers, and Slip Detection in `datasets/preprocessed`.

## Training

```bash
python locsnn/train_location_snn.py --epoch 500 --lr 0.001 --sample_file 1 --batch_size 8 --fingers both --data_dir <preporcessed data dir> --hidden_size 32 --loss NumSpikes --mode location --network_config <network_config>/container_weight_location.yml  --task cw --checkpoint_dir <checkpoint dir>
```

## Experiments

1. The hybrid model with the whorl-like location order:
```bash
python locsnn/train_location_snn.py --epoch 500 --lr 0.001 --sample_file 1 --batch_size 8 --fingers both --data_dir <preporcessed data dir> --hidden_size 32 --loss NumSpikes --mode location_cat_whorl --network_config <network_config>/container_weight_location.yml  --task cw --checkpoint_dir <checkpoint dir>
```

2. Location Tactile SNN:
```bash
python locsnn/train_location_snn.py --epoch 500 --lr 0.001 --sample_file 1 --batch_size 8 --fingers both --data_dir <preporcessed data dir> --hidden_size 32 --loss NumSpikes --mode only_location --network_config <network_config>/container_weight_location_only.yml --task cw --checkpoint_dir <checkpoint dir>
```

3. $\lambda$ tuning in the weighted loss function:
$\lambda$ value can be changed in `slayerPytorch/src/spikeLoss.py`, but remember to install slayerPytorch again to activate the changes.
```bash
python locsnn/train_location_snn.py --epoch 500 --lr 0.001 --sample_file 1 --batch_size 8 --fingers both --data_dir <preporcessed data dir> --hidden_size 32 --loss WeightedLocationNumSpikes --mode location --network_config <network_config>/container_weight_location.yml --task cw --checkpoint_dir <checkpoint dir>
```

4. Confusion matrices on Containers
```bash
python confusion/confusion_location.py --runs <checkpoint dir>/cw_location_1
```

5. Timestep-wise inference
```bash
python timestep_inference/inference_timestep.py --runs <checkpoint dir>/cw_location_1 --save <timestep inference dir>
```

## Tranined model examples
1. Download models from [https://drive.google.com/drive/folders/1XBzpbk5Vt7E7qevlOW06GvFY0N_N8ymU?usp=sharing].
2. Save the models in `history` folder.


## Troubleshooting

if your scripts cannot find the `locsnn` module, please run in the root directory:

``` 
export PYTHONPATH=.
```

## Credits
The codes of this work are based on [slayerPytorch](https://github.com/bamsumit/slayerPytorch) and [VT-SNN](https://github.com/clear-nus/VT_SNN).






