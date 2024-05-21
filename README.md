# MonoFRD:Color intuitive feature guided depth-height fusion and volume rendering for monocular 3D detection
## Introduction


![Framework](./fig1.png)



## Overview
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Pretrained Models](#pretrained-models)
- [Citation](#citation)


## Installation

### Requirements
All the codes for training and evaluation are tested in the following environment:
* Linux (tested on Ubuntu 18.04)
* Python 3.8
* PyTorch 1.8.1
* Torchvision 0.9.1
* CUDA 11.1.1
* [`spconv 1.2.1 (commit f22dd9)`](https://github.com/traveller59/spconv)


### Installation Steps

a. Clone this repository.
```shell
git clone https://github.com/xinghanliuying/monofrd.git
```

b. Install the dependent libraries as follows:

* Install the dependent python libraries: 
```shell
pip install -r requirements.txt 
```


* Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 

```shell
git clone https://github.com/traveller59/spconv
git reset --hard f22dd9
git submodule update --recursive
python setup.py bdist_wheel
pip install ./dist/spconv-1.2.1-cp38-cp38m-linux_x86_64.whl
```

* Install modified mmdetection from [`[mmdetection_kitti]`](https://github.com/xy-guo/mmdetection_kitti)
```shell
git clone https://github.com/xy-guo/mmdetection_kitti
python setup.py develop
```

c. Install this library by running the following command:
```shell
python setup.py develop
```

## Getting Started
### Dataset Preparation
For KITTI, dataset configs are located within [configs/stereo/dataset_configs](../configs/stereo/dataset_configs), 
and the model configs are located within [configs/stereo/kitti_models](../configs/stereo). 

```
PATH
├── data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & image_3
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── configs
├── monofrd
├── tools
```

* You can also choose to link your KITTI and Waymo dataset path by
```
YOUR_KITTI_DATA_PATH=~/data/kitti_object
ln -s $YOUR_KITTI_DATA_PATH/ImageSets/ ./data/kitti/
ln -s $YOUR_KITTI_DATA_PATH/training/ ./data/kitti/
ln -s $YOUR_KITTI_DATA_PATH/testing/ ./data/kitti/
```

* Generate the data infos by running the following command: 
```python 
python -m monofrd.datasets.kitti.lidar_kitti_dataset create_kitti_infos
python -m monofrd.datasets.kitti.lidar_kitti_dataset create_gt_database_only
```

### Training & Testing
#### Train a model  

* Train with multiple GPUs
```
./scripts/dist_train.sh ${NUM_GPUS} 'exp_name' ./configs/stereo/kitti_models/monofrd.yaml
```

#### Test and evaluate the pretrained models

* To test with multiple GPUs:
```
./scripts/dist_test_ckpt.sh ${NUM_GPUS} ./configs/stereo/kitti_models/monofrd.yaml ./ckpt/pretrained_monofrd.pth
```

## Pretrained Models
### KITTI 3D Object Detection Baselines
The results are the BEV / 3D detection performance of Car class on the *val* set of KITTI dataset.
* All models are trained with 5 NVIDIA 2080Ti GPUs and are available for download.
* The training time is measured with 5 NVIDIA 2080Ti GPUs and PyTorch 1.8.1.

| Training Time |     Easy@R40 | Moderate@R40  |   Hard@R40    |
|---------------|-------------:|:-------------:|:-------------:|
| ~10 hours     | 31.54/ 23.61 | 23.67 / 17.50 | 21.12 / 15.28 |
## Citation



## Acknowledgements
This project benefits from the following codebases. Thanks for their great works! 
* [monofrd](https://github.com/cskkxjk/monofrd.git) 
* [CaDDN](https://github.com/TRAILab/CaDDN) 
* [LIGA-Stereo](https://github.com/xy-guo/LIGA-Stereo)
* [volSDF](https://github.com/lioryariv/volsdf)
* [mipnerf_pl](https://github.com/hjxwhy/mipnerf_pl) 

