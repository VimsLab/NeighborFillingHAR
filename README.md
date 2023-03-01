## Neighborhood Filling for Human Activity Recognition
Querying for Human Activity Recognition (Point Cloud Sequences)
An extension of ellipsoid querying in human activity recognition (in PST-Transformer model).

# PST-Transformer with ellipsoid querying
## Installation
The code is tested with Red Hat Enterprise Linux Workstation release 7.7 (Maipo), g++ (GCC) 9.4.0, PyTorch (v1.10.2), CUDA 11.3.1 and cuDNN v8.2.0.

Compile the CUDA layers for [PointNet++](http://arxiv.org/abs/1706.02413), which we used for furthest point sampling (FPS) and radius neighbouring search:
```
cd modules
python setup.py install
```

## Download MSRAction3D dataset
Resquest for MSRAction3D dataset following the [instructions](https://sites.google.com/view/wanqingli/data-sets/msr-action3d)

mkdir data
Unzip MSRAction3D.zip and place the folder inside the data directory

## Training
Run the below command
python train-msr-small.py

### Log file and pre-trained model
Check log file log_9617.txt for the reported accuracy and the outputs folder for the pre-trained model.
