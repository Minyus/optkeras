# optkeras
OptKeras: Wrapper of Optuna and Keras to optimize hyperparameters of Deep Learning models

### What is Optuna?

Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. 

Find details at:
https://github.com/pfnet/optuna
https://optuna.readthedocs.io/en/latest/index.html

### What is Keras?

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.

Find details at:
https://github.com/keras-team/keras
https://keras.io/


### How to install


Option 1: directly install from the GitHub repository

```
pip install git+https://github.com/Minyus/optkeras.git
```

Option 2: clone this GitHub repository, cd into the downloaded repository, and run 
```
python setup.py install
```

### Tested environment for OptKeras 0.0.1:

```
Google Colaboratory with GPU enabled
NVIDIA-SMI 410.79 
Driver Version: 410.79 
CUDA Version: 10.0
Ubuntu 18.04.1 LTS
Python 3.6.7
Optuna 0.7.0
Keras 2.2.4
TensorFlow 1.13.0-rc1
```
