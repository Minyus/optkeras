# OptKeras

A Python wrapper around Optuna and Keras to optimize hyperparameters of Deep Learning models

## Q&A

### What is Optuna?

Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. 

Find details at:
```
https://github.com/pfnet/optuna

https://optuna.readthedocs.io/en/latest/index.html
```

### What is Keras?

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.

Find details at:
```
https://github.com/keras-team/keras

https://keras.io/
```


### How to install OptKeras?


Option 1: directly install from the GitHub repository

```
pip install git+https://github.com/Minyus/optkeras.git
```

Option 2: clone this GitHub repository, cd into the downloaded repository, and run:
```
python setup.py install
```

### Why OptKeras was developed?
Current version of Optuna supports minimization but not maximization. 
This becomes a problem to use Optuna's pruning feature based on accuracy value (objective to maximize) as Keras logs acc (and val_acc) and loss (and val_loss) but not error (= 1 - accuracy) in the default callback.

### Why does the developer believe OptKeras is better than other the Python wrappers of Keras to optimize hyperparameters?

1. Optuna has pruning option which can stop trials early based on the the interim objective (error rate, loss, etc. to minimize) values.  
OptKeras can leverage this Optuna's option. If enable_pruning = True, OptKeras can stop training models (after the first epoch at the earliest) if the interim objective values are not good. Optuna's pruning algorithm is "smarter" than Early-Stopping callback of Keras. (If you disagree, please share the evidence. I'm interested.) 
  
2. Optuna manages logs in database using SQLAlchemy (https://www.sqlalchemy.org/) and can resume trials if it is saved as a database file. 

3. OptKeras save both the Keras model files (or only the best Keras model) and CSV logs.

### Will OptKeras limit features of Keras or Optuna?

No. You can access the full feaures of Keras and Optuna even if OptKeras is used. 

### What was the tested environment for OptKeras 0.0.1?

```
Google Colaboratory with GPU enabled
NVIDIA Tesla K80
Driver Version: 410.79 
CUDA Version: 10.0
Ubuntu 18.04.1 LTS
Python 3.6.7
Optuna 0.7.0
Keras 2.2.4
TensorFlow 1.13.0-rc1
```

