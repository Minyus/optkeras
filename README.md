# OptKeras, a wrapper around Keras and Optuna

[![PyPI version](https://badge.fury.io/py/optkeras.svg)](https://badge.fury.io/py/optkeras)
![Python Version](https://img.shields.io/badge/python-3.5%20%7C%203.6%20%7C%203.7-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Minyus/optkeras/blob/master/examples/OptKeras_Example.ipynb)

A Python package designed to optimize hyperparameters of Keras Deep Learning models using Optuna. Supported features include pruning, logging, and saving models.


### What is Keras?

[Keras](https://keras.io/) is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.


### What is Optuna?

[Optuna](https://optuna.org/) is an automatic hyperparameter optimization software framework, particularly designed for machine learning. 


### What are the advantages of OptKeras?

- Optuna supports pruning option which can terminate the trial (training) early based on the interim objective values (loss, accuracy, etc.). Please see [Optuna's key features](https://optuna.org/#key_features). OptKeras can leverage Optuna's pruning option. If enable_pruning is set to True and the performance in early epochs is not good, OptKeras can terminate training (after the first epoch at the earliest) and try another parameter set.
- Optuna manages logs in database using [SQLAlchemy](https://www.sqlalchemy.org/) and can resume trials after interruption, even after the machine is rebooted (after 90 minutes of inactivity or 12 hours of runtime of Google Colab) if the database is saved as a storage file. OptKeras can leverage this feature.
- More epochs do not necessarily improve the performance of Deep Neural Network. OptKeras keeps the best value though epochs so it can be used as the final value.
- OptKeras can log metrics (loss, accuracy, etc. for train and test datasets) with trial id and timestamp (begin and end) for each epoch to a CSV file.
- OptKeras can save the best Keras models (only the best Keras model overall or all of the best models for each parameter set) with trial id in its file name so you can link to the log.
- OptKeras supports randomized grid search (randomized search by sampling parameter sets without replacement; grid search in a randomized order) useful if your primary purpose is benchmarking/comparison rather than optimization. 


### How to install OptKeras?

Option 1: install from the PyPI

```bash
	pip install optkeras
```

Option 2: install from the GitHub repository

```bash
	pip install git+https://github.com/Minyus/optkeras.git
```

Option 3: clone the [GitHub repository](https://github.com/Minyus/optkeras.git), cd into the downloaded repository, and run:

```bash
	python setup.py install
```

### How to use OptKeras?

Please see the [OptKeras example]( 
https://colab.research.google.com/github/Minyus/optkeras/blob/master/examples/OptKeras_Example.ipynb
) available in Google Colab (free cloud GPU) environment.

To run the code, navigate to "Runtime" >> "Run all".

To download the notebook file, navigate to "File" >> "Download .ipynb".

Here are the basic steps to use.

```python
""" Step 0. Import Keras, Optuna, and OptKeras """

from keras.models import Sequential
from keras.layers import Flatten, Dense, Conv2D
from keras.optimizers import Adam
import keras.backend as K

import optuna

from optkeras.optkeras import OptKeras


study_name = dataset_name + '_Simple'

""" Step 1. Instantiate OptKeras class
You can specify arguments for Optuna's create_study method and other arguments 
for OptKeras such as enable_pruning. 
"""

ok = OptKeras(study_name=study_name)


""" Step 2. Define objective function for Optuna """

def objective(trial):
    
    """ Step 2.1. Define parameters to try using methods of optuna.trial such as 
    suggest_categorical. In this simple demo, try 2*2*2*2 = 16 parameter sets: 
    2 values specified in list for each of 4 parameters 
    (filters, kernel_size, strides, and activation for convolution).
    """    
    model = Sequential()
    model.add(Conv2D(
        filters = trial.suggest_categorical('filters', [32, 64]), 
        kernel_size = trial.suggest_categorical('kernel_size', [3, 5]), 
        strides = trial.suggest_categorical('strides', [1, 2]), 
        activation = trial.suggest_categorical('activation', ['relu', 'linear']), 
        input_shape = input_shape ))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer = Adam(), 
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    """ Step 2.2. Specify callbacks(trial) and keras_verbose in fit 
    (or fit_generator) method of Keras model
    """
    model.fit(x_train, y_train, 
              validation_data = (x_test, y_test), shuffle = True,
              batch_size = 512, epochs = 2,
              callbacks = ok.callbacks(trial), 
              verbose = ok.keras_verbose )  
    
    """ Step 2.3. Return trial_best_value (or latest_value) """
    return ok.trial_best_value

""" Step 3. Run optimize. 
Set n_trials and/or timeout (in sec) for optimization by Optuna
"""
ok.optimize(objective, timeout = 60) # 1 minute for demo
```


### Will OptKeras limit features of Keras or Optuna?

Not at all! You can access the full feaures of Keras and Optuna even if OptKeras is used. 


### What parameaters are available for OptKeras?

- monitor: The metric to optimize by Optuna.
'val_loss' in default.
- enable_pruning: Enable pruning by Optuna.
False in default.
Reference: https://optuna.readthedocs.io/en/latest/tutorial/pruning.html
- enable_keras_log: Enable logging by Keras CSVLogger callback.
rue in default.
Reference: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger
- keras_log_file_suffix: Suffix of the file if enable_keras_log is True.
'_Keras.csv' in default.
- enable_optuna_log: Enable generating a log file by Optuna study.trials_dataframe().
True in default.
Reference: https://optuna.readthedocs.io/en/latest/reference/study.html#optuna.study.Study.trials_dataframe
- optuna_log_file_suffix: Suffix of the file if enable_optuna_log is True.
'Optuna.csv' in default.
- models_to_keep: The number of models to keep. Either 1 , 0, or -1 (save all models).
1 in default.
- ckpt_period: Period to save model check points.
1 in default.
Reference: https://keras.io/callbacks/#modelcheckpoint
- save_weights_only: if True, then only the model's weights will be saved (model.save_weights(filepath)), else the full model is saved (model.save(filepath)).
False in default.
Reference: https://keras.io/callbacks/#modelcheckpoint
- save_best_only: if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
True in default.
Reference: https://keras.io/callbacks/#modelcheckpoint
- model_file_prefix: Prefix of the model file path if models_to_keep is not 0.
'model_' in default.
- model_file_suffix: Suffix of the model file path if models_to_keep is not 0.
'.h5' in default.
- directory_path: The path of the directory for the files.
Current working directory in default.
- verbose: How much info to print onto the screen.
Either 0 (no messages), 1 , or 2 (troubleshooting)
1 in default.
- random_grid_search_mode: Run randomized grid search instead of optimization.
False in default.
- **kwargs: parameters for optuna.study.create_study():
study_name, storage, sampler=None, pruner=None, direction='minimize'
Reference: https://optuna.readthedocs.io/en/latest/reference/study.html#optuna.study.create_study


### Which version of Python is supported?
Python 3.5 or later

### What was the tested environment for OptKeras?

- Keras 2.2.4
- TensorFlow 1.14.0
- Optuna 0.14.0
- OptKeras 0.0.7

### About author 

Yusuke Minami

- https://github.com/Minyus
- https://www.linkedin.com/in/yusukeminami/
- https://twitter.com/Minyus86


### License

MIT License (see https://github.com/Minyus/optkeras/blob/master/LICENSE).
