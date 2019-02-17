# OptKeras

A Python package designed to optimize hyperparameters of Deep Learning models (a wrapper around Keras and Optuna)


### What is Keras?

Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano.

See:

https://github.com/keras-team/keras

https://keras.io/


### What is Optuna?

Optuna is an automatic hyperparameter optimization software framework, particularly designed for machine learning. 

See:
	
https://optuna.org/

https://github.com/pfnet/optuna

https://optuna.readthedocs.io/en/latest/index.html




### How to install OptKeras?


Option 1: directly install from the GitHub repository


	pip install git+https://github.com/Minyus/optkeras.git


Option 2: clone this GitHub repository, cd into the downloaded repository, and run:

	python setup.py install


### How to use OptKeras?


#### 0. Import OptKeras class

    from optkeras.optkeras import OptKeras
    
#### 1. Instantiate OptKeras class
	
  You can specify arguments for Optuna's create_study method and other arguments for OptKeras such as enable_pruning.
  
    ok = OptKeras(study_name = 'my_optimization', enable_pruning=False)


#### 2. Define objective function for Optuna

##### 2.1 Specify callbacks(trial) and keras_verbose to fit (or fit_generator) method of Keras
  
    model.fit(x_train, y_train, 
        validation_data = (x_test, y_test),
        callbacks = ok.callbacks(trial), 
        verbose = ok.keras_verbose )


##### 2.2 Return trial_best_value from OptKeras
  

    return ok.trial_best_value

	
#### 3. Run optimize

  You can specify arguments for Optuna's optimize method.
    
    ok.optimize(objective, n_trials=10, timeout=12*60*60)

  
Please see the examples at https://github.com/Minyus/optkeras/tree/master/examples.


### Parameaters for OptKeras

            monitor: The metric to optimize by Optuna. 'val_error' in default or 'val_loss'.
            enable_pruning: Enable pruning by Optuna. False in default.
                See https://optuna.readthedocs.io/en/latest/tutorial/pruning.html
            enable_keras_log: Enable logging by Keras CSVLogger callback. True in default.
                See https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger
            keras_log_file_suffix: Suffix of the file if enable_keras_log is True.
                '_Keras.csv' in default.
            enable_optuna_log: Enable generating a log file by Optuna study.trials_dataframe().
                True in default.
                See https://optuna.readthedocs.io/en/latest/reference/study.html#optuna.study.Study.trials_dataframe
            optuna_log_file_suffix: Suffix of the file if enable_optuna_log is True.
            models_to_keep: The number of models to keep.
                Either 1 in default , 0, or -1 (save all models).
            model_file_prefix: Prefix of the model file path if models_to_keep is not 0.
                'model_' in default.
            model_file_suffix: Suffix of the model file path if models_to_keep is not 0.
                '.hdf5' in default.
            directory_path: The path of the directory for the files.
                '' (Current working directory) in default.
            verbose: How much to print messages onto the screen.
                0 (no messages), 1 in default, 2 (troubleshooting)
            grid_search_mode: Run grid search instead of optimization. False in default.
            **kwargs: parameters for optuna.study.create_study():
                study_name, storage, sampler=None, pruner=None, direction='minimize'
                See https://optuna.readthedocs.io/en/latest/reference/study.html#optuna.study.create_study

### Why OptKeras was developed?
Current version of Optuna supports minimization but not maximization. 
This becomes a problem to use Optuna's pruning feature based on accuracy value (an objective to maximize) as Keras does not log error (= 1 - accuracy) in the default callback. OptKeras calculates error and val_error from acc and val_acc, respectively, in a Keras callback so Optuna can use it. 

### Why does the developer believe OptKeras is better than the other Python wrappers of Keras to optimize hyperparameters?

1. Optuna supports pruning option which can stop trials early based on the the interim objective values (error rate, loss, etc.). See https://optuna.org/#key_features . OptKeras can leverage Optuna's pruning option. If enable_pruning = True, OptKeras can stop training models (after the first epoch at the earliest) if the performance in early epochs are not good. Optuna's pruning algorithm is apparently "smarter" than Early-Stopping callback of Keras. Please note that some models which will achieve better performance later might be pruned due to bad performance in early epochs. It might be better to enable pruning in early phase of optimization for rough search and disable pruning in later phase.
  
2. Optuna manages logs in database using SQLAlchemy (https://www.sqlalchemy.org/) and can resume trials after interruption, even after the machine is rebooted (after 90 minutes of inactivity or 12 hours of runtime of Google Colab) if the databse is saved as a storage file. OptKeras can leverage this feature.

3. OptKeras can log metrics (accuracy, loss, and error for train and test datasets) with trial id and timestamp (begin and end) for each epoch to a CSV file.

4. OptKeras can save the Keras model files (only the best Keras model or all the models) with trial id in its file name so you can link to the log.

5. OptKeras supports grid search useful for benchmarking in addition to optimization.

### Will OptKeras limit features of Keras or Optuna?

Not at all! You can access the full feaures of Keras and Optuna even if OptKeras is used. 

### What was the tested environment for OptKeras?

	Google Colaboratory with GPU enabled
	NVIDIA Tesla K80
	Driver Version: 410.79 
	CUDA Version: 10.0
	Ubuntu 18.04.1 LTS
	Python 3.6.7
	Keras 2.2.4
	TensorFlow 1.13.0-rc1
	Optuna 0.7.0
	OptKeras 0.0.1

### About author 

Yusuke Minami

https://github.com/Minyus

https://www.linkedin.com/in/yusukeminami/

https://twitter.com/Minyus86


### License

MIT License (see https://github.com/Minyus/optkeras/blob/master/LICENSE).
