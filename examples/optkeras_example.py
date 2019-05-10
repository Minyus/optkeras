#%% Import modules

import numpy as np
import pandas as pd
from IPython.core.display import display

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Activation, Flatten, Dense, Conv2D
from keras.layers import MaxPooling2D, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adagrad, RMSprop, Adam, Adadelta, Adamax, Nadam
import keras.backend as K

import keras
print('Keras', keras.__version__)

import tensorflow as tf
print('TensorFlow', tf.__version__)

# import Optuna and OptKeras after Keras
import optuna 
print('Optuna', optuna.__version__)

from optkeras.optkeras import OptKeras
import optkeras
print('OptKeras', optkeras.__version__)

# (Optional) Disable messages from Optuna below WARN level.
optuna.logging.set_verbosity(optuna.logging.WARN)


#%% Set up Dataset

dataset_name = 'MNIST'

if dataset_name in ['MNIST']:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    img_x, img_y = x_train.shape[1], x_train.shape[2]
    x_train = x_train.reshape(-1, img_x, img_y, 1)
    x_test = x_test.reshape(-1, img_x, img_y, 1)   
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    num_classes = 10
    input_shape = (img_x, img_y, 1)

print('x_train: ', x_train.shape)
print('y_train', y_train.shape)
print('x_test: ', x_test.shape)
print('y_test', y_test.shape)
print('input_shape: ', input_shape )

#%%  A simple Keras model

model = Sequential()
model.add(Conv2D(
    filters=32,
    kernel_size=3,
    strides=1,
    activation='relu',
    input_shape=input_shape ))
model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer=Adam(),
            loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train,
          validation_data=(x_test, y_test), shuffle=True,
          batch_size=512, epochs=2)

#%%  Optimization of a simple Keras model without pruning

study_name = dataset_name + '_Simple'

""" Step 1. Instantiate OptKeras class
You can specify arguments for Optuna's create_study method and other arguments 
for OptKeras such as enable_pruning. 
"""

ok = OptKeras(study_name=study_name,
              monitor='val_acc',
              direction='maximize')


""" Step 2. Define objective function for Optuna """

def objective(trial):
    
    """ Clear the backend (TensorFlow). See:
    https://www.tensorflow.org/api_docs/python/tf/keras/backend/clear_session
    """
    K.clear_session() 
    
    """ Step 2.1. Define parameters to try using methods of optuna.trial such as 
    suggest_categorical. In this simple demo, try 2*2*2*2 = 16 parameter sets: 
    2 values specified in list for each of 4 parameters 
    (filters, kernel_size, strides, and activation for convolution).
    """    
    model = Sequential()
    model.add(Conv2D(
        filters=trial.suggest_categorical('filters', [32, 64]),
        kernel_size=trial.suggest_categorical('kernel_size', [3, 5]),
        strides=trial.suggest_categorical('strides', [1, 2]),
        activation=trial.suggest_categorical('activation', ['relu', 'linear']),
        input_shape=input_shape ))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(),
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    """ Step 2.2. Specify callbacks(trial) and keras_verbose in fit 
    (or fit_generator) method of Keras model
    """
    model.fit(x_train, y_train, 
              validation_data=(x_test, y_test), shuffle=True,
              batch_size=512, epochs=2,
              callbacks=ok.callbacks(trial),
              verbose=ok.keras_verbose )
    
    """ Step 2.3. Return trial_best_value (or latest_value) """
    return ok.trial_best_value

""" Step 3. Run optimize. 
Set n_trials and/or timeout (in sec) for optimization by Optuna
"""
ok.optimize(objective, timeout = 3*60) # Run for 3 minutes for demo

""" Show Results """
print('Best trial number: ', ok.best_trial.number)
print('Best value:', ok.best_trial.value)
print('Best parameters: \n', ok.best_trial.params)

"""
Alternatively, you can access Optuna's study object to, for example, 
get the best parameters as well.
Please note that study.best_trial returns error if optimization trials 
were not completed (e.g. if you interupt execution) as of Optuna 0.7.0, 
so usage of OptKeras is recommended.
"""
print("Best parameters (retrieved directly from Optuna)", ok.study.best_trial.params)

""" Check the Optuna CSV log file """
pd.options.display.max_rows = 8 # limit rows to display
print('Data Frame read from', ok.optuna_log_file_path, '\n')
display(pd.read_csv(ok.optuna_log_file_path))

""" Check the Keras CSV log file """
pd.options.display.max_rows = 8 # limit rows to display
print('Data Frame read from', ok.keras_log_file_path, '\n')
display(pd.read_csv(ok.keras_log_file_path))


#%%  Randomized Grid Search of a simple Keras model

study_name = dataset_name + '_GridSearch'

""" To run randomized grid search, set random_grid_search_mode True """
ok = OptKeras(study_name=study_name, random_grid_search_mode=True)

def objective(trial):

    K.clear_session()

    model = Sequential()
    model.add(Conv2D(
        filters=trial.suggest_categorical('filters', [32, 64]),
        kernel_size=trial.suggest_categorical('kernel_size', [3, 5]),
        strides=trial.suggest_categorical('strides', [1]),
        activation=trial.suggest_categorical('activation', ['relu', 'linear']),
        input_shape=input_shape ))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=Adam(),
                loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train,
              validation_data=(x_test, y_test), shuffle = True,
              batch_size=512, epochs=2,
              callbacks=ok.callbacks(trial),
              verbose=ok.keras_verbose )

    return ok.trial_best_value

""" Set the number of parameter sets as n_trials for complete grid search """
ok.random_grid_search(objective, n_trials = 2*2*2*2) # 2*2*2*2 = 16 param sets

#%%  Optimization of a Keras model using more Optuna's features such as pruning

study_name = dataset_name + '_Optimized'

ok = OptKeras( 
    # parameters for optuna.create_study
    storage='sqlite:///' + study_name + '_Optuna.db', 
    sampler=optuna.samplers.TPESampler(
        consider_prior=True, prior_weight=1.0, 
        consider_magic_clip=True, consider_endpoints=False, 
        n_startup_trials=10, n_ei_candidates=24, 
        seed=None), 
    pruner=optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=4, min_early_stopping_rate=0), 
    study_name=study_name,
    load_if_exists=True,
    # parameters for OptKeras
    monitor='val_acc',
    direction='maximize',
    enable_pruning=True, 
    models_to_keep=1, # Either 1, 0, or -1 (save all models) 
    verbose=1,
    )

def objective(trial): 
    epochs = 10
    
    K.clear_session()   
    model = Sequential()
    
    if trial.suggest_int('Conv', 0, 1):  
        # 1 Convolution layer
        i = 1
        model.add(Conv2D(
            filters=int(trial.suggest_discrete_uniform(
                'Conv_{}_num_filters'.format(i), 32, 64, 32)), 
            kernel_size=tuple([trial.suggest_int(
                'Conv_{}_kernel_size'.format(i), 2, 3)] * 2),
            activation='relu',
            input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=tuple([trial.suggest_int(
                'Conv_{}_max_pooling_size'.format(i), 2, 3)] * 2)))
        model.add(Dropout(trial.suggest_discrete_uniform(
                'Conv_{}_dropout_rate'.format(i), 0, 0.5, 0.25) ))
        model.add(Flatten())        
    else:
        model.add(Flatten(input_shape=input_shape))
    # 2 Fully connected layers
    for i in np.arange(2) + 1:
        model.add(Dense(int(trial.suggest_discrete_uniform(
            'FC_{}_num_hidden_units'.format(i), 256, 512, 256))))
        if trial.suggest_int('FC_{}_batch_normalization'.format(i), 0, 1):
            model.add(BatchNormalization())
        model.add(Activation(trial.suggest_categorical(
            'FC_{}_acivation'.format(i), ['relu'])))
        model.add(Dropout(
            trial.suggest_discrete_uniform(
                'FC_{}_dropout_rate'.format(i), 0, 0.5, 0.25) ))
        
    # Output layer    
    model.add(Dense(num_classes, activation='softmax'))
    
    optimizer_dict = { \
    #'Adagrad': Adagrad(),
    'Adam': Adam() }
    
    model.compile(optimizer = optimizer_dict[
        trial.suggest_categorical('Optimizer', list(optimizer_dict.keys()))],
          loss='sparse_categorical_crossentropy', metrics=['accuracy'])    
    
    if ok.verbose >= 2: model.summary()
    
    batch_size = trial.suggest_int('Batch_size', 256, 256)
    data_augmentation = trial.suggest_int('Data_augmentation', 0, 1)
    
    if not data_augmentation:
        # [Required] Specify callbacks(trial) in fit method
        model.fit(x_train, y_train, batch_size=batch_size,
                  epochs=epochs, validation_data=(x_test, y_test),
                  shuffle=True,
                  callbacks=ok.callbacks(trial),
                  verbose=ok.keras_verbose )
    
    if data_augmentation:
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            width_shift_range=[-1, 0, +1], # 1 pixel
            height_shift_range=[-1, 0, +1], # 1 pixel
            zoom_range=[0.95,1.05],  # set range for random zoom
            horizontal_flip=False,  # disable horizontal flip
            vertical_flip=False )  # disable vertical flip
        datagen.fit(x_train)
        # [Required] Specify callbacks(trial) in fit_generator method
        model.fit_generator(datagen.flow(x_train, y_train, 
                                         batch_size=batch_size),
                            epochs=epochs, validation_data=(x_test, y_test),
                            steps_per_epoch=len(x_train) // batch_size,
                            callbacks=ok.callbacks(trial),
                            verbose=ok.keras_verbose )
    
    # [Required] return trial_best_value (recommended) or latest_value
    return ok.trial_best_value

# Set n_trials and/or timeout (in sec) for optimization by Optuna
ok.optimize(objective, timeout=3*60) # Run for 3 minutes for demo

#%%
"""## The end of code.

Please feel free to post questions or feedback [here](
https://github.com/Minyus/optkeras/issues
)
"""