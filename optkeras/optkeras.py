# -*- coding: utf-8 -*-

from keras.callbacks import Callback, CSVLogger, ModelCheckpoint
import optuna

import os, glob
import numpy as np
from datetime import datetime


class OptKeras(Callback):
    def __init__(self, 
                 monitor = 'val_error', # alternatively 'val_loss'
                 enable_pruning = False,
                 verbose = 1,                                          
                 num_models_to_save = 1, # either 1, 0, or -1 (save all models)
                 model_file_prefix = 'model_', 
                 model_file_suffix = '.hdf5', 
                 save_log = True,
                 **kwargs):                     
        # Create Optuna Study
        self.study = optuna.create_study(**kwargs)
        self.study_name = self.study.study_name
        self.keras_log_file_path = study_name + '_Keras.csv'
        self.optuna_log_file_path = study_name + '_Optuna.csv'
        self.model_file_prefix = study_name + '_' + model_file_prefix
        self.model_file_suffix = model_file_suffix
        self.monitor = monitor    
        # The larger acc or val_acc, the better
        self.mode_max = (monitor in ['acc', 'val_acc'])
        self.mode = 'max' if self.mode_max else 'min'
        self.default_value = -np.Inf if self.mode_max else np.Inf
        self.latest_logs = {}
        self.latest_value = self.default_value
        self.trial_best_logs = {}
        self.trial_best_value = self.default_value
        self.num_models_to_save = num_models_to_save
        self.save_log = save_log
        self.enable_pruning = enable_pruning
        self.verbose = verbose
        self.keras_verbose = max(self.verbose - 1 , 0) # decrement
        if self.verbose >= 1:
            print('[{}] '.format(self.get_datetime()), 
            'Ready for optimization. (message printed as verbose is set to 1+)')

    def optimize(self, *args, **kwargs):
        self.study.optimize(*args, **kwargs)
        self.post_process()

    def get_model_file_path(self, trial_id = None):
        return ''.join([self.model_file_prefix, 
                        '*' if trial_id is None else '{:06d}'.format(trial_id),
                        self.model_file_suffix ])

    def clean_up_model_files(self): 
        # currently version supports only num_models_to_save <= 1
        if self.num_models_to_save in [1]: 
            self.best_model_file_path = \
                self.get_model_file_path(self.best_trial.trial_id)
            self.model_file_list = \
                glob.glob(self.get_model_file_path())
            for model_file in self.model_file_list:
                if model_file not in [self.best_model_file_path]:
                    os.remove(model_file)

    def get_datetime(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

    def callbacks(self, trial):
        self.synch_with_optuna()
        callbacks = []
        self.trial = trial
        self.model_file_path = self.get_model_file_path(trial.trial_id)     
        callbacks.append(self)
        if self.save_log:
            csv_logger = CSVLogger(self.keras_log_file_path, append = True)
            callbacks.append(csv_logger)
        if self.num_models_to_save != 0:            
            check_point = ModelCheckpoint(filepath = self.model_file_path, 
                monitor = self.monitor, mode = self.mode, save_best_only = True, 
                save_weights_only = False, period = 1, 
                verbose = self.keras_verbose)
            callbacks.append(check_point)
            self.clean_up_model_files()                
        if self.enable_pruning:
            pruning = \
                optuna.integration.KerasPruningCallback(trial, self.monitor)
            callbacks.append(pruning)       
        return callbacks

    def save_logs_as_optuna_attributes(self):
        for key, val in self.trial_best_logs.items():
            self.trial.set_user_attr(key, val)

    def generate_optuna_log_file(self):
        try:
            df = self.study.trials_dataframe()
            df.columns = ['_'.join(str_list(col)).rstrip('_').
                          replace('user_attrs_','').replace('params_','') \
                          for col in df.columns.values]
            self.optuna_log = df
            self.optuna_log.to_csv(self.optuna_log_file_path, index=False)
        except:
            print('[{}] '.format(self.get_datetime()), 
            'Failed to generate Optuna log file. Continue to run.')

    def synch_with_optuna(self):
        # Generate the Optuna CSV log file
        self.generate_optuna_log_file()
        # best_trial
        try:
            self.best_trial = self.study.best_trial
        except:
            self.best_trial = optuna.structs.FrozenTrial(
                None, None, None, None, None, None, None, None, None, None)
        # latest_trial
        self.latest_trial = optuna.structs.FrozenTrial(
                None, None, None, None, None, None, None, None, None, None)
        if len(self.study.trials) >= 1:
            if self.study.trials[-1].state == optuna.structs.TrialState.RUNNING:
                if len(self.study.trials) >= 2:
                    self.latest_trial = self.study.trials[-2]
            else: 
                self.latest_trial = self.study.trials[-1]         
        self.print_results()

    def print_results(self):
        if self.verbose >= 1 and len(self.study.trials) > 0:
            # if any trial with a valid value is found, show the result
            print(
                '[{}] '.format(self.get_datetime()) + \
                'Latest trial id: {}'.format(self.latest_trial.trial_id) + \
                ', value: {}'.format(self.latest_trial.value) + \
                ' ({}) '.format(self.latest_trial.state) + \
                '| Best trial id: {}'.format(self.best_trial.trial_id) + \
                ', value: {}'.format(self.best_trial.value) + \
                ', parameters: {}'.format(self.best_trial.params) )

    def post_process(self):
        self.synch_with_optuna()
        self.clean_up_model_files()

    def on_epoch_begin(self, epoch, logs={}):
        self.datetime_epoch_begin = self.get_datetime()
        ## Reset trial best logs
        self.trial_best_logs = {}

    def on_epoch_end(self, epoch, logs={}):
        self.datetime_epoch_end = self.get_datetime()
        # Add error and val_error to logs for use as an objective to minimize
        logs.setdefault('error', 1 - logs.get('acc', 0))
        logs.setdefault('val_error', 1 - logs.get('val_acc', 0))
        logs.setdefault('_Datetime_epoch_begin', self.datetime_epoch_begin)
        logs.setdefault('_Datetime_epoch_end', self.datetime_epoch_end)
        logs.setdefault('_Trial_id', self.trial.trial_id)
        logs.setdefault('_Monitor', self.monitor)
        # Update the best logs

        def update_flag(latest, best, mode_max = False):
            return (mode_max and (latest > best)) \
                or ((not mode_max) and (latest < best))

        def update_best_logs(latest_logs = {}, best_logs = {}, 
                             monitor = 'val_error', 
                             mode_max = False, default_value=np.Inf):
            latest = latest_logs.get(monitor, default_value)
            best = best_logs.get(monitor, default_value)
            if update_flag(latest, best, mode_max = mode_max):
                best_logs.update(latest_logs)
        self.latest_logs = logs.copy()
        # Update trial best
        update_best_logs(self.latest_logs, self.trial_best_logs,  
                         self.monitor, self.mode_max, self.default_value)
        self.trial_best_value = \
            self.trial_best_logs.get(self.monitor, self.default_value)        
        # Recommended: save the logs from the best epoch as attributes
        # (logs include timestamp, monitor, val_acc, val_error, val_loss)
        self.save_logs_as_optuna_attributes()


def str_list(input_list):
    return ['{}'.format(e) for e in input_list]
