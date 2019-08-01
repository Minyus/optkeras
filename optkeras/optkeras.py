import keras.backend as K
from keras.callbacks import Callback, CSVLogger, ModelCheckpoint

import os, glob
import numpy as np
from datetime import datetime
from pathlib import Path

import optuna


class OptKeras(Callback):
    """ The main class of OptKeras which can act as a callback for Keras.
    """
    def __init__(self,
                 monitor='val_loss',
                 enable_pruning=False,
                 enable_keras_log=True,
                 keras_log_file_suffix='Keras.csv',
                 enable_optuna_log=True,
                 optuna_log_file_suffix='Optuna.csv',
                 models_to_keep=1,
                 ckpt_period=1,
                 save_weights_only=False,
                 save_best_only=True,
                 model_file_prefix='model_',
                 model_file_suffix='.h5',
                 directory_path='',
                 verbose=1,
                 random_grid_search_mode=False,
                 **kwargs):
        """ Wrapper of optuna.create_study
        Args:
            monitor: The metric to optimize by Optuna.
                'val_loss' in default.
            enable_pruning: Enable pruning by Optuna.
                False in default.
                Reference: https://optuna.readthedocs.io/en/latest/tutorial/pruning.html
            enable_keras_log: Enable logging by Keras CSVLogger callback.
                rue in default.
                Reference: https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/CSVLogger
            keras_log_file_suffix: Suffix of the file if enable_keras_log is True.
                '_Keras.csv' in default.
            enable_optuna_log: Enable generating a log file by Optuna study.trials_dataframe().
                True in default.
                See https://optuna.readthedocs.io/en/latest/reference/study.html#optuna.study.Study.trials_dataframe
            optuna_log_file_suffix: Suffix of the file if enable_optuna_log is True.
                'Optuna.csv' in default.
            models_to_keep: The number of models to keep. Either 1 , 0, or -1 (save all models).
                1 in default.
            ckpt_period: Period to save model check points.
                1 in default.
                Reference: https://keras.io/callbacks/#modelcheckpoint
            save_weights_only: if True, then only the model's weights will be saved (model.save_weights(filepath)), else the full model is saved (model.save(filepath)).
                False in default.
                Reference: https://keras.io/callbacks/#modelcheckpoint
            save_best_only: if save_best_only=True, the latest best model according to the quantity monitored will not be overwritten.
                True in default.
                Reference: https://keras.io/callbacks/#modelcheckpoint
            model_file_prefix: Prefix of the model file path if models_to_keep is not 0.
                'model_' in default.
            model_file_suffix: Suffix of the model file path if models_to_keep is not 0.
                '.h5' in default.
            directory_path: The path of the directory for the files.
                Current working directory in default.
            verbose: How much info to print onto the screen.
                Either 0 (no messages), 1 , or 2 (troubleshooting)
                1 in default.
            random_grid_search_mode: Run randomized grid search instead of optimization.
                False in default.
            **kwargs: parameters for optuna.study.create_study():
                study_name, storage, sampler=None, pruner=None, direction='minimize'
                Reference: https://optuna.readthedocs.io/en/latest/reference/study.html#optuna.study.create_study
        """
        self.random_grid_search_mode = random_grid_search_mode
        if self.random_grid_search_mode:
            kwargs.setdefault('sampler', optuna.samplers.RandomSampler())
            kwargs.setdefault('pruner', RepeatPruner())
            enable_pruning = True
        self.gs_progress = 0
        self.study = optuna.create_study(**kwargs)
        self.study_name = self.study.study_name
        self.directory_path = directory_path
        self.keras_log_file_path = self.add_dir(keras_log_file_suffix)
        self.optuna_log_file_path = self.add_dir(optuna_log_file_suffix)

        self.monitor = monitor
        self.direction = kwargs.get('direction', 'minimize')
        self.mode_max = self.direction != 'minimize'
        self.default_value = -np.Inf if self.mode_max else np.Inf

        self.latest_logs = {}
        self.latest_value = np.Inf
        self.trial_best_logs = {}
        self.trial_best_value = np.Inf
        self.enable_pruning = enable_pruning
        self.enable_keras_log = enable_keras_log
        self.enable_optuna_log = enable_optuna_log
        self.models_to_keep = models_to_keep
        self.ckpt_period = ckpt_period
        self.save_weights_only = save_weights_only
        self.save_best_only=save_best_only
        self.model_file_prefix = self.add_dir(model_file_prefix)
        self.model_file_suffix = model_file_suffix
        self.verbose = verbose
        self.keras_verbose = max(self.verbose - 1, 0) # decrement
        if self.verbose >= 1:
            print('[{}]'.format(self.get_datetime()),
            '[OptKeras] Ready for optimization. (message printed as verbose is set to 1+)')

    def add_dir(self, suffix_str):
        p = Path(self.directory_path) / (self.study_name + '_' + suffix_str)
        return str(p)

    def optimize(self, *args, **kwargs):
        """
        Args:
            *args: parameters for study.optimize optimize(): func
            **kwargs: parameters for study.optimize optimize():
                n_trials=None, timeout=None, n_jobs=1, catch=(<class 'Exception'>, )
                See https://optuna.readthedocs.io/en/latest/reference/study.html#optuna.study.Study.optimize

        Returns: None
        """
        if K.backend() == 'tensorflow':
            fun = args[0]

            def fun_tf(trial):
                K.clear_session()
                fun(trial)

            args = (fun_tf,) + args[1:]

        self.study.optimize(*args, **kwargs)
        self.post_process()

    def get_model_file_path(self, trial_num = None):
        """
        Args:
            trial_num: Trial Id of Optuna study

        Returns: model file path string
        """
        return ''.join([self.model_file_prefix, 
                        '*' if trial_num is None else '{:06d}'.format(trial_num),
                        self.model_file_suffix ])

    def clean_up_model_files(self):
        """ Delete model files not needed. Currently version supports only models_to_keep <= 1
        Returns: None
        """
        if self.models_to_keep in [1]:
            self.best_model_file_path = \
                self.get_model_file_path(self.best_trial.number)
            self.model_file_list = \
                glob.glob(self.get_model_file_path())
            for model_file in self.model_file_list:
                if model_file not in [self.best_model_file_path]:
                    os.remove(model_file)

    def get_datetime(self):
        """ Get the date time now
        Returns: the date time string in '%Y-%m-%d %H:%M:%S.%f' format
        """
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')

    def callbacks(self, trial):
        """ Callbacks to be passed to Keras
        Args:
            trial: Optuna trial
        """
        self.synch_with_optuna()
        callbacks = []
        self.trial = trial
        self.model_file_path = self.get_model_file_path(trial.number)
        callbacks.append(self)
        if self.enable_keras_log:
            csv_logger = CSVLogger(self.keras_log_file_path, append=True)
            callbacks.append(csv_logger)
        if self.models_to_keep != 0:
            check_point = ModelCheckpoint(
                filepath=self.model_file_path,
                monitor=self.monitor,
                mode=self.direction[:3],
                save_best_only=self.save_best_only,
                save_weights_only=self.save_weights_only,
                period=self.ckpt_period,
                verbose=self.keras_verbose
                )
            callbacks.append(check_point)
            self.clean_up_model_files()                
        if self.enable_pruning:
            pruning = \
                optuna.integration.KerasPruningCallback(trial, self.monitor)
            callbacks.append(pruning)       
        return callbacks

    def save_logs_as_optuna_attributes(self):
        """ Save the logs in Optuna's database as user attributes.
        Returns: None
        """
        for key, val in self.trial_best_logs.items():
            self.trial.set_user_attr(key, val)

    def generate_optuna_log_file(self):
        """ Generate log file from Optuna
        Returns: None
        """
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
        """ Exchange information with Optuna
        Returns: None
        """
        # Generate the Optuna CSV log file
        if self.enable_optuna_log:
            self.generate_optuna_log_file()

        # best_trial
        try:
            self.best_trial = self.study.best_trial
        except:
            self.best_trial = get_trial_default()

        # latest_trial
        self.latest_trial = get_trial_default()
        if len(self.study.trials) >= 1:
            if self.study.trials[-1].state == optuna.structs.TrialState.RUNNING:
                if len(self.study.trials) >= 2:
                    self.latest_trial = self.study.trials[-2]
            else: 
                self.latest_trial = self.study.trials[-1]         
        if self.verbose >= 1:
            self.print_results()

    def print_results(self):
        """ Print summary of results
        Returns: None
        """
        if len(self.study.trials) > 0 and \
            (self.verbose >= 2 or \
            (self.verbose == 1 and \
             self.latest_trial.state != optuna.structs.TrialState.PRUNED)):
            # if any trial with a valid value is found, show the result
            report_list = ['[{}] '.format(self.get_datetime())]
            if self.latest_trial.number is not None:
                report_list.extend([
                'Trial#: {}'.format(self.latest_trial.number),
                ])
            if self.latest_trial.value is not None:
                report_list.extend([
                    ', value: {:.6e}'.format(self.latest_trial.value),
                    ])
            if self.latest_trial.state != TrialState.COMPLETE:
                report_list.extend([
                        ' ({}) '.format(self.latest_trial.state),
                        ])
            if self.best_trial.value is not None:
                report_list.extend([
                    '| Best trial#: {}'.format(self.best_trial.number),
                    ', value: {:.6e}'.format(self.best_trial.value),
                    ', params: {}'.format(self.best_trial.params),
                    ])

            report_str = ''.join(report_list)
            print(report_str)

    def post_process(self):
        """ Process after optimization
        Returns: None
        """
        self.synch_with_optuna()
        self.clean_up_model_files()

    def on_epoch_begin(self, epoch, logs={}):
        """ Called at the beginning of every epoch by Keras
        Args:
            epoch:
            logs:
        """
        self.datetime_epoch_begin = self.get_datetime()
        # Reset trial best logs
        self.trial_best_logs = {}

    def on_epoch_end(self, epoch, logs={}):
        """ Called at the end of every epoch by Keras
        Args:
            epoch:
            logs:
        """
        assert self.monitor in logs, '[OptKeras] Monitor variable needs to be in the logs dictionary. Use a callback.'

        self.datetime_epoch_end = self.get_datetime()
        # Add error and val_error to logs for use as an objective to minimize

        logs['_Datetime_epoch_begin'] = self.datetime_epoch_begin
        logs['_Datetime_epoch_end'] = self.datetime_epoch_end
        logs['_Trial_num'] = self.trial.number
        # Update the best logs

        def update_flag(mode_max, latest, best):
            return (mode_max and (latest > best)) \
                or ((not mode_max) and (latest < best))

        def update_best_logs(
                monitor, mode_max, default_value,
                latest_logs={}, best_logs={}
                ):
            latest = latest_logs.get(monitor, default_value)
            best = best_logs.get(monitor, default_value)
            if update_flag(mode_max, latest, best):
                best_logs.update(latest_logs)
        self.latest_logs = logs.copy()
        # Update trial best
        update_best_logs(
            self.monitor, self.mode_max, self.default_value,
            self.latest_logs, self.trial_best_logs,
            )
        self.trial_best_value = \
            self.trial_best_logs.get(self.monitor, self.default_value)
        # Recommended: save the logs from the best epoch as attributes
        # (logs include timestamp, monitor, val_acc, val_error, val_loss)
        self.save_logs_as_optuna_attributes()

    def random_grid_search(self, func, n_trials, **kwargs):
        """ Grid search
        Args:
            func: A callable that implements objective function.
            n_trials: Number of combinations of parameters.
            **kwargs: The other parameters for study.optimize:
                timeout=None, n_jobs=1, catch=(<class 'Exception'>, )
                See https://optuna.readthedocs.io/en/latest/reference/study.html#optuna.study.Study.optimize
        Returns: None
        """
        while True:
            trials = self.study.trials
            completed_params_list = []
            if len(trials) >= 1:
                completed_params_list = \
                    [t.params for t in trials if t.state == optuna.structs.TrialState.COMPLETE]
                if self.verbose >= 3:
                    print('[{}] '.format(self.get_datetime()) + 'Parameters completed: ', completed_params_list)
            self.n_completed = len(completed_params_list) # TODO: change to unique num of completed_params_list
            n_trials = int(n_trials)
            gs_progress = self.n_completed / n_trials
            if gs_progress > self.gs_progress:
                self.gs_progress = gs_progress
                if self.verbose >= 1:
                    print('[{}] '.format(self.get_datetime()) + \
                          'Completed: {:3.0f}% ({:5d} / {:5d})'.
                          format(self.gs_progress * 100, self.n_completed, n_trials))
            if gs_progress >= 1: break
            self.study.optimize(func, n_trials=1, **kwargs)
        self.post_process()


def get_trial_default():
    num_fields = optuna.structs.FrozenTrial._field_types.__len__()
    assert num_fields in (10, 11, 12)
    if num_fields == 12: # possible future version
        return optuna.structs.FrozenTrial(
            None, None, None, None, None, None, None, None, None, None, None, None)
    elif num_fields == 11: # version 0.9.0 or later
        return optuna.structs.FrozenTrial(
            None, None, None, None, None, None, None, None, None, None, None)
    elif num_fields == 10: # version 0.8.0 or prior
        return optuna.structs.FrozenTrial(
            None, None, None, None, None, None, None, None, None, None)

def str_list(input_list):
    """ Convert all the elements in a list to str
    Args:
        input_list: list of any elements
    Returns: list of string elements
    """
    return ['{}'.format(e) for e in input_list]


import math

from optuna.pruners import BasePruner
from optuna.storages import BaseStorage  # NOQA
from optuna.structs import TrialState


class RepeatPruner(BasePruner):
    """ Prune if the same parameter set was found in Optuna database
        Coded based on source code of MedianPruner class at
        https://github.com/pfnet/optuna/blob/master/optuna/pruners/median.py
    """
    def prune(self, storage, study_id, trial_id, step):
        # type: (BaseStorage, int, int, int) -> bool
        """Please consult the documentation for :func:`BasePruner.prune`."""

        n_trials = storage.get_n_trials(study_id, TrialState.COMPLETE)

        if n_trials == 0:
            return False

        trials = storage.get_all_trials(study_id)
        assert storage.get_n_trials(study_id, TrialState.RUNNING)
        assert trials[-1].state == optuna.structs.TrialState.RUNNING
        completed_params_list = \
            [t.params for t in trials \
             if t.state == optuna.structs.TrialState.COMPLETE]
        if trials[-1].params in completed_params_list:
            return True

        return False
