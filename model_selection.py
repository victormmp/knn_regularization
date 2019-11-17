import csv
import math

import pandas as pd
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from random import randint
from config import Config
from collections import Counter
import psutil

from build_features import load_moons

########################################################################################################################
# GLOBAL VARIABLES                                                                                                     #
########################################################################################################################

LOG = Config(logger_name='Main').Logger

########################################################################################################################
# DATASET                                                                                                              #
########################################################################################################################

x_train, x_test, y_train, y_test = load_moons()

########################################################################################################################
# GET BEST CONFIG                                                                                                      #
########################################################################################################################

def run_model_selection(x_train, y_train, verbose: bool = False):

    params = [
        {
            'hidden_layer_sizes': [100, 500],
            'activation': ['relu', 'tanh', 'logistic'],
            'max_iter': [1000],
            'solver': ['sgd'],
            'alpha': [0],
            'batch_size': [1, 10, 100, 200],
            'learning_rate_init': [0.001, 0.01, 0.05],
            'tol': [1e-6, 1e-4, 1e-12]
        },
        {
            'hidden_layer_sizes': [100, 500],
            'activation': ['relu', 'tanh', 'logistic'],
            'max_iter': [1000],
            'solver': ['adam'],
            'alpha': [0],
            'batch_size': [1, 10, 100, 200],
            'learning_rate_init': [0.001, 0.01, 0.05],
            'tol': [1e-6, 1e-4, 1e-12]
        }
    ]

    clf = GridSearchCV(
        MLPClassifier(), 
        param_grid=params, 
        verbose=10 if verbose else 0, 
        n_jobs=-2)

    clf.fit(x_train, y_train)

    LOG.info(f'Finished grid search. Best Params:')
    print(clf.best_params_)

    return clf


def load_model(params= None, n_epochs = 50000, verbose=False):
    if params is None:
        params = {
        'hidden_layer_sizes': (100,),
        'solver': 'sgd',
        'activation': 'relu',
        'alpha': 0,
        'max_iter': n_epochs,
        'tol':1e-12,
        'verbose': True,
        'batch_size': 100,
        'verbose': verbose,
        'n_iter_no_change': n_epochs,
        'learning_rate_init': 0.05
        }
    else:
        params['alpha'] = 0
        params['max_iter'] = n_epochs
        params['verbose'] = verbose
        params['n_iter_no_change'] = n_epochs
    
    model = MLPClassifier(**params)
    return model

if __name__ == '__main__':
    run_model_selection(x_train, y_train)
