#!/usr/bin/env python
# coding: utf-8


from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split, cross_validate, ShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    roc_curve, 
    f1_score, 
    roc_auc_score, 
    precision_score, 
    recall_score
)
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from os.path import join
import seaborn as sns
import pandas as pd
import logging

from joblib import Parallel, delayed
import warnings

from build_features import load_moons, load_concentric
from utils import euclidian, plot_plain_separator
from model_selection import run_model_selection, load_model

LOG_LEVEL = logging.INFO
formatter = logging.Formatter('%(asctime)s - [%(levelname)s] - %(name)s - %(message)s')
logging.root.setLevel(LOG_LEVEL)
STREAM = logging.StreamHandler()
STREAM.setLevel(LOG_LEVEL)
STREAM.setFormatter(formatter)
LOG = logging.getLogger(__name__)
LOG.setLevel(LOG_LEVEL)
LOG.addHandler(STREAM)

warnings.filterwarnings('ignore')

results = []

NUMBER_OF_RUNS = 30
RESULT_CSV_NAME = 'result_moons.csv'

def run_instance(run):
    LOG.info(f"Running sample {run} from {NUMBER_OF_RUNS} of Moons code")

    # Generate Test data
    LOG.info(f'[RUN {run}] Generating test data.')
    x1, x2, y1, y2 = load_moons(noise=0.2, samples=2000)
    x_test = np.vstack([x1, x2])
    y_test = np.concatenate([y1, y2])
    data_test = pd.DataFrame(x_test, columns=['x1', 'x2'])
    data_test['class'] = y_test

    # Generate Train data
    LOG.info(f'[RUN {run}] Generating train data.')
    x1, x2, y1, y2 = load_moons(noise=0.35, samples=200)
    x_train = np.vstack([x1, x2])
    y_train = np.concatenate([y1, y2])
    data_train = pd.DataFrame(x_train, columns=['x1', 'x2'])
    data_train['class'] = y_train

    # Loading model
    LOG.info(f'[RUN {run}] Loading and training model.')
    mlp = load_model()
    mlp.fit(x_train, y_train)

    # Generate scores
    LOG.info(f'[RUN {run}] Generating f1 score.')
    predicted = mlp.predict(x_test)
    f1_before = f1_score(y_pred=predicted, y_true=y_test)
    norm_before = np.linalg.norm(mlp.coefs_[0]), np.linalg.norm(mlp.coefs_[1])

    # Run relabeling
    LOG.info(f'[RUN {run}] Running relabeling.')

    def norm_distance(dist):
        return 1 - normalize(dist).ravel()

    knn = KNeighborsClassifier(n_neighbors=10, weights=norm_distance)

    y_classes = []
    for index in range(x_train.shape[0]):
        x_t = np.delete(x_train, index, 0)
        y_t = np.delete(y_train, index, 0)
        knn.fit(x_t, y_t)
        y_classes.append(knn.predict([x_train[index]])[0])

    errors = y_train - y_classes

    wrong_classes = np.where(errors != 0)[0]
    y_train_2 = y_train.copy()

    for i in wrong_classes:
        y_train_2[i] = 0 if y_train[i] == 1 else 1

    # Running training on relabeled dataset
    LOG.info(f'[RUN {run}] Runing training on relabeled dataset.')
    mlp = load_model()
    mlp.fit(x_train, y_train_2)

    # Generate scores
    LOG.info(f'[RUN {run}] Generating f1 score for relabeled dataset.')
    predicted = mlp.predict(x_test)
    f1_after = f1_score(y_pred=predicted, y_true=y_test)
    norm_after = np.linalg.norm(mlp.coefs_[0]), np.linalg.norm(mlp.coefs_[1])

    return {
        'f1_score_before': f1_before,
        'f1_score_after': f1_after,
        'norm_1_before': norm_before[0],
        'norm_1_after': norm_after[0],
        'norm_2_before': norm_before[1],
        'norm_2_after': norm_after[1]   
    }

    LOG.info(f'[RUN {run}] Finished run.')

results = Parallel(n_jobs=-2)(delayed(run_instance)(run) for run in range(NUMBER_OF_RUNS))

df = pd.DataFrame(results)
df.to_csv(RESULT_CSV_NAME)

LOG.info(f'Results saved in file {RESULT_CSV_NAME}')