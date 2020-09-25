#!/usr/bin/env python

import numpy as np
from os import listdir
from os.path import join, isfile
from pandas import DataFrame
from typing import Any, Dict, List, Tuple
from xgboost import Booster, DMatrix

from get_12ECG_features import get_12ECG_features, init_12ECG_features

def run_12ECG_classifier(
        data: np.ndarray,
        header_data: List[str],
        models: Tuple[Dict[str, Booster], Dict[str, np.float64]]
    ) -> Tuple[List[int], List[np.float64], List[str]]:
    """
    Given the raw 12-lead ECG signal and the header data for one sample,
    as well as the model, will run the classifier over the data and give
    a prediction

    Params
    ------
    data
        raw 12-lead ECG signal in a np.array of shape (12, N) and dtype
        np.float64 for a signal of length N
    header_data
        string metadata describing voltages and other properties of
        each lead as well as patient metadata and label
    model
        tuple for for models with two dictionaries as returned by
        `load_12ECG_model`

    Returns
    ------
    three ordered lists containing labels for each class (0 or 1),
    predictions for each class (float in between 0 and 1), and the
    actual classes (conditions SNOMED codes)
    """
    xgbs, thres = models

    # classes (conditions) are the keys for the dictionaries
    conds = list(xgbs.keys())

    # extract features from data and construct DMatrix required by model
    feats = DMatrix(init_12ECG_features(get_12ECG_features(data, header_data)))

    # run each classifier on the features
    preds = [ xgbs[c].predict(feats)[0] for c in conds ]

    # label that class as 1 if prediction for that class is higher than
    # the threshold for that class
    labels = [ 1 * (preds[i] > thres[c]) for (i, c) in enumerate(conds)]

    return labels, preds, conds


def load_12ECG_model(
        input_directory: str
    ) -> Tuple[Dict[str, Booster], Dict[str, float]]:
    """
    Load one XGB model per condition from given input dir and also
    loads classification thresholds from the same dir

    Params
    ------
    input_directory
        relative path to directory to load models from

    Returns
    ------
    a tuple containing two dictionaries with the condition SNOMED
    code as keys, which contain the models as xgboost objects and
    classification thresholds to be used by the models at prediction
    time
    """

    xgbs, thres = dict(), dict()

    # Load models from files in input dir
    files = [ f for f in listdir(input_directory)
	      if isfile(join(input_directory, f)) and '.model' in f ]

    for f in files:
        cond = str(f.split('.')[0])
        model = Booster()
        model.load_model(f'{input_directory}/{f}')
        xgbs[cond] = model

    # Load clas thresholds from file in input dir
    lines = np.loadtxt(f'{input_directory}/thresholds.csv', delimiter = ',',
                       usecols = (0, 1), dtype = np.ndarray)
    thres = { str(k): float(v) for k, v in lines }

    return xgbs, thres
