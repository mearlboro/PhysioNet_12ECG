#!/usr/bin/env python

import numpy as np
from pandas import DataFrame
from typing import Any, Dict, List

from info_tools import get_info_measures


def get_12ECG_metadata(
        header: List[str]
    ) -> Dict[str, Any]:
    """
    Given the contents of a .hea file, extract relevant data about the
    signal and the patient as well as the label(s) for that sample

    Params
    ------
    header
        the contents of a .hea file as a list of strings representing
        each line in the file, for example:

        A0003 12 500 5000 12-May-2020 12:33:59
        A0003.mat 16+24 1000/mV 16 0 43 64 0 I
        A0003.mat 16+24 1000/mV 16 0 8 42 0 II
        A0003.mat 16+24 1000/mV 16 0 -35 -29 0 III
        A0003.mat 16+24 1000/mV 16 0 -25 -56 0 aVR
        A0003.mat 16+24 1000/mV 16 0 38 -6 0 aVL
        A0003.mat 16+24 1000/mV 16 0 -14 12 0 aVF
        A0003.mat 16+24 1000/mV 16 0 32 23 0 V1
        A0003.mat 16+24 1000/mV 16 0 65 38 0 V2
        A0003.mat 16+24 1000/mV 16 0 39 23 0 V3
        A0003.mat 16+24 1000/mV 16 0 42 50 0 V4
        A0003.mat 16+24 1000/mV 16 0 13 16 0 V5
        A0003.mat 16+24 1000/mV 16 0 -14 24 0 V6
        #Age: 81
        #Sex: Female
        #Dx: 164889003,59118001
        #Rx: Unknown
        #Hx: Unknown
        #Sx: Unknown

    Returns
    ------
    the relevant metadata in the form of a dictionary, for example

        {'subject': 'A0957', 'sample_rate': 500, 'age': 64, 'female': 0,
         'male': 1, 'conds': '164889003,59118001'}
    """
    meta_dict = dict()

    # extract subject ID and sample rate from the first line
    tmp_hea = header[0].split(' ')
    meta_dict['subject']     = tmp_hea[0]
    meta_dict['sample_rate'] = int(tmp_hea[2])

    for line in header:
        if ': ' in line:
            value = line.split(': ')[1].strip()

        if line.startswith('#Age'):
            # negative age will be replaced with average value after preprocessing
            meta_dict['age'] = int(value if value != 'NaN' else -1)
        elif line.startswith('#Sex'):
            # one-hot encoding: categorical data is used as booleans
            meta_dict['female'] = 1 if value == 'Female' else 0
            meta_dict['male']   = 1 if value == 'Male'   else 0
        elif line.startswith('#Dx'):
            meta_dict['conds'] = value

    return meta_dict


def get_12ECG_features(
        data: np.ndarray,
        header: List[str]
    ) -> Dict[str, np.float64]:
    """
    Given the contents of a .hea file, extract relevant data about the
    signal and the patient as well as the label(s) for that sample

    Params
    ------
    data
        raw 12-lead ECG signal in a np.array of shape (12, N) and dtype
        np.float64 for a signal of length N over 12 leads
    header
        the contents of a .hea file containing metadata about the 12-lead
        signal as a list of strings representing each line in the file

    Returns
    ------
    dictionary with the feature names as key and float values for feats
    which will be properly typed for xgboost in the preprocessing phase
    """
    meta_dict = get_12ECG_metadata(header)
    fs = meta_dict['sample_rate']

    # for very long recordings, only extract features from the first 60 sec
    T = int(60 * fs)
    data = data.T
    if (len(data) > T):
        data = data[:T]

    # construct feature dictionary
    feats_dict = dict()

    info_dict = get_info_measures(data, fs)
    feats_dict.update(info_dict)

    # include patient metadata as well as subject ID (same as file name)
    feats_dict.update({ k: meta_dict[k]
                        for k in ['age', 'male', 'female', 'subject'] })

    return feats_dict


def init_12ECG_features(feats_dict: Dict[str, np.float64]) -> DataFrame:
    """
    Given a dictionary with feature names as keys and features from a
    single recording as values, initialises a pandas dataframe in the
    format needed to be used as input by the XGBoost model

    Params
    ------
    feats_dict
        feature dictionary

    Returns
    ------
    DataFrame with features
    """
    # subject will be the index of dataframe, remove from dict
    subject = feats_dict['subject']
    del feats_dict['subject']

    df = DataFrame(
        np.asarray(list(feats_dict.values()), dtype = np.float64).reshape(1, -1),
        columns = list(feats_dict.keys()), index = [subject])

    return df

