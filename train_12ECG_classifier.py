#!/usr/bin/env python

import logging
import numpy as np
from time import strftime, localtime
from pandas import concat, DataFrame, read_csv
from os import listdir
from os.path import join, isfile, isdir
from scipy.io import loadmat
from typing import Any, Dict, List, Tuple

from get_12ECG_features import get_12ECG_features, init_12ECG_features, get_12ECG_metadata
from XGB import xgb_train, xgb_test, grid_search, resample_train_data, load_params

log_path = f"/logs/train_{strftime('%Y%m%d_%H%M', localtime())}.log"
logging.basicConfig(filename = log_path, filemode = 'w', level = logging.INFO,
                    format = '%(asctime)s %(levelname)s %(message)s')

def load_mat_data(hea_file: str) -> np.ndarray:
    """
    Loads data from header file and corresponding .mat file

    Params
    ------
    header_file
        full path to header .hea file for the sample being loaded

    Returns
    ------
    a tuple containing the raw 12-lead ECG signal in a np.array of shape
    (12, N) and dtype np.float64 for a signal of length N, and the list
    of lines contained in the header file

    IOError
        if corresponding .mat file does not exist
    """
    mat_file = hea_file.replace('.hea', '.mat')

    if not isfile(mat_file):
        raise IOError(f'File {mat_file} not found')

    data = loadmat(mat_file)
    data = np.asarray(data['val'], dtype = np.float64)

    return data


def get_equivalent_conds(
        dx_file: str = 'dx_mapping_scored.csv'
    ) -> List[Tuple[str, str]]:
    """
    Reads local file 'dx_mapping_scored.csv' and returns equivalent
    conditions from column 'Notes', which has the form:

        We score 427172004 and 17338001 as the same diagnosis.

    Returns
    ------
    a list of 2-tuples with each pair of equivalent conds as strings

    Raises
    ------
    IOError
        when the dx_mapping_scored.csv file is missing
    """
    if not isfile(dx_file):
        raise IOError(f'No file found {dx_file}')

    dx_csv = read_csv(dx_file)
    notes = [ line.split()
                for line in dx_csv.Notes.dropna().unique().astype(str) ]
    equiv = [ ( words[2], words[4] ) for words in notes ]

    return equiv


def get_conds(
        headers:     List[List[str]],
        only_scored: bool = False,
        no_equiv:    bool = False,
        dx_file:     str  = 'dx_mapping_scored.csv'
    ) -> List[str]:
    """
    Extract all unique SNOMED codes for all conditions found in the
    header files of a set of samples

    Params
    ------
    headers
        a list of the contents (list of lines) of each header file
    only_scored
        return condition SNOMED codes only if they are scored
    no_equiv
        remove condition SNOMED codes which are scored as the same
    dx_file
        If any bool param is set, requires file 'dx_mapping_scored.csv',
        to extract the codes for the scored conditions from column

    Returns
    ------
    List of unique SNOMED codes

    Raises
    ------
    IOError
        when the dx_mapping_scored.csv file is missing
    """
    all_conds = set()
    for header in headers:
        conds_str = get_12ECG_metadata(header)['conds']
        conds     = conds_str.split(',')
        for c in conds:
            all_conds.add(c.strip())

    if only_scored or no_equiv:
        if not isfile(dx_file):
            raise IOError(f'No file found {dx_file}')

        if only_scored:
            dx_csv = read_csv(dx_file)
            all_conds = set(dx_csv['SNOMED CT Code'].astype(str).values)

        if no_equiv:
            equiv = get_equivalent_conds(dx_file)

            for _, cond in equiv:
                all_conds.remove(cond)

    return sorted(all_conds)


def init_labels(
        conds_str: str,
        all_conds: List[str]
    ) -> DataFrame:
    """
    Create the label with the diagnoses for a given sample to be used
    by the classifier

    Params
    ------
    conds
        comma-separated string listing SNOMED codes for a given sample,
        e.g.
            '63593006,59118001'

    all_conds
        ordered list of all unique SNOMED codes in the dataset,
        e.g.
            [ '713427006', '59118001', '284470004', '63593006' ]

    Returns
    ------
    np.ndarray containing columns identifying all conditions and values
    of 0 and 1, 1 if the sample has been labelled with the condition
    e.g.
                 0            1           0           1
    """
    labels = np.zeros(len(all_conds))

    for c in conds_str.split(','):
        # for training with only scored conditions, we need to check
        # if c is in all_conds first
        if (c in all_conds):
            i = all_conds.index(c.rstrip()) # Only use first positive index
            labels[i] = 1

    return labels


def get_training_data(indir: str) -> Tuple[DataFrame, DataFrame]:
    """
    Load training data from input directory, extract features and build
    dataframes for the features and labels, producing the data structure
    required for training

    Params
    ------
    indir
        load training data from this folder

    Returns
    ------
    dataframes used by classifier, with the subject (e.g. 'A0001') as index
    and the feature names and SNOMED codes as column names respectively
    """
    # Read all headers first
    logging.info('Loading header data...')
    hea_files = []
    for fname in listdir(indir):
        path = join(indir, fname)
        if not path.startswith('.') and path.endswith('hea') and isfile(path):
            hea_files.append(path)
    headers = []
    for hea in hea_files:
        with open(hea, 'r') as f:
            headers.append(f.readlines())

    # Get conditions, filter only for scored ones and removing equivalent
    logging.info('Getting scored conditions, removing equivalent ones')
    all_conds = get_conds(headers, only_scored = True, no_equiv = True)

    # save subjects, features and labels into lists
    subjects, features, labels = [], [], []

    logging.info('Processing raw recoding files...')

    for i, hea in enumerate(hea_files):
        logging.info(f'Processing file {i}:{hea}')
        data = load_mat_data(hea)

        # extract features and save in df with feature names as columns
        feat_dict = get_12ECG_features(data, headers[i])
        features.append(feat_dict)

        meta_dict = get_12ECG_metadata(headers[i])
        subjects.append(meta_dict['subject'])
        del meta_dict['subject']
        labels.append(init_labels(meta_dict['conds'], all_conds))

    # create dataframe of features and labels
    logging.info('Done loading files and extracting features.')

    dfeats = DataFrame(features)
    dfeats.index = subjects
    dlabel = DataFrame(labels)
    dlabel.index   = subjects
    dlabel.columns = all_conds

    return dfeats, dlabel


def train_12ECG_condition_classifier(
        cond:     str,
        dfeats:   DataFrame,
        dlabs:    DataFrame,
        outdir:   str,
        optimise: bool = False
    ):
    """
    Train a single XGBoost classifier for a condition

    Params
    -----
    cond
        condition SNOMED code to train for
    dfeats
        features dataframe produced by `init_12ECG_features` with float
        values for each sample, columns signifying feature names, and
        indexed with the subject ID
    dlabs
        single-column dataframe, containing 0 and 1 labels for each
        sample to represent sample's diagnosis with the condition
    outdir
        directory to save the model in
    optimise
        if true, runs grid search to choose params, otherwise load pre-
        optimised params from local file

    Returns
    ------
    None

    Side-effects
    ------
    Save XGBoost model in output directory as {cond}.model
    """
    # first resample and check data encoding
    x, y = resample_train_data(dfeats, dlabs, 'under')

    if optimise:
        logging.info(f'Optimising parameters.')
        # optimise parameters with grid search and cross-validation
        params = grid_search(cond, x, y, k = 2, save = True)
    else:
        logging.info(f'Loading pre-optimised parameters.')
        params = load_params(cond)

    # train again using above params and a small subset of data for testing
    logging.info(f'Training model...')
    model = xgb_train(x, y, params, test_size = 0.1)

    # save trained model
    logging.info(f'Saving model...')
    model.save_model(f'{outdir}/{cond}.model')


def train_12ECG_classifier(indir: str, outdir: str):
    """
    Train XGBoost classifier, function used in `driver.py`

    Params
    ------
    indir
        load training data from this folder
    outdir
        save trained models in this folder

    Returns
    ------
    None

    Side-effects
    ------
    Saves a binary model file for each classifier

    """

    dfeats, dlabels = get_training_data(indir)

    conds = dlabels.columns.values.astype(str)

    for cond in conds:
        logging.info(f'Training XGB for {cond}')
        dlabel = dlabels[cond].astype(int)
        train_12ECG_condition_classifier(cond, dfeats, dlabel, outdir)
