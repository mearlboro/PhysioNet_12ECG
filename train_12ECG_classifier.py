#!/usr/bin/env python

import numpy as np
from pandas import concat, DataFrame, read_csv
from os import listdir
from os.path import join, isfile, isdir
from scipy.io import loadmat
from typing import Any, Dict, List, Tuple

from get_12ECG_features import get_12ECG_features, init_12ECG_features, get_12ECG_metadata


def load_mat_data(hea_file: str) -> np.ndarray:
    """
    Loads data given header file for the corresponding .mat file

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
        subject: str,
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
    DataFrame containing columns identifying all conditions and values
    of 0 and 1, 1 if the sample has been labelled with the condition
    e.g.
           713427006    59118001    284470004   63593006
           0            1           0           1
    """
    labels = np.zeros(len(all_conds))

    for c in conds_str.split(','):
        i = all_conds.index(c.rstrip()) # Only use first positive index
        labels[i] = 1

    df = DataFrame(
        labels.reshape(1, -1), columns = all_conds, dtype = int,
        index = [subject])

    return df


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

    # Open and read all headers first
    hea_files = []
    for fname in listdir(indir):
        path = join(indir, fname)
        if not path.startswith('.') and path.endswith('hea') and isfile(path):
            hea_files.append(path)

    headers = []
    for hea in hea_files:
        with open(hea, 'r') as f:
            headers.append(f.readlines())

    # Extract unique conditions from the headers
    all_conds = get_conds(headers)

    # all features and labels will be added to dataframes
    dlabel = DataFrame(columns = all_conds)
    dfeats = DataFrame()

    for i, hea in enumerate(hea_files):
        data = load_mat_data(hea)

        # extract features and save in df with feature names as columns
        fdf = init_12ECG_features(get_12ECG_features(data, headers[i]))
        dfeats = dfeats.append(fdf)

        # arrange labels into df with SNOMED codes as columns
        meta_dict = get_12ECG_metadata(headers[i])
        ldf = init_labels(meta_dict['subject'], meta_dict['conds'], all_conds)
        dlabel = dlabel.append(ldf)

    return dfeats, dlabel


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

    dfeats, dlabel = get_training_data(indir)

    print(dfeats.head())
    print(dlabel.head())
    # TODO: training code here

