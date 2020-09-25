#!/usr/bin/env python

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import multiprocessing as mp
import numpy as np
from os.path import isfile
from pandas import DataFrame, read_csv
from sklearn.metrics import precision_score, recall_score, fbeta_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Any, Dict, List, Tuple
from xgboost import Booster, DMatrix, train, XGBClassifier


default_steps  = 5000
default_thre   = 0.9
default_params = {
    'max_depth'  : 5,
    'eta'        : 0.1,
    'gamma'      : 0,
    'lambda'     : 0,
    'objective'  : 'binary:logistic',
    'eval_metric': 'aucpr',
}
early_stopping = 250


def resample_train_data(
        x:        DataFrame,
        y:        DataFrame,
        approach: str = 'under'
    ) -> Tuple[DataFrame, DataFrame]:
    """
    Randomly delete examples from the majority class or duplicate
    examples from the minority class

    Params
    ------
    approach: { 'over', 'under' }
        whether to oversample or undersample - defaults to undersampling
    x
        features dataframe as produced by `init_12ECG_features`, with
        float values for each sample, columns signifying feature names,
        and indexed with the subject ID
    y
        single-column dataframe, containing 0 and 1 labels for each
        sample to represent sample's diagnosis with the condition

    Returns
    ------
    subset dataframes of x and y containing the same amount of samples
    labelled as 0 and 1

    Raises
    ------
    ValueError
        if called with an unsupported approach
    """
    if approach not in { 'over', 'under' }:
        raise ValueError(f'Unsupported approach for resampling: {approach}')

    if approach == 'under':
        resampler = RandomUnderSampler()
    else:
        resampler = RandomOverSampler()

    final_x, final_y = resampler.fit_resample(x, y)

    return final_x, final_y


def fbeta(
        pred,
        dmatrix
    ) -> Tuple[str, float]:
    """
    Scoring function to be optimised by model. Only predictions with a
    probability higher than the threshold get classified as the predicted
    condition.

    Params
    ------
    pred
        an array of predictions
    dmatrix
        DMatrix with the samples and true labels
    """
    outs = 1 * (pred > default_thre)
    labels = dmatrix.get_label()
    score = fbeta_score(labels, outs, beta = 2, pos_label = 1, average = 'binary')
    return 'f2', score


def xgb_train(
        x:         DataFrame,
        y:         DataFrame,
        params:    Dict[str, Any] = default_params,
        test_size: float          = 0.1,
    ) -> Booster:
    """
    Train a XGBoost classifier with params on given data, splitting the
    dataset in training and testing data. The small testing set is used
    by XGB for computing measures and optimising the ensemble itself
    using early stopping.

    Params
    -----
    x
        features dataframe as produced by `init_12ECG_features`, with
        float values for each sample, columns signifying feature names,
        and indexed with the subject ID
    y
        single-column dataframe, containing 0 and 1 labels for each
        sample to represent sample's diagnosis with the condition
    params
        dict of params for XGBoost, with defaults
    test_size
        float value for partitioning the dataset into train and test
    """
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size = test_size)

    D_train = DMatrix(x_train, label = y_train)
    D_test  = DMatrix(x_test,  label = y_test)
    evallist = [(D_test, 'eval')]

    model = train(
                params, D_train, default_steps, feval = fbeta,
                maximize = True, evals = evallist, verbose_eval = 100,
                early_stopping_rounds = early_stopping)

    # print out some stats from the testing set
    xgb_test(model, x_test, y_test)

    return model


def xgb_test(
        model: Booster,
        x_test: DataFrame,
        y_test: DataFrame
    ) -> Tuple[float, float, float, float, List[int]]:
    """
    Record model characteristics, such as prediction probabilities, scores
    and feature importances

    Params
    ------
    x_test
        features dataframe as produced by `init_12ECG_features`, with
        float values for each sample, columns signifying feature names,
        and indexed with the subject ID
    y_test
        single-column dataframe, containing 0 and 1 labels for each
        sample to represent sample's diagnosis with the condition
    """
    # get predictors from column names
    predictors = x_test.columns

    D_test = DMatrix(x_test, label = y_test)
    test_probs  = model.predict(D_test)

    # Filter all results higher than classification threshold
    outputs = 1 * (test_probs > default_thre)

    # Compute importance of each predictor in this model
    importances = []
    for pred in predictors:
        if pred in model.get_fscore().keys():
            importances.append(model.get_fscore()[pred])
        else:
            importances.append(0)

    # Compute scores
    labels = D_test.get_label()

    # precision = tp / (tp + fp)
    prec = precision_score(
        labels, outputs, pos_label=1, average='binary', zero_division='warn')

    # recall = tp / (tp + fn)
    rec = recall_score(
        labels, outputs, pos_label=1, average='binary', zero_division='warn')

    # accuracy = tp + tn / (tp + fp + tn + fn)
    acc = np.sum(outputs == labels) / len(labels)

    # F_beta = (1 + beta^2) * prec * rec / (beta^2 * prec + rec)
    f2 = fbeta_score(
        labels, outputs, beta = 2, pos_label = 1, average = 'binary',
        zero_division = 'warn')

    print(f'Precision: {prec}, Recall: {rec}, Accuracy: {acc}, F2 Score: {f2}')
    print(f'Feature importance: {zip(predictors, importances)}')
    return prec, rec, acc, f2, importances


def grid_search(
        cond: str,
        x:    DataFrame,
        y:    DataFrame,
        k:    int  = 3,
        save: bool = True
    ) -> Dict[str, Any]:
    """
    Parameter sweep using GridSearchCV which also performs KFold cross
    validation

    Params
    -----
    cond
        condition SNOMED code
    x
        features dataframe as produced by `init_12ECG_features`, with
        float values for each sample, columns signifying feature names,
        and indexed with the subject ID
    y
        single-column dataframe, containing 0 and 1 labels for each
        sample to represent sample's diagnosis with the condition
    k
        number of folds
    save
        save params to file
    pfile
        CSV file containing the params compatible with pandas, with the
        param names as columns and condition SNOMED code as index
    """

    print(f'Running parameter optimisation for {cond} with Grid Search')

    # params to search through
    param_grid = {
        'max_depth': [10, 15],
        'eta': [0.1, 0.01],
        'gamma': [0, 0.25, 0.5],
        'lambda': [0, 1.0, 10.0]
    }

    # will optimise f2 score
    f2_score = make_scorer(fbeta_score, beta=2)

    optimal_params = GridSearchCV(
        estimator = XGBClassifier(
		objective='binary:logistic',
                scale_pos_weight = 1,
                subsample = 1,
	    ),
        param_grid = param_grid,
        scoring = f2_score,
	    # parallelise for number of cores
        n_jobs = mp.cpu_count(),
	    # k-fold cross validation
        cv = k,
        verbose = 1,
    )

    optimal_params.fit(x, y)

    params = optimal_params.best_params_

    if (save):
        save_params(cond, params)

    print(optimal_params.best_params_)
    return optimal_params.best_params_


def save_params(
        cond:        str,
        params:      Dict[str, Any],
        file_params: str = 'model/params.csv'
    ):
    """
    Save set of params for cond to file

    Params
    ------
    cond
        SNOMED code of condition params to be fetched
    params
        dict of params to save
    file_params
        path to CSV file containing the params, in the form

             , SNOMED Code, param1, param2, ..., paramN
            0,   270492004,     10,    0.1           18
            1,   164889003,     10,    0.2           20
            2,   164890007,     15,    0.1           10

    Returns
    ------
    None

    Side-effects
    ------
    Creates or modifies params file
    """

    if isfile(file_params):
        # if the file exists load it and update it
        params_dict = load_params(cond)
        params_dict[cond] = params
        # turn into dataframe to write back to file
        params_df = DataFrame(params).T
    else:
        # create single entry dataframe otherwise
        params_df = DataFrame({cond:params}).T

    # add numeric index and make SNOMED code the first column
    cols = params_df.columns
    cols = ['SNOMED CT Code'] + cols
    params_df['SNOMED CT Code'] = params_df.index
    params_df.reindex(columns = cols)
    params_df.index = range(len(params_df))

    params_df.to_csv(file_params)


def load_params(
        cond:        str,
        file_params: str = 'model/params.csv'
    ) -> Dict[str, Any]:
    """
    Get params produced by optimisation from a CSV file, produce a dict
    of dicts indexed by the string condition name that fits into XGBoost
    params, and return the entry for given condition

    Params
    ------
    cond
        SNOMED code of condition params to be fetched
    file_params
        path to CSV file containing the params, in the form

             , SNOMED Code, param1, param2, ..., paramN
            0,   270492004,     10,    0.1           18
            1,   164889003,     10,    0.2           20
            2,   164890007,     15,    0.1           10
    """

    # read the dataframe from CSV and turn into a list of dicts
    param_df = read_csv(file_params, index_col=0)
    param_dicts = [ param_df[columns].loc[i].to_dict() for i in range(len(param_df)) ]

    for i in range(len(param_dicts)):
        # cast to int to have correct types for XGB, the rest default to float
        if ('max_depth' in columns):
            param_dicts[i]['max_depth'] = int(param_dicts[i]['max_depth'])
        # add eval metric and objective
        param_dicts[i]['eval_metric'] = 'aucpr'
        param_dicts[i]['objective']   = 'binary:logistic'

    params = { str(cond):params for (cond, params) in zip(param_df['SNOMED Code'], param_dicts) }

    return params[cond]


