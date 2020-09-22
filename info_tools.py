#!/usr/bin/env/python

"""
This file includes all the tools required to compute infomation theory
measures, total correlation, dual total corellation, O-Information and
S-information.

Uses the Java Infodynamics package
http://jlizier.github.io/jidt/
"""

from jpype import startJVM, isJVMStarted, shutdownJVM, getDefaultJVMPath, JPackage
import numpy as np
from os import getcwd
from os.path import join
from scipy.signal import butter, filtfilt
from typing import Dict, List


def start_jpype():
    """
    Initialise and start a Java virtual machine unless it has already
    been started

    Side-effects
    ------
    Starts a singleton JVM to be used by all function calls
    """
    jarLocation = join(getcwd(),"infodynamics.jar")

    max_heap_size = 1024

    if not isJVMStarted():
        # the "-Xmx" option specifies maximum heap size
        startJVM(getDefaultJVMPath(), '-ea',
                 f'-Xmx{max_heap_size}m',
                 f'-Djava.class.path={jarLocation}')

def stop_jpype():
    """
    Shutdown a Java virtual machine if it's running

    Side-effects
    ------
    Shuts down singleton JVM
    """
    if isJVMStarted():
        shutdownJVM()


def Info_Calculator(
        data:    np.ndarray,
        measure: str
    ) -> np.float64:
    """
    Compute total correlation or dual total correlation for given data

    Params
    ------
    measure: { 'TC', 'DTC' }
        the information theoretic measure to be computed
    data
        raw 12-lead ECG signal in a np.array of shape (L, N) and dtype
        np.float64 for a signal of length N over L leads

    Returns
    ------
    average of given info measure computed over the given leads

    Raises
    ------
    ValueError
        if data is 1-dimensional or measure is incorrect
    """
    if (data.shape[0] < 2):
        raise ValueError('Cannot compute correlations for 1D time series')
    if (measure not in ('TC', 'DTC')):
        raise ValueError('Supported measures are "TC" and "DTC"')

    start_jpype()

    if (measure is 'TC'):
        calc = (JPackage("infodynamics.measures.continuous.kraskov")
                .MultiInfoCalculatorKraskov1)()
    else:
        calc = (JPackage("infodynamics.measures.continuous.kraskov")
                .DualTotalCorrelationCalculatorKraskov)()

    N = int(data.shape[1])
    calc.initialise(N)
    calc.setObservations(data.tolist())
    return calc.computeAverageLocalOfObservations()


def butter_bandpass_filter(
        data:    np.ndarray,
        fs:      int,
        lowcut:  float = 0.5,
        highcut: float = 100,
        order:   int   = 1,
        axis:    int   = 0,
    ):
    """
    Apply a Butterworth band pass filter on the signal to remove both
    high and low frequencies

    Params
    ------
    data
        numpy 2D array of shape (12, N) with the 12-lead signal
    fs, lowcut, highcut, order, axis
        as requited by butter and filtfilt

    Returns
    ------
    numpy 2D array of shape (12, N) with the cleaned 12-lead signal
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis)
    return filtered_data


def get_info_measures(
        data: np.ndarray,
        sample_rate: int,
    ) -> Dict[str, np.float64]:
    """
    Compute information-theoretic correlation measures between leads

    The leads are split into four groups: inferior, lateral, sptal and
    anterior.

    Params
    ------
    data
        numpy 2D array of shape (12, N) with the 12-lead signal
    sample_rate
        as extracted from the header file for the current recording

    Returns
    ------
    dictionary of features
    """
    # Group leads
    inf = [1, 2, 5]
    lat = [0, 4, 10, 11]
    sep = [6, 7]
    ant = [8, 9]

    # first filter the data
    data = butter_bandpass_filter(data, sample_rate)

    info_dict = dict()
    info_dict['tc_inf'  ] = Info_Calculator(data[:,inf], 'TC')
    info_dict['dtc_inf' ] = Info_Calculator(data[:,inf], 'DTC')
    info_dict['oinf_inf'] = info_dict['tc_inf'] - info_dict['dtc_inf']
    info_dict['sinf_inf'] = info_dict['tc_inf'] + info_dict['dtc_inf']
    info_dict['tc_lat'  ] = Info_Calculator(data[:,lat], 'TC')
    info_dict['dtc_lat' ] = Info_Calculator(data[:,lat], 'DTC')
    info_dict['oinf_lat'] = info_dict['tc_lat'] - info_dict['dtc_lat']
    info_dict['sinf_lat'] = info_dict['tc_lat'] + info_dict['dtc_lat']
    info_dict['tc_sep'  ] = Info_Calculator(data[:,sep], 'TC')
    info_dict['tc_ent'  ] = Info_Calculator(data[:,ant], 'TC')

    return info_dict
