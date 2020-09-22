# -*- coding: utf-8 -*-

import numpy as np
from scipy.signal import butter, filtfilt

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
