#!/us/bin/env python
# -*- coding: utf-8 -*-
"""
This file includes functions to extract signal features such as
properties of the QRS complex, the P and T-waves etc

TODO: @Hardik @Max please add comments describing these
"""

import numpy as np
import pywt
from scipy.signal import find_peaks, hilbert, peak_widths, peak_prominences, argrelextrema
from scipy.stats import iqr, skew, kurtosis
from typing import Any, Dict, List

def get_qrs_props(
        qrs,
        fs: float
    ) -> Dict[str, Any]:
    """
    Gets properties of a single QRS complex such as height of the peaks,
    distance between them etc.

    Params
    -----
    qrs

    fs
        sample rate of the time series

    Returns
    -----
    dictionary with feature names as strings and int or float values
    """

    qrs_win = int(0.1 * fs)
    if (qrs[qrs_win] < 0):
        qrs = -1 * qrs

    r_peak = np.argmax(qrs)

    # TODO: define those magic numbers in variables that describe what they are
    # and add comments explaining why they have that value (0.04 , 0.16, also 0.75
    # below on line 68)
    minima0 = argrelextrema(qrs[:qrs_win], np.less)[0]
    vals0 = qrs[minima0]
    if (len(minima0) > 0):
        q_peak = minima0[np.argmin(vals0)]
    else:
        q_peak = int(0.04 * fs)

    minima1 = argrelextrema(qrs[qrs_win:], np.less)[0]
    vals1 = qrs[qrs_win + minima1]
    if (len(minima1) > 0):
        s_peak = qrs_win + minima1[np.argmin(vals1)]
    else:
        s_peak = int(0.16 * fs)

    qrs_dict = dict()
    qrs_dict['height'] = max(qrs) - min(qrs)
    qrs_dict['r_sign'] = np.sign(qrs[qrs_win])

    qrs_dict['qval']      = qrs[q_peak]
    qrs_dict['sval']      = qrs[s_peak]
    qrs_dict['rval']      = qrs[r_peak]
    qrs_dict['qs_height'] = qrs_dict['sval'] - qrs_dict['qval']

    q_width,_,q_beg,_ = peak_widths(-1 * qrs,[q_peak], 0.75)
    s_width,_,_,s_end = peak_widths(-1 * qrs,[s_peak], 0.75)
    r_width,_,_,_     = peak_widths(qrs, [r_peak], 1)

    qrs_dict['q_width'] = q_width[0] / fs
    qrs_dict['r_width'] = r_width[0] / fs
    qrs_dict['s_width'] = s_width[0] / fs

    q_beg = int(q_beg[0])
    s_end = int(s_end[0])

    qrs_dict['mean']  = np.mean(qrs[q_beg:s_end])
    qrs_dict['std']   = np.std(qrs[q_beg: s_end])
    qrs_dict['width'] = (s_end - q_beg) / fs

    qrs_dict['qrs_iqr']      = iqr(qrs[q_beg:s_end])
    qrs_dict['qrs_iqr_norm'] = qrs_dict['qrs_iqr'] / qrs_dict['mean']

    qrs_dict['skew'] = skew(qrs[q_beg: s_end])
    qrs_dict['kurt'] = kurtosis(qrs[q_beg: s_end])

    qr_dist = r_peak - q_peak
    rs_dist = s_peak - r_peak
    qrs_dict['asymmetry'] = rs_dist/qr_dist

    #props = [height, width, mean, std, qrs_iqr, qrs_iqr_norm, sk,
    #    kurt, qval, rval, sval, q_width, r_width, s_width, asymmetry, qs_height, r_sign]
    #return props, [q_peak - r_peak, s_peak - r_peak], [q_beg - r_peak, s_end - r_peak]
    # TODO: the second and third params can also be included in the dict
    # but i didn't know what names to use
    return qrs_dict


def get_p_props(
        p_wave,
        fs: float
    ) -> Dict[str, Any]:
    """
    Gets properties of a single P wave

    Params
    -----
    p_wave

    fs
        sample rate of the time series

    Returns
    -----
    dictionary with feature names as strings and int or float values
    """
    qrs_win = int(0.1*fs)
    if (p_wave[qrs_win] < 0):
        p_wave = -1 * p_wave

    p_dict = dict()
    p_dict['p_sign'] = np.sign(p_wave[qrs_win])
    p_dict['pval'] = max(p_wave)

    # TODO: explain why the 5 is a magic num
    peak = (qrs_win - 5) + np.argmax(p_wave[qrs_win-5: qrs_win+5])
    p_peak_props = peak_widths(p_wave, [peak], 0.75)
    p_beg = int(p_peak_props[2][0])
    p_end = int(p_peak_props[3][0])

    p_dict['p_width'] = p_peak_props[0][0] / fs
    p_dict['asymmetry']= (p_end - peak)/(peak - p_beg)

    prom = peak_prominences(p_wave, [peak], int(p_dict['p_width'] * fs) + 5)
    p_dict['p_prom'] = prom[0][0]

    # TODO: same as above rearding the two values after the list
    # return [p_width, pval, p_prom, p_asymmetry, p_sign], p_beg - peak, p_end - peak
    return p_dict

