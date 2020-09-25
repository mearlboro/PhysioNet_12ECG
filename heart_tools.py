#!/us/bin/env python
# -*- coding: utf-8 -*-
"""
This file includes functions to extract signal features such as
properties of the QRS complex, the P and T-waves etc

TODO: @Hardik @Max please add comments describing these
"""

import numpy as np
import pywt
from scipy.signal import find_peaks, hilbert, peak_widths, peak_prominences, argrelextrema, welch
from scipy.stats import iqr, skew, kurtosis
import filters
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
    qrs_dict['qrs_height'] = max(qrs) - min(qrs)
    qrs_dict['r_sign'] = np.sign(qrs[qrs_win])

    qrs_dict['qval']      = qrs[q_peak]
    qrs_dict['qval_norm'] = qrs_dict['qval']/qrs_dict['qrs_height']
    qrs_dict['sval']      = qrs[s_peak]
    qrs_dict['sval_norm'] = qrs_dict['sval']/qrs_dict['qrs_height']
    qrs_dict['rval']      = qrs[r_peak]
    qrs_dict['rval_norm'] = qrs_dict['rval']/qrs_dict['qrs_height']
    qrs_dict['qrs_axis']  = qrs_dict['rval_norm'] - qrs_dict['sval_norm']
    qrs_dict['qs_height'] = qrs_dict['sval'] - qrs_dict['qval']

    q_width,_,q_beg,_ = peak_widths(-1 * qrs,[q_peak], 0.75)
    s_width,_,_,s_end = peak_widths(-1 * qrs,[s_peak], 0.75)
    r_width,_,_,_     = peak_widths(qrs, [r_peak], 1)

    qrs_dict['q_width'] = q_width[0] / fs
    qrs_dict['r_width'] = r_width[0] / fs
    qrs_dict['s_width'] = s_width[0] / fs

    q_beg = int(q_beg[0])
    s_end = int(s_end[0])

    qrs_dict['qrs_avg']  = np.mean(qrs[q_beg:s_end])
    qrs_dict['qrs_dev']   = np.std(qrs[q_beg: s_end])
    qrs_dict['qrs_iqr']  = iqr(qrs[q_beg:s_end])
    qrs_dict['qrs_iqr_norm'] = qrs_dict['qrs_iqr']/qrs_dict['qrs_avg']
    qrs_dict['qrs_width'] = (s_end - q_beg) / fs

    qrs_dict['qrs_skew'] = skew(qrs[q_beg: s_end])
    qrs_dict['qrs_kurt'] = kurtosis(qrs[q_beg: s_end])

    qr_dist = r_peak - q_peak
    rs_dist = s_peak - r_peak
    qrs_dict['qrs_asymmetry'] = rs_dist/qr_dist
    
    qrs_dict['q_peak'] = q_peak - r_peak
    qrs_dict['s_peak'] = s_peak - r_peak
    qrs_dict['q_beg'] = q_beg - r_peak
    qrs_dict['s_end'] = s_end - r_peak

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
    p_dict['p_height'] = max(p_wave)

    # TODO: explain why the 5 is a magic num
    peak = (qrs_win - 5) + np.argmax(p_wave[qrs_win-5: qrs_win+5])
    p_peak_props = peak_widths(p_wave, [peak], 0.75)
    p_beg = int(p_peak_props[2][0])
    p_end = int(p_peak_props[3][0])

    p_dict['p_width'] = p_peak_props[0][0] / fs
    p_dict['p_asymmetry']= (p_end - peak)/(peak - p_beg)

    prom = peak_prominences(p_wave, [peak], int(p_dict['p_width'] * fs) + 5)
    p_dict['p_prom'] = prom[0][0]
    p_dict['p_beg'] = p_beg - peak
    p_dict['p_end'] = p_end - peak

    # TODO: same as above rearding the two values after the list
    # return [p_width, pval, p_prom, p_asymmetry, p_sign], p_beg - peak, p_end - peak
    return p_dict

def get_t_props(t_wave,
                fs: float
            )-> Dict[str, Any]:
    '''
    Gets properties of a sigle t_wave

    Parameters
    ----------
    t_wave : 1d numpy array
    
    fs : float
        sample rate of the time series.

    Returns
    -------
    Dictionary with feature names as keys (string) and float/int as values.

    '''
    qrs_win = int(0.1*fs)
    t_dict = dict()
    t_dict['t_sign'] = np.sign(t_wave[qrs_win])
    t_dict['t_height'] = max(t_wave)
    
    if(t_wave[qrs_win]<0):
        t_wave = -1*t_wave
    peak = (qrs_win-5) + np.argmax(t_wave[qrs_win-5:qrs_win+5])
    t_peak_props = peak_widths(t_wave, [peak], 0.75)
    t_dict['t_width'] = t_peak_props[0][0]/fs
    t_beg = int(t_peak_props[2][0])
    t_end = int(t_peak_props[3][0])

    prom = peak_prominences(t_wave, [peak], int(t_dict['t_width']*fs)+5)
    t_dict['t_prom'] = prom[0][0]
    t_dict['t_asymmetry'] = (t_end - peak)/(peak - t_beg)
    t_dict['t_beg'] = t_beg - peak
    t_dict['t_end'] = t_end - peak
    #return [t_width, tval, t_prom, t_asymmetry, t_sign], t_beg - peak, t_end - peak
    return t_dict

def get_dist_props(data, q_begs, r_peaks, s_ends, t_begs, t_ends, p_begs, p_ends, m_hrs, fs):
    '''
    Gets all the measures related to distance segments of the ECG data

    Parameters
    ----------
    data : 1d array 
        Cleaned ECG data.
    q_begs : list
        list of timestamps of brgining of q waves.
    r_peaks : list
        list of timestamps of r peaks.
    s_ends : list
        list of timestamps of end of s waves.
    t_begs : list
        list of timestamps of brgining of t waves.
    t_ends : list
        list of timestamps of end of t waves.
    p_begs : list
        list of timestamps of brgining of p waves.
    p_ends : list
        list of timestamps of end of p waves.
    fs : float
        Sampling frequency of the given ECG data.

    Returns
    -------
    props : dict
        A dictionary of all the distance propoerties (mean, std, iqr of each segment)
        feature names as strings and float/int as values.

    '''
    props = dict()
    
    q_begs = np.array(q_begs)
    r_peaks = np.array(r_peaks)
    s_ends = np.array(s_ends)
    t_begs = np.array(t_begs)
    t_ends = np.array(t_ends)
    p_begs = np.array(p_begs)
    p_ends = np.array(p_ends)

    start = q_begs[0]
    r_peaks = r_peaks[r_peaks>start]
    s_ends = s_ends[s_ends>start]
    t_begs = t_begs[t_begs>start]
    t_ends = t_ends[t_ends>start]
    p_begs = p_begs[p_begs>start]
    p_ends = p_ends[p_ends>start]

    lens = [len(q_begs), len(r_peaks), len(s_ends), len(t_begs), len(t_ends), len(p_begs), len(p_ends)]
    min_len = min(lens)
    q_begs = q_begs[:min_len]
    r_peaks = r_peaks[:min_len]
    s_ends = s_ends[:min_len]
    t_begs = t_begs[:min_len]
    t_ends = t_ends[:min_len]
    p_begs = p_begs[:min_len]
    p_ends = p_ends[:min_len]


    qr_dist = (r_peaks[:min_len] - q_begs[:min_len])/fs
    props['qr_dist_mean'] = np.mean(qr_dist)
    props['qr_dist_std'] = np.std(qr_dist)
    props['qr_dist_iqr'] = iqr(qr_dist,nan_policy='omit')

    #rs_dist = (s_ends[:min_len] - r_peaks[:min_len])/fs
    #props['rs_dist_mean'] = np.mean(rs_dist)
    #props['rs_dist_std'] = np.mean(rs_dist)
    #props['rs_dist_iqr'] = iqr(rs_dist, nan_policy='omit')

    sq_lens = [(q_begs[i+1] - s_ends[i])/fs for i in range(len(s_ends)-1)]
    props['sq_dist_mean'] = np.nanmean(sq_lens)
    props['sq_dist_std'] = np.nanstd(sq_lens)
    props['sq_dist_iqr'] = iqr(sq_lens, nan_policy='omit')

    pr_dist = [(q_begs[i+1] - p_begs[i])/fs for i in range(len(p_begs)-1)]
    props['pr_dist_mean'] = np.nanmean(pr_dist)
    props['pr_dist_std'] = np.nanstd(pr_dist)
    props['pr_dist_iqr'] = iqr(pr_dist, nan_policy='omit')

    st_seg = [data[s_ends[i]:t_begs[i]] for i in range(len(t_begs)-1)]
    st_slopes = [np.mean(np.diff(i)) for i in st_seg]
    st_curve = [np.mean(np.diff(i,2)) for i in st_seg]
    #st_avg = [np.mean(i) for i in st_seg]
    #st_dev = [np.std(i) for i in st_seg]
    #st_dist = [(t_ends[i] - s_ends[i])/fs for i in range(len(t_ends))]
    tp_dist = [(p_begs[i] - t_ends[i])/fs for i in range(len(p_begs))]
    qt_dist = [(t_ends[i] - q_begs[i])/fs for i in range(len(t_ends))]

    props['st_slope_mean'] = np.nanmean(st_slopes)
    props['st_slope_std'] = np.nanstd(st_slopes)
    props['st_curve_mean'] = np.nanmean(st_curve)
    props['st_curve_std'] = np.nanstd(st_curve)
    #props['st_avg_mean'] = np.nanmean(st_avg)
    #props['st_avg_std'] = np.nanstd(st_avg)
    #props['st_dev_mean'] = np.nanmean(st_dev)
    #props['st_dev_std'] = np.nanstd(st_dev)
    #props['st_dist_mean'] = np.nanmean(st_dist)
    #props['st_dist_std'] = np.nanstd(st_dist)
    props['tp_dist_mean'] = np.nanmean(tp_dist)
    props['tp_dist_std'] = np.nanstd(tp_dist)
    props['qt_dist_mean'] = np.nanmean(qt_dist)
    props['qt_dist_std'] = np.nanstd(qt_dist)

    props['st_slope_iqr'] = iqr(st_slopes,nan_policy='omit')
    props['st_curve_iqr'] = iqr(st_curve,nan_policy='omit')
    #props['st_avg_iqr'] = iqr(st_avg,nan_policy='omit')
    #props['st_dev_iqr'] = iqr(st_dev,nan_policy='omit')
    #props['st_dist_iqr'] = iqr(st_dist,nan_policy='omit')
    props['tp_dist_iqr'] = iqr(tp_dist,nan_policy='omit')
    props['qt_dist_iqr'] = iqr(qt_dist,nan_policy='omit')
    
    dist_prop_names =list(props.keys())
    dist_prop_names_mean = [i for i in dist_prop_names if i.split('_')[-1]=='mean']
    for i,prop_name in enumerate(dist_prop_names_mean):
        pname = prop_name[:-5]
        props[pname+'_mean_norm'] = props[pname+'_mean']/m_hrs
        props[pname+'_iqr_norm'] = props[pname+'_iqr']/m_hrs
        props[pname+'_mean_rootnorm'] = props[pname+'_mean']/np.sqrt(m_hrs)
        #props[pname+'_iqr_rootnorm'] = props[pname+'_iqr']/np.sqrt(m_hrs)
        
    prop_names_all = list(props.keys())
    prop_names_final = [i for i in prop_names_all if i.split('_')[-1]=='norm' or i.split('_')[-1]=='rootnorm']
    props_final = {i:props[i] for i in prop_names_final}

    '''
    prop_means = [qr_dist_mean, rs_dist_mean, sq_dist_mean, pr_dist_mean, st_slope_mean, st_curve_mean,
                  st_avg_mean, st_dev_mean, st_dist_mean, tp_dist_mean, qt_dist_mean]

    prop_stds = [qr_dist_std, rs_dist_std, sq_dist_std, pr_dist_std, st_slope_std, st_curve_std,
                 st_avg_std, st_dev_std, st_dist_std, tp_dist_std, qt_dist_std]

    prop_iqr = [qr_dist_iqr, rs_dist_iqr, sq_dist_iqr, pr_dist_iqr, st_slope_iqr, st_curve_iqr,
                 st_avg_iqr, st_dev_iqr, st_dist_iqr, tp_dist_iqr, qt_dist_iqr]
    '''
    return props_final, props['qt_dist_mean']

def sign_interactions(signs1, signs2):
    '''
    Returns average sign interaction (same or opposite signs)
    between two different peak types (eg. r_peaks and p_peaks)
    Order agnostic.

    Parameters
    ----------
    signs1 : list
        A list of signs (+1/-1) of the first peak type.
    signs2 : list
        A list of signs (+1/-1) of the first peak type.

    Returns
    -------
    mean_interaction
        Average signs in the same/opposite (1,-1) directions between the 
        two given list of signs.

    '''
    signs1 = np.array(signs1)
    signs2 = np.array(signs2)
    minlen = min(len(signs1),len(signs2))
    signs1 = signs1[:minlen]
    signs2 = signs2[:minlen]
    interaction = signs1*signs2
    return np.mean(interaction)

def wavelet_denoise(a):
    '''
    Gets rid of high frequency noise from a given raw ecg signal
    using db4 wavelet denoising (fs approx 500 Hz).

    Parameters
    ----------
    a : 1d array
        ECG signal to be denoised.

    Returns
    -------
    denoised : 1d array
        Denoised signal.

    '''
    wavelet = pywt.Wavelet('db4')
    coeffs = pywt.wavedec(a,wavelet,level = pywt.dwt_max_level(len(a),wavelet.dec_len))
    coeffs[-1] = np.zeros_like(coeffs[-1])
    coeffs[-2] = np.zeros_like(coeffs[-2])
    denoised = pywt.waverec(coeffs,wavelet)
    return denoised

def find_prom_peak(seg):
    '''
    Returns the most prominent peak index of a given segment

    Parameters
    ----------
    seg : 1d array
        Segment of the data that has the peak.

    Returns
    -------
    prom_peak : int
        index of the prominent peak.
    '''
    ps = argrelextrema(seg,np.greater)[0]
    if(len(ps)>0):
        prom_peak = ps[np.argmax(peak_prominences(seg,ps)[0])]
    else:
        prom_peak = np.argmax(seg)
    return prom_peak

def norm_abs(x):
    '''
    Normalizes the given time series by its Median Average Deviation (MAD)

    Parameters
    ----------
    x : 1d array
        Input timeseries.

    Returns
    -------
    1d array
        Normalized timeseries.

    '''
    std = 1.4826 * np.median(abs(x - np.median(x)))
    return (x - np.median(x))/std

def add_lead_str(dic, lead_index):
    '''
    Appends lead names to the end of each key of the dictionary passed

    Parameters
    ----------
    dic : dict
        A dictionary with string keys and numeric values.
    lead_index : int
        Index of the lead in the 12 ECG config [0-11].

    Returns
    -------
    dic : dict
        Dictionary with updated keys with lead strings appended.

    '''
    lead_names = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']
    str_to_add = lead_names[lead_index]
    dict_keys = list(dic.keys())
    for old_key in dict_keys:
        dic[old_key+'_'+str_to_add] = dic.pop(old_key)
    return dic

def get_peak_measures(data,fs, lead):
    '''
    Sequentially sifts through the data to find all the peak and distance measures
    of a given ECG signal.
    The sequence of detection is as follows:
        r_peaks -> qrs complex -> p_waves and t_waves -> distances between the waves
    
    The code returns a dictionary of all the measures.
    It returns an empty dictionary if sufficient r_peaks are not found.

    Parameters
    ----------
    data : 2d array
        TxN array of ECG data where T = time and N = number of leads.
    fs : int
        Sampling frequency of the ECG data.
    lead : int
        Lead index.

    Returns
    -------
    peak_props : dict
        A dictionary of all the estimated peak+distance measures for the given input lead.

    '''
    peak_props = dict()
    data = data[:,lead]
    data1 = filters.butter_bandpass_filter(data,lowcut=1.5,highcut=25,fs=fs)
    data1 = wavelet_denoise(data1)

    if(fs==1000):
        data1 = data1[::2]
        fs=500

    b = data1.copy()

    if(len(b)%2 != 0):
        b = b[:-1]
    b = np.abs(hilbert(b))
    b = norm_abs(b)
    th = 2
    peaks, _ = find_peaks(b, height = th, distance = int(0.3*fs))
    qrs_win = int(fs*0.1)
    if(len(peaks)>int(0.5*len(data1)/fs)):
        if(peaks[0]<qrs_win):
            peaks = peaks[1:]
        if(len(data1) - peaks[-1] < qrs_win):
            peaks = peaks[:-1]
        peak_vals = [max(np.abs(b[p-(qrs_win//2):p+(qrs_win//2)])) for p in peaks]
        r_peaks = peaks[np.argwhere(peak_vals>0.7*np.median(peak_vals))].flatten()
        shifts = [np.argmax(np.abs(data1[p-(qrs_win//2):p+(qrs_win//2)]))-(qrs_win//2) for p in r_peaks if (p>(qrs_win//2))&(p<len(data1)-(qrs_win//2))]
        r_peaks = [p+s for p,s in zip(r_peaks,shifts)]
        hrs = np.diff(r_peaks)/fs
        peak_props['mean_hrs'] = np.mean(hrs)
        peak_props['std_hrs'] = np.std(hrs)
        peak_props['iqr_hrs'] = iqr(hrs,nan_policy='omit')
        peak_props['skew_hrs'] = skew(hrs)
        peak_props['kurtosis_hrs'] = kurtosis(hrs)
        peak_props['std_hrs_norm'] = peak_props['std_hrs']/peak_props['mean_hrs']
        peak_props['skew_hrs_norm'] = peak_props['skew_hrs']/peak_props['mean_hrs']
        peak_props['kurtosis_hrs_norm'] = peak_props['kurtosis_hrs']/peak_props['mean_hrs']
        
        
        qrs = [data1[i-qrs_win:i+qrs_win] for i in r_peaks if (i>qrs_win)&(i<len(data1)-qrs_win)]
        r_signs = [np.sign(i[qrs_win]) for i in qrs]
        if(len(qrs)==0):
            qrs_props_mean = [np.nan for i in range(11)]
            qrs_props_std = [np.nan for i in range(11)]
        qrs_props = [get_qrs_props(i,fs) for i in qrs]
        
        q_peaks = [qrs_prop['q_peak'] + r_peak for (qrs_prop,r_peak) in zip(qrs_props,r_peaks)]
        s_peaks = [qrs_prop['s_peak'] + r_peak for (qrs_prop,r_peak) in zip(qrs_props,r_peaks)]
        q_begs = [qrs_prop['q_beg'] + r_peak for (qrs_prop,r_peak) in zip(qrs_props,r_peaks)]
        s_ends = [qrs_prop['s_end'] + r_peak for (qrs_prop,r_peak) in zip(qrs_props,r_peaks)]
        
        qrs_prop_names = list(qrs_props[0].keys())[:-4]
        qrs_prop_names_mean = [i+'_mean' for i in qrs_prop_names]
        qrs_prop_names_std = [i+'_std' for i in qrs_prop_names]
        qrs_prop_vals = [list(i.values()) for i in qrs_props]
        qrs_props_mean = np.nanmean(qrs_prop_vals,axis=0)[:-4]
        qrs_props_std = np.nanstd(qrs_prop_vals,axis=0)[:-4]
        peak_props.update(dict(zip(qrs_prop_names_mean,qrs_props_mean)))
        peak_props.update(dict(zip(qrs_prop_names_std,qrs_props_std)))

        if(np.mean(data1[r_peaks])<0):
            data1 = -1*data1.copy()
        else:
            data1 = data1.copy()
        
        sq_segs = [np.abs(data1[s_ends[i] :q_begs[i+1]]) for i in range(len(s_ends)-1)]
        sq_lens = [len(i) for i in sq_segs]

        t_peaks = [s_ends[i] + (sq_lens[i]//8) + find_prom_peak(sq_segs[i][sq_lens[i]//8:sq_lens[i]//2]) for i in range(len(s_ends)-1) if sq_lens[i]>0 and sq_lens[i]/fs<3]
        p_peaks = [s_ends[i]+  (6*sq_lens[i]//8) + find_prom_peak(sq_segs[i][6*sq_lens[i]//8:]) for i in range(len(s_ends)-1) if sq_lens[i]>0 and sq_lens[i]/fs<3]


        t_waves = [data1[i-qrs_win:i+qrs_win] for i in t_peaks if (len(data1)-i)>qrs_win]
        t_signs = [np.sign(i[qrs_win]) for i in t_waves]
        p_waves = [data1[i-qrs_win:i+qrs_win] for i in p_peaks if (len(data1)-i)>qrs_win]
        p_signs = [np.sign(i[qrs_win]) for i in p_waves]

        t_props = [get_t_props(i,fs) for i in t_waves]
        p_props = [get_p_props(i,fs) for i in p_waves]

        t_begs = [t_prop['t_beg'] + t_peak for (t_prop,t_peak) in zip(t_props,t_peaks)]
        t_ends = [t_prop['t_end'] + t_peak for (t_prop,t_peak) in zip(t_props,t_peaks)]
        p_begs = [p_prop['p_beg'] + p_peak for (p_prop,p_peak) in zip(p_props,p_peaks)]
        p_ends = [p_prop['p_end'] + p_peak for (p_prop,p_peak) in zip(p_props,p_peaks)]
        
        t_prop_names = list(t_props[0].keys())[:-2]
        t_prop_names_mean = [i+'_mean' for i in t_prop_names]
        t_prop_names_std = [i+'_std' for i in t_prop_names]
        t_props = [list(i.values()) for i in t_props]
        t_props_mean = np.nanmean(t_props,axis=0)[:-2]
        t_props_std = np.nanstd(t_props,axis=0)[:-2]
        peak_props.update(dict(zip(t_prop_names_mean,t_props_mean)))
        peak_props.update(dict(zip(t_prop_names_std,t_props_std)))
        
        p_prop_names = list(p_props[0].keys())[:-2]
        p_prop_names_mean = [i+'_mean' for i in p_prop_names]
        p_prop_names_std = [i+'_std' for i in p_prop_names]
        p_props = [list(i.values()) for i in p_props]
        p_props_mean = np.nanmean(p_props,axis=0)[:-2]
        p_props_std = np.nanstd(p_props,axis=0)[:-2]
        peak_props.update(dict(zip(p_prop_names_mean,p_props_mean)))
        peak_props.update(dict(zip(p_prop_names_std,p_props_std)))

        peak_props['rt_inversion_mean'] = sign_interactions(r_signs,t_signs)
        peak_props['rp_inversion_mean'] = sign_interactions(r_signs,p_signs)

        dist_props, qt_dist_mean = get_dist_props(data1, q_begs, r_peaks, s_ends, t_begs, t_ends, p_begs, p_ends, peak_props['mean_hrs'], fs)
        peak_props.update(dist_props)
        
        peak_props['q_width_mean_norm'] = peak_props['q_width_mean']/peak_props['mean_hrs']
        peak_props['r_width_mean_norm'] = peak_props['r_width_mean']/peak_props['mean_hrs']
        peak_props['s_width_mean_norm'] = peak_props['s_width_mean']/peak_props['mean_hrs']
        peak_props['qrs_width_mean_norm'] = peak_props['qrs_width_mean']/peak_props['mean_hrs']
        peak_props['t_width_mean_norm'] = peak_props['t_width_mean']/peak_props['mean_hrs']
        peak_props['p_width_mean_norm'] = peak_props['p_width_mean']/peak_props['mean_hrs']
        peak_props['p_height_mean_norm'] = peak_props['p_height_mean']/peak_props['qrs_height_mean']
        peak_props['p_prom_mean_norm'] = peak_props['p_prom_mean']/peak_props['qrs_height_mean']
        peak_props['p_asymmetry_mean_norm'] = peak_props['p_asymmetry_mean']/peak_props['qrs_height_mean']
        peak_props['t_height_mean_norm'] = peak_props['t_height_mean']/peak_props['qrs_height_mean']
        peak_props['t_prom_mean_norm'] = peak_props['t_prom_mean']/peak_props['qrs_height_mean']
        peak_props['t_asymmetry_mean_norm'] = peak_props['t_asymmetry_mean']/peak_props['qrs_height_mean']
        
        peak_props['qtcorr2mean'] = qt_dist_mean/ (peak_props['mean_hrs']**(1./3))
        
        
        peak_props = add_lead_str(peak_props, lead)
        
    else:
        peak_props = dict()
        #peak_measures = [np.nan for i in range(147)]
    return peak_props


def get_power_measures(data,fs, lead):
    '''
    Returns a dictionary of power spectrum features of the given ECG data

    data : 2d array
        TxN array of ECG data where T = time and N = number of leads.
    fs : int
        Sampling frequency of the ECG data.
    lead : int
        Lead index.

    Returns
    -------
    pow_props : dict
        Dictionary with feature names as keys and float/int as values.

    '''
    pow_props = dict()
    if (fs == 1000):
        data = data[::2]
        fs = 500

    d = data[:,lead]
    f, Pxx_den = welch(d, fs,nperseg=16,scaling='density')
    power_abs = np.log(Pxx_den)
    power_names = [f'power{i}' for i in range(1,10)]
    pow_props.update(dict(zip(power_names,power_abs.tolist())))
    
    power_rel = np.log(Pxx_den/np.sum(Pxx_den))
    m_power = np.mean(power_rel)
    
    ratios = power_rel/m_power
    ratios_names = [f'relpower{i}_norm' for i in range(1,10)]
    pow_props.update(dict(zip(ratios_names,ratios)))
    
    pow_props = add_lead_str(pow_props, lead)
    
    return pow_props #power_abs.tolist() + ratios.tolist() + [m_power,std_power,iqr_power,max_power,min_power]

def get_ECG_measures(data, fs):
    '''
    Combines peak and power measures of all leads into one single dictionary
    This dictionary will be passed to get_12ECG_features.py

    Parameters
    ----------
    data : 2d array
        TxN array of ECG data where T = time and N = number of leads.
    fs : int
        Sampling frequency of the ECG data.

    Returns
    -------
    ecg_measures : dict
        Dictionary with the peak and power measures of all the leads.

    '''
    ecg_measures = dict()
    for i in range(12):
        peak_dict = get_peak_measures(data, fs, i)
        pow_dict = get_power_measures(data, fs, i)
        ecg_measures.update(peak_dict)
        ecg_measures.update(pow_dict)
    return ecg_measures        

