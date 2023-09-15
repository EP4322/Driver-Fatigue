
import fns2
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pywt
import antropy as ent
from sklearn.decomposition import FastICA
from scipy.signal import butter, lfilter,freqz
from wfdb import processing
from ecgdetectors import Detectors
import heartpy as hp
from scipy import signal
import mne
import hrvanalysis
import hrv
from hrv.filters import quotient
from sklearn.preprocessing import normalize
import pandas as pd
import neurokit2 as nk
from scipy.signal import argrelextrema
from scipy import stats
import math
import pickle
import biosppy
from scipy.io import savemat
import pickle
import scipy.io
import pyhrv.tools as tools
import pyhrv


def file_list():
    training = fns2.get_files()
    return training

def bandpower(x, fs, fmin, fmax):
    f, pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(pxx[ind_min: ind_max], f[ind_min: ind_max])


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def ECG_highpass(record, sfreq, nsample):
    order = 3
    lowcut = 0.1
    nyq = 0.5 * sfreq
    low = lowcut / nyq
    numerator_coeffs, denominator_coeffs = signal.butter(order, low, btype='high')

    sig = record
    filtered_record = signal.lfilter(numerator_coeffs, denominator_coeffs, sig)

    return filtered_record


# Low pass butterworth function
def ECG_lowpass(record, sfreq, nsample):
    cutoff_freq = 30
    order = 10
    normalized_cutoff_freq = 2 * cutoff_freq / sfreq

    numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq)

    time = np.linspace(0, nsample / sfreq, nsample, endpoint=False)

    sig = record
    filtered_record = signal.lfilter(numerator_coeffs, denominator_coeffs, sig)

    return filtered_record


def ECG_lowpass_anti_aliasing(record, sfreq, nsample):
    cutoff_freq = 125
    order = 16
    normalized_cutoff_freq = 2 * cutoff_freq / sfreq

    numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq)

    time = np.linspace(0, nsample / sfreq, nsample, endpoint=False)

    sig = record
    filtered_record= signal.lfilter(numerator_coeffs, denominator_coeffs, sig)

    return filtered_record


def get_power(ECG, fs, t):
    T = 1/fs
    win = int(t/T)  # Calculates the number of samples per window
    if t == 15:
        n = 40
    elif t == 30:
        n = 20

    # Run for each segment
    for a in range(n):
        ECG_features_out0 = ECG_features(a, win, fs, ECG)

        if a == 0:
            ECG_features_out = ECG_features_out0
        else:
            ECG_features_out = np.vstack((ECG_features_out, ECG_features_out0))

    return ECG_features_out


def ECG_features(a, win, fs, ECG):
    # Section to use
    c = ECG[range(a * win, a * win + win)].T

    # Show some data
    if a == 4:
        r_peaks = biosppy.signals.ecg.ecg(c, sampling_rate=fs, show=True)[2]
    else:
        r_peaks = biosppy.signals.ecg.ecg(c, sampling_rate=fs, show=False)[2]

    # Calculate RRI
    rri = (np.diff(r_peaks) / fs) * 10 ** 3

    # Try with hrv_analysis
    filt_rri = hrvanalysis.preprocessing.get_nn_intervals(rr_intervals=rri) # NNi list actually https://aura-healthcare.github.io/hrv-analysis/hrvanalysis.html

    # Delete nan values until a non nan is first in the array
    while np.isnan(filt_rri[0]):
        filt_rri = filt_rri[1:]

    # Interpolate remaining values
    filt_rri = hrvanalysis.preprocessing.interpolate_nan_values(rr_intervals=filt_rri)

    # Get features
    time = hrvanalysis.extract_features.get_time_domain_features(nn_intervals=filt_rri)

    # Some features need the 30 second window or it doesn't work
    if t == 15:
        win2 = int(30 * fs)
        if (a+2) % 2 == 0:
            c2 = ECG[range(int((a+2)/2-1) * win2, int((a+2)/2-1) * win2 + win2)].T
        else:
            c2 = ECG[range(int((a+1)/2-1) * win2, int((a+1)/2-1) * win2 + win2)].T
        r_peaks = biosppy.signals.ecg.ecg(c2, sampling_rate=fs, show=False)[2]
    else:
        c2 = c

    # Using neurokit for breathing
    ecg_rate = nk.ecg_rate(r_peaks, sampling_rate=fs, desired_length=len(c2))

    # Breathing
    edr = nk.ecg_rsp(ecg_rate, sampling_rate=fs)

    # Clean signal
    cleaned = nk.rsp_clean(edr, sampling_rate=fs)

    # Extract peaks
    df, peaks_dict = nk.rsp_peaks(cleaned)
    info = nk.rsp_fixpeaks(peaks_dict)
    formatted = pd.DataFrame({"RSP_Raw": edr, "RSP_Clean": cleaned})

    # Extract rate
    rsp_rate = nk.rsp_rate(formatted, sampling_rate=fs)

    # Check for Nans
    if math.isnan(rsp_rate[0]) == False:
        rsp = np.nanmean(rsp_rate)
        rrv = nk.rsp_rrv(rsp_rate, info, sampling_rate=fs, show=False)
        RRV_median = rrv['RRV_MedianBB'][0]
        RRV_mean = rrv['RRV_MeanBB'][0]
        RRV_ApEn = rrv['RRV_ApEn'][0]
    else:
        rsp = float('nan')
        RRV_median = float('nan')
        RRV_mean = float('nan')
        RRV_ApEn = float('nan')

    # HR and Entropy
    HR = time['mean_hr']

    # Adjust irregular values
    if HR > 120:
        ECG_features_out = np.empty((1, 20))
        ECG_features_out[:] = np.NaN

    elif HR < 40:
        ECG_features_out = np.empty((1, 20))
        ECG_features_out[:] = np.NaN
    else:
        # Collect and format features
        HR_std = time['std_hr']
        if len(filt_rri) < 4:
            ECG_features_out = np.empty((1, 20))
            ECG_features_out[:] = np.NaN
        else:
            Sh_Ent = nk.entropy_shannon(filt_rri)
            Approx = nk.entropy_approximate(filt_rri)
            fuzzy = nk.entropy_fuzzy(filt_rri)
            wave_ent = WE(filt_rri)
            HRV_mean = np.nanmedian(filt_rri)
            HRV_std = np.nanstd(filt_rri)
            HRV_kurt = stats.kurtosis(filt_rri, nan_policy='omit')
            HRV_var = np.nanvar(filt_rri)
            HRV_skew = stats.skew(filt_rri, nan_policy='omit')

            frequency_domain = hrvanalysis.get_frequency_domain_features(filt_rri, sampling_frequency=fs)
            VLF = frequency_domain['vlf']
            LF = frequency_domain['lf']
            HF = frequency_domain['hf']
            LF_HF = frequency_domain['lf_hf_ratio']
            Power = frequency_domain['total_power']

            ECG_features_out = np.hstack((HR, VLF, LF, HF, LF_HF, Power, rsp, RRV_median, RRV_mean, RRV_ApEn, HR_std,
                                          Sh_Ent, Approx, fuzzy, wave_ent, HRV_mean, HRV_std, HRV_kurt, HRV_var,
                                          HRV_skew))
    return ECG_features_out


def WE(y, level=4, wavelet='coif2'):
    # Wavelet entropy calculation
    from math import log
    n = len(y)

    sig = y

    ap = {}

    for lev in range(0, level):
        (y, cD) = pywt.dwt(y, wavelet)
        ap[lev] = y

    # Energy
    Enr = np.zeros(level)
    for lev in range(0, level):
        Enr[lev] = np.sum(np.power(ap[lev], 2)) / n

    Et = np.sum(Enr)

    Pi = np.zeros(level)
    for lev in range(0, level):
        Pi[lev] = Enr[lev] / Et

    we = - np.sum(np.dot(Pi, np.log(Pi)))

    return we


def load_data(name):
    with open(name, 'rb') as f:
        loaded = pickle.load(f)
    return loaded


#######################################################################################################################
# Set the window (15 or 30 seconds)
t = 15
#######################################################################################################################

# Collect the file list
training = file_list()

# The number of files there are
sub = 36
for c in range(sub):
    # Print which number (of 36) is being extracted from
    print(c)

    # Collect the data for a subject
    raw_s = mne.io.read_raw_edf(training.edf[c])

    # Sample frequency
    fs1 = 512

    # Desired frequency
    fs = 200

    # Convert data
    data, times = raw_s[:]
    CH = raw_s.ch_names

    # Data is inverted here
    data_ECG = -(data[len(CH)-1, :])

    # Filter the data and resample
    M11 = ECG_lowpass_anti_aliasing(data_ECG, fs1, len(data_ECG))
    M12, time = processing.resample_sig(data_ECG, fs1, fs)

    M1 = ECG_lowpass(M12, fs, len(data_ECG))
    ECG = ECG_highpass(M1, fs, len(data_ECG))

    # Some ECG recordings need inverting
    if c == 3 or c == 8 or c == 11 or c == 17 or c == 18 or c == 19 or c == 20 or c == 22 or c == 23 or c == 24 or c == 25 or c > 27:
        ECG = -ECG
        # Get the features
        ECG_features_out = get_power(ECG, fs, t)
    elif c == 26:
        if t == 30:
            ECG_features_out = np.empty((20, 20))
            ECG_features_out[:] = np.NaN
        elif t == 15:
            ECG_features_out = np.empty((40, 20))
            ECG_features_out[:] = np.NaN
    else:
        # Get the features
        ECG_features_out = get_power(ECG, fs, t)

    # Make the arrays
    if c == 0:
        ECG_feat = ECG_features_out
        Subject = c * np.ones((len(ECG_features_out), 1))
    else:
        ECG_feat = np.vstack((ECG_feat, ECG_features_out))
        Subject = np.vstack((Subject, c * np.ones((len(ECG_features_out), 1))))

# Set the subjects to their trial numbers
Subject2 = Subject.copy()
Subject2 = np.where(Subject2 == 0, 1.1, Subject2)
Subject2 = np.where(Subject2 == 1, 1.2, Subject2)
Subject2 = np.where(Subject2 == 2, 1.3, Subject2)
Subject2 = np.where(Subject2 == 3, 10.1, Subject2)
Subject2 = np.where(Subject2 == 4, 10.3, Subject2)
Subject2 = np.where(Subject2 == 5, 11.1, Subject2)
Subject2 = np.where(Subject2 == 6, 11.2, Subject2)
Subject2 = np.where(Subject2 == 7, 11.3, Subject2)
Subject2 = np.where(Subject2 == 8, 12.1, Subject2)
Subject2 = np.where(Subject2 == 9, 13.1, Subject2)
Subject2 = np.where(Subject2 == 10, 13.2, Subject2)
Subject2 = np.where(Subject2 == 11, 14.1, Subject2)
Subject2 = np.where(Subject2 == 12, 14.2, Subject2)
Subject2 = np.where(Subject2 == 13, 14.3, Subject2)
Subject2 = np.where(Subject2 == 14, 2.1, Subject2)
Subject2 = np.where(Subject2 == 15, 2.2, Subject2)
Subject2 = np.where(Subject2 == 16, 2.3, Subject2)
Subject2 = np.where(Subject2 == 17, 3.1, Subject2)
Subject2 = np.where(Subject2 == 18, 3.2, Subject2)
Subject2 = np.where(Subject2 == 19, 3.3, Subject2)
Subject2 = np.where(Subject2 == 20, 4.1, Subject2)
Subject2 = np.where(Subject2 == 21, 4.2, Subject2)
Subject2 = np.where(Subject2 == 22, 4.3, Subject2)
Subject2 = np.where(Subject2 == 23, 5.1, Subject2)
Subject2 = np.where(Subject2 == 24, 5.2, Subject2)
Subject2 = np.where(Subject2 == 25, 5.3, Subject2)
Subject2 = np.where(Subject2 == 26, 6.1, Subject2) # ECG data not available.
Subject2 = np.where(Subject2 == 27, 6.2, Subject2)
Subject2 = np.where(Subject2 == 28, 6.3, Subject2)
Subject2 = np.where(Subject2 == 29, 7.2, Subject2)
Subject2 = np.where(Subject2 == 30, 7.3, Subject2)
Subject2 = np.where(Subject2 == 31, 8.1, Subject2)
Subject2 = np.where(Subject2 == 32, 8.2, Subject2)
Subject2 = np.where(Subject2 == 33, 8.3, Subject2)
Subject2 = np.where(Subject2 == 34, 9.2, Subject2)
Subject2 = np.where(Subject2 == 35, 9.3, Subject2)

# Add the subject trials to the array
A = np.hstack((ECG_feat, Subject2))

# Feature names
names = ['HR', 'VLF', 'LF', 'HF', 'LF_HF', 'Power',
         'rsp', 'RRV_median', 'RRV_mean', 'RRV_ApEn', 'HR_std', 'Sh_Ent', 'Approx', 'fuzzy', 'wave_ent', 'HRV_mean',
         'HRV_std', 'HRV_kurt', 'HRV_var', 'HRV_skew', 'Subject']

# Make a data frame with the names
df = pd.DataFrame(A, columns=names)

# Save the ECG features
df.to_pickle("DROZY_ECG_final" + str(t) + ".pkl")

