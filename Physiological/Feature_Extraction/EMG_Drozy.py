
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
from hrv.filters import quotient
from sklearn.preprocessing import normalize
import pandas as pd
import neurokit2 as nk
from scipy.signal import argrelextrema
from scipy import stats
import math
import pickle
from scipy.integrate import simps


def file_list():
    training= fns2.get_files()
    return training


def bandpower(x, fs, fmin, fmax):
    f, pxx = scipy.signal.periodogram(x, fs=fs)
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(pxx[ind_min: ind_max], f[ind_min: ind_max])


# Low pass butterworth function
def EMG_lowpass(record, sfreq):
    cutoff_freq = 45
    order = 10
    normalized_cutoff_freq = 2 * cutoff_freq / sfreq

    numerator_coeffs, denominator_coeffs = signal.butter(order, normalized_cutoff_freq)

    sig = record
    filtered_record = signal.lfilter(numerator_coeffs, denominator_coeffs, sig)

    return filtered_record


def rel_abs(data, fs, low, high):
    win = 4 * fs
    freqs, psd = signal.welch(data, fs, nperseg=win)
    idx_delta = np.logical_and(freqs >= low, freqs <= high)
    freq_res = freqs[1] - freqs[0]
    delta_power = simps(psd[idx_delta], dx=freq_res)
    total_power = simps(psd, dx=freq_res)
    delta_rel_power = delta_power / total_power
    return delta_power, delta_rel_power


def get_power(EMG, fs, t):
    T = 1/fs
    win = int(t/T)  # Calculates the number of samples per window
    n = int(len(EMG)/win)

    # Run for each segment
    for a in range(n):
        # One stage
        c = EMG[range(a * win, a * win + win)].T

        # Band Power
        low = bandpower(c, fs, 1, 15)
        med = bandpower(c, fs, 15, 30)
        high = bandpower(c, fs, 30, 50)

        # Absolute and relative power
        abst0, relt0 = rel_abs(c, fs, 1, 15)
        abst1, relt1 = rel_abs(c, fs, 15, 30)
        abst2, relt2 = rel_abs(c, fs, 30, 50)

        # Shape factor
        rms = np.sqrt(np.nanmean(c ** 2))
        MA = np.nanmean(abs(c))
        shape = rms/MA

        amp1 = np.nanmean(abs(c))
        amp2 = np.nanmedian(abs(c))
        var = np.nanvar(abs(c))
        EMG_feat0 = np.hstack((amp1, amp2, var, low, med, high, abst0, abst1, abst2, relt0, relt1, relt2, shape))

        if a == 0:
            EMG_feat = EMG_feat0
        else:
            EMG_feat = np.vstack((EMG_feat, EMG_feat0))

    return EMG_feat


#######################################################################################################################
# Set the window (15 or 30 seconds)
t = 15
#######################################################################################################################

# Collect the file list
training = file_list()

# The number of files there are
sub = 36
for c in range(sub):
    print(c)
    # Collect the data for a subject
    raw_s = mne.io.read_raw_edf(training.edf[c])

    fs1 = 512
    fs = 200

    # Convert data
    data, times = raw_s[:]
    CH = raw_s.ch_names

    # Put in easier format
    EMGt = data[len(CH)-2, :]*10**6

    # Pre-Process features
    EMGt2, time = processing.resample_sig(EMGt, fs1, fs)
    EMG = EMG_lowpass(EMGt2, fs)

    # Get features
    EMG_featt = get_power(EMG, fs, t)

    # Make the arrays
    if c == 0:
        EMG_feat = EMG_featt
        Subject = c * np.ones((len(EMG_featt), 1))
    else:
        EMG_feat = np.vstack((EMG_feat, EMG_featt))
        Subject = np.vstack((Subject, c * np.ones((len(EMG_featt), 1))))

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
Subject2 = np.where(Subject2 == 26, 6.1, Subject2)
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
A = np.hstack((EMG_feat, Subject2))

# Feature names
names = ['Mean', 'Median', 'Variance', 'Low_freq', 'Med_freq', 'High_freq', 'AbsL', 'AbsM', 'AbsH', 'RelL', 'RelM',
         'RelH', 'Shape_factor', 'Subject']
cols = names

# Make a data frame with the names
df = pd.DataFrame(A, columns=cols)

# Save EMG features
df.to_pickle("DROZY_EMG" + str(t) + ".pkl")
