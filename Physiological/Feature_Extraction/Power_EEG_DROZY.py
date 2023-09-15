
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


def my_wica(raw1, fs, c):
    # Frequency cuttoffs
    lowcut = 0.5
    highcut = 40

    # Filter data
    raw = butter_bandpass_filter(raw1, lowcut, highcut, fs, order=6)

    # Set up and fit ICA
    ica = FastICA(n_components=5, random_state=0) #, tol=0.05)
    components = ica.fit_transform(raw.T)

    data_ICA = components.copy()

    # Wavelet addition
    waveletname = 'coif5'
    for IC in range(5):
        coeffs = pywt.wavedec(data_ICA[:, IC], waveletname, level=5)  # cA5, cD5, cD4.....
        K = 0.01  # MAY NEED TO BE CHANGED
        # Find where the coefficient value exceeds K and set them to 0
        for loop in range(len(coeffs)):
            coeffs[loop][abs(coeffs[loop][:]) > K] = 0
        # Inverse dwt
        rebuilt = pywt.waverec(coeffs, waveletname)
        if IC != 0:
            rebuilt_all = np.vstack((rebuilt_all, rebuilt))
        else:
            rebuilt_all = rebuilt

    # Need to use rebuilt_all in reversing ICA.
    out = ica.inverse_transform(rebuilt_all.T)
    out = out.T

    return out


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


def rel_abs(data, fs, low, high):
    win = 4 * fs
    freqs, psd = signal.welch(data, fs, nperseg=win)
    idx_delta = np.logical_and(freqs >= low, freqs <= high)
    freq_res = freqs[1] - freqs[0]
    delta_power = simps(psd[idx_delta], dx=freq_res)
    total_power = simps(psd, dx=freq_res)
    delta_rel_power = delta_power / total_power
    return delta_power, delta_rel_power


def get_power(EEG, fs, t):
    T = 1/fs
    win = int(t/T)  # Calculates the number of samples per window
    n = int(len(EEG.T)/win)

    # Run for each electrode
    for b in range(5):
        # Run for each segment
        for a in range(n):
            # One stage
            c = EEG[b, range(a * win, a * win + win)].T

            # Relative and absolute power
            Delta0, Delta0b = rel_abs(c, fs, 0.5, 4)
            Theta0, Theta0b = rel_abs(c, fs, 4, 7)
            Alpha0, Alpha0b = rel_abs(c, fs, 8, 12)
            Beta10, Beta10b = rel_abs(c, fs, 12.5, 16)

            EEG_features_out0 = np.hstack((Delta0, Theta0, Alpha0, Beta10, Delta0b, Theta0b, Alpha0b, Beta10b))

            if a == 0:
                EEG_features_outT = EEG_features_out0

            else:
                EEG_features_outT = np.vstack((EEG_features_outT, EEG_features_out0))

        if b == 0:
            EEG_features_out = EEG_features_outT
        else:
            EEG_features_out = np.hstack((EEG_features_out, EEG_features_outT))

    return EEG_features_out


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

    # Sampling frequency
    fs1 = 512

    # Desired frequency
    fs = 200

    # Convert data
    data, times = raw_s[:]
    CH = raw_s.ch_names

    # EEG electrodes
    s = ['Fz', 'Cz', 'C3', 'C4', 'Pz']
    data_EEG = data[0:5, :]

    # Resample EEG data
    M_EEG0, time = processing.resample_sig(data_EEG[0, :], fs1, fs)
    M_EEG1, time = processing.resample_sig(data_EEG[1, :], fs1, fs)
    M_EEG2, time = processing.resample_sig(data_EEG[2, :], fs1, fs)
    M_EEG3, time = processing.resample_sig(data_EEG[3, :], fs1, fs)
    M_EEG4, time = processing.resample_sig(data_EEG[4, :], fs1, fs)
    M_EEG = np.vstack((M_EEG0, M_EEG1, M_EEG2, M_EEG3, M_EEG4))

    # Make into more appropriate format
    M_EEG = M_EEG*10**6

    # Wavelet-ICA on EEG data
    EEG = my_wica(M_EEG, fs, c)

    # Get the features
    EEG_features_out = get_power(EEG, fs, t)

    # Make the arrays
    if c == 0:
        EEG_feat = EEG_features_out
        Subject = c * np.ones((len(EEG_features_out), 1))
    else:
        EEG_feat = np.vstack((EEG_feat, EEG_features_out))
        Subject = np.vstack((Subject, c * np.ones((len(EEG_features_out), 1))))

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
A = np.hstack((EEG_feat, Subject2))

# Make the names for the data frame
names = ['DeltaA', 'ThetaA', 'AlphaA', 'BetaA', 'DeltaR', 'ThetaR', 'AlphaR', 'BetaR', 'Subject']
cols0 = []
for a in range(5):
    cols0.append(names[0:8])
cols = cols0[0] + cols0[1] + cols0[2] + cols0[3] + cols0[4] + names[8:len(names)]

# Make a data frame with the names
df = pd.DataFrame(A, columns=cols)

# Save the features
df.to_pickle("DROZY_Power" + str(t) + ".pkl")
