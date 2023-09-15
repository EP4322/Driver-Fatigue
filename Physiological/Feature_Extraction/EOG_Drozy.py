
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


def EOG_highpass(record, sfreq):
    order = 3
    lowcut = 0.1
    # highcut = 50
    nyq = 0.5 * sfreq
    low = lowcut / nyq
    # high = highcut / nyq
    numerator_coeffs, denominator_coeffs = signal.butter(order, low, btype='high')

    sig = record
    filtered_record = signal.lfilter(numerator_coeffs, denominator_coeffs, sig)

    return filtered_record


# Low pass butterworth function
def EOG_lowpass(record, sfreq):
    cutoff_freq = 10
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


def get_power(EOG, fs, t):
    T = 1/fs
    win = int(t/T)  # Calculates the number of samples per window
    n = int(len(EOG)/win)

    # Run for each sections
    for a in range(n):
        # One stage
        c = EOG[range(a * win, a * win + win)].T

        # First order derivative
        New = np.diff(c)/T

        # Set the threshold for blink detection
        Thresh1 = 150
        Thresh2 = -175
        New = New/10

        # Blink detection using external package (in case some are not above the set threshold)
        blinks0 = nk.eog_findpeaks(c, sampling_rate=fs, method="mne")

        # Plot a 30 second segment if needed
        '''if a == 6:
            V1 = np.ones(len(c)) * Thresh2
            V2 = np.ones(len(c)) * Thresh1
            leg = ["Raw", "Diff"]
            fig, (ax) = plt.subplots()
            ax.plot(c)
            ax.plot(New)
            ax.plot(blinks0, c[blinks0], 'ro')
            ax.plot(V1, linestyle='--', color="black")
            ax.plot(V2, linestyle='--', color="black")
            ax.set_title("EOG")
            ax.legend(leg)
            fig.show()'''

        # Find where it is above/below set thresholds
        T1 = np.where(New > Thresh1)
        T2 = np.where(New < Thresh2)

        # If statement for empty tuple
        if len(T1[0]) != 0 and len(T2[0]) != 0:
            # Find where the indicies we want are
            T1_points = np.diff(T1[0])
            T2_points = np.diff(T2[0])

            if len(T1_points) != np.sum(T1_points) and len(T2_points) != np.sum(T2_points):
                T1_p = np.where(T1_points > 1)
                T2_p = np.where(T2_points > 1)

                # Initialise variables
                T1_start = T1[0][0]
                T1_end = np.zeros(1)

                # Collect the closing times
                for g in T1_p[0]:
                    T1_end = np.hstack((T1_end, T1[0][g]))
                    T1_start = np.hstack((T1_start, T1[0][g+1]))

                # Remove the zero initialisation
                T1_end = np.delete(T1_end, 0)

                # Set the last value as this was not set in the loop
                T1_end = np.append(T1_end, T1[0][-1])

                # Remove type errors
                T1_end = T1_end.astype(int)
                T1_start = T1_start.astype(int)

                # If the data started above the threshold, remove it
                if New[0] > Thresh1:
                    T1_start = np.delete(T1_start, 0)
                    T1_end = np.delete(T1_end, 0)

                # If the data is aboove the threshold at the end, remove it
                if New[-1] > Thresh1:
                    T1_start = np.delete(T1_start, -1)
                    T1_end = np.delete(T1_end, -1)

                # Repeat for re-opening
                T2_start = np.zeros(1)
                T2_start[0] = T2[0][0]
                T2_end = np.zeros(1)
                for g in T2_p[0]:
                    T2_end = np.hstack((T2_end, T2[0][g]))
                    T2_start = np.hstack((T2_start, T2[0][g+1]))

                T2_end = np.delete(T2_end, 0)
                T2_end = np.append(T2_end, T2[0][-1])
                T2_end = T2_end.astype(int)
                T2_start = T2_start.astype(int)

                if New[0] < Thresh2:
                    T2_start = np.delete(T2_start, 0)
                    T2_end = np.delete(T2_end, 0)

                if New[-1] < Thresh2:
                    T2_start = np.delete(T2_start, -1)
                    T2_end = np.delete(T2_end, -1)

                # Delete the data is the start and end are the same (ie zero in duration)
                dub = np.where(T1_start == T1_end)
                T1_start = np.delete(T1_start, dub)
                T1_end = np.delete(T1_end, dub)

                dub2 = np.where(T2_start == T2_end)
                T2_start = np.delete(T2_start, dub2)
                T2_end = np.delete(T2_end, dub2)

                # Aligning detection - if two closed are detected before an open and vice versa
                longest = [len(T1_end), len(T2_start)]
                ind = np.max(longest)

                aa = 0
                # While loop to break out when at the end of an array - delete remaining values (if any) in the other array
                while aa < ind:
                    if aa > len(T2_start)-1:
                        T1_start = np.delete(T1_start, range(aa, len(T1_start)))
                        T1_end = np.delete(T1_end, range(aa, len(T1_start)))
                        break
                    elif aa > len(T1_start)-1:
                        T2_start = np.delete(T2_start, range(aa, len(T2_start)))
                        T2_end = np.delete(T2_end, range(aa, len(T2_start)))
                        break

                    # Check for 2 opens before a close (Delete the occurrence)
                    elif T2_start[aa] < T1_start[aa]:
                        T2_start = np.delete(T2_start, aa)
                        T2_end = np.delete(T2_end, aa)

                    # Check for 2 closes before an open (Delete the first one)
                    elif aa < len(T1_start)-1:
                        if T1_start[aa+1] < T2_start[aa]:
                            T1_start = np.delete(T1_start, aa)
                            T1_end = np.delete(T1_end, aa)
                        else:
                            aa += 1
                    else:
                        aa += 1

                # Initialise and Calculate features
                Duration0 = np.zeros(len(T1_start))
                Close0 = np.zeros(len(T1_start))
                Open0 = np.zeros(len(T1_start))
                Delay0 = np.zeros(len(T1_start))
                Vel_close0 = np.zeros(len(T1_start))
                Vel_open0 = np.zeros(len(T1_start))
                Amplitude0 = np.zeros(len(T1_start))
                Delay_ratio0 = np.zeros(len(T1_start))
                for aa in range(len(T1_start)-1):
                    Duration0[aa] = (T2_end[aa] - T1_start[aa]) * T * 10 ** 3 # In ms
                    Close0[aa] = (T1_end[aa] - T1_start[aa]) * T * 10 ** 3
                    Open0[aa] = (T2_end[aa] - T1_start[aa]) * T * 10 ** 3
                    Delay0[aa] = (T2_start[aa] - T1_end[aa]) * T * 10 ** 3
                    Vel_close0[aa] = np.max(New[T1_start[aa]:T1_end[aa]])
                    Vel_open0[aa] = np.min(New[T2_start[aa]:T2_end[aa]])
                    Amplitude0[aa] = np.max(c[T1_start[aa]:T2_end[aa]])
                    Delay_ratio0[aa] = Delay0[aa]/Duration0[aa]

                # Find the averages/median
                freq0 = len(blinks0)
                PSD0 = bandpower(c, fs, 0, 1)/bandpower(c, fs, 1, 20)
                Duration0 = np.nanmedian(Duration0)
                Close0 = np.nanmedian(Close0)
                Open0 = np.nanmedian(Open0)
                Delay0 = np.nanmedian(Delay0)
                Vel_close0 = np.nanmedian(Vel_close0)
                Vel_open0 = np.nanmedian(Vel_open0)
                Amplitude0 = np.nanmedian(Amplitude0)
                Delay_ratio0 = np.nanmedian(Delay_ratio0)
                abs0, rel0 = rel_abs(c, fs, 0, 1)
                EOG_mean0 = np.nanmedian(c)
                EOG_std0 = np.nanstd(c)
                EOG_kurt0 = stats.kurtosis(c, nan_policy='omit')
                EOG_var0 = np.nanvar(c)
                EOG_skew0 = stats.skew(c, nan_policy='omit')
                EOG_features_out0 = np.hstack((freq0, PSD0, Duration0, Close0, Open0, Delay0, Vel_close0, Vel_open0,
                                               Amplitude0, Delay_ratio0, abs0, rel0, EOG_mean0, EOG_std0, EOG_kurt0,
                                               EOG_var0, EOG_skew0))
                # Good reference paper: "An EOG-based Vigilance Estimation Method Applied for Driver Fatigue"

                if a == 0:
                    EOG_features_out = EOG_features_out0
                else:
                    EOG_features_out = np.vstack((EOG_features_out, EOG_features_out0))
            else:
                if a == 0:
                    EOG_features_out = np.zeros(17)
                else:
                    EOG_features_out = np.vstack((EOG_features_out, np.zeros(17)))
        else:
            if a == 0:
                EOG_features_out = np.zeros(17)
            else:
                EOG_features_out = np.vstack((EOG_features_out, np.zeros(17)))
    return EOG_features_out


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

    # Make data in easier format: Horizontal no longer used
    #data_EOGH = data[len(CH)-3, :]*10**6
    data_EOGV = data[len(CH) - 4, :]*10**6

    # Resample and filter data
    #Horro, time = processing.resample_sig(data_EOGH, fs1, fs)
    Vert, time = processing.resample_sig(data_EOGV, fs1, fs)
    EOG_filta = EOG_highpass(Vert, fs)
    EOG_filt = EOG_lowpass(EOG_filta, fs)

    # Collect features
    EOG_features_out = get_power(EOG_filt, fs, t)

    # Plotting features
    if c == 0:
        EOG_feat = EOG_features_out
        Subject = c * np.ones((len(EOG_features_out), 1))
    else:
        EOG_feat = np.vstack((EOG_feat, EOG_features_out))
        Subject = np.vstack((Subject, c * np.ones((len(EOG_features_out), 1))))

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
A = np.hstack((EOG_feat, Subject2))

# Feature names
names = ["Freq", "PSD", "Duration", "Close", "Open", "Delay", "Vel_C", "Vel_O", "Amplitude", "Delay_ratio", "abs",
         "rel", "MeanEOG", "std", "kurt", "var", "skew", "Subject"]

# Make a data frame with the names
df = pd.DataFrame(A, columns=names)

# Save the ECG features
df.to_pickle("DROZY_EOG" + str(t) + ".pkl")
