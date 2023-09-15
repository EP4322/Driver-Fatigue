
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

    # Plot using MNE
    '''sig = ['Fz', 'Cz', 'C3', 'C4', 'Pz']
    raw1_temp = raw1/10**6
    info = mne.create_info(sig, fs, ch_types='eeg')
    ORIG = mne.io.RawArray(raw1_temp, info)
    mne.viz.plot_raw(ORIG)

    raw2_temp = out/10**6
    info = mne.create_info(sig, fs, ch_types='eeg')
    FILT = mne.io.RawArray(raw2_temp, info)
    mne.viz.plot_raw(FILT)'''
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


def get_power(EEG, ECG, fs, t):
    # Collect features

    T = 1/fs
    win = int(t/T)  # Calculates the number of samples per window
    n = int(len(ECG)/win)

    # IGNORE FOR NOW
    ECG_features_out = np.empty((1, 1))
    detectors1 = Detectors(fs-50)

    # Run for each electrode
    for b in range(5):
        # Run for each segment
        for a in range(n):
            # One stage
            c = EEG[b, range(a * win, a * win + win)].T
            Delta0 = bandpower(c, fs, 0.5, 4)
            Theta0 = bandpower(c, fs, 4, 7)
            Alpha0 = bandpower(c, fs, 8, 12)
            Beta10 = bandpower(c, fs, 12.5, 16)

            # Entropy
            # FROM NEUROKIT2
            ApEnt0 = nk.entropy_approximate(c)
            SaEnt0 = nk.entropy_sample(c)
            Sh_Ent0 = nk.entropy_shannon(c)
            fuzzy0 = nk.entropy_fuzzy(c)
            Multiscale0 = nk.entropy_multiscale(c)
            # Other ENT
            Sp_ent0 = ent.spectral_entropy(c, fs)
            wave_ent0 = WE(c)
            # OTHER FEATURES
            EEG_mean0 = np.nanmedian(c)
            EEG_std0 = np.nanstd(c)
            EEG_kurt0 = stats.kurtosis(c, nan_policy='omit')
            EEG_var0 = np.nanvar(c)
            EEG_skew0 = stats.skew(c, nan_policy='omit')

            # Format the features
            EEG_features_out0 = np.hstack((Delta0, Theta0, Alpha0, Beta10, ApEnt0, SaEnt0, Sh_Ent0,
                                           fuzzy0, Multiscale0, Sp_ent0, wave_ent0, EEG_mean0, EEG_std0,
                                           EEG_kurt0, EEG_var0, EEG_skew0))

            if b == 0:
                # Collect ECG features for the first electrode iteration
                ECG_features_out0 = ECG_features(a, win, fs, ECG, detectors1)

                if a == 0:
                    ECG_features_out = ECG_features_out0
                else:
                    ECG_features_out = np.vstack((ECG_features_out, ECG_features_out0))

            if a == 0:
                EEG_features_outT = EEG_features_out0

            else:
                EEG_features_outT = np.vstack((EEG_features_outT, EEG_features_out0))

        if b == 0:
            EEG_features_out = EEG_features_outT
        else:
            EEG_features_out = np.hstack((EEG_features_out, EEG_features_outT))


    return EEG_features_out, ECG_features_out


def ECG_features(a, win, fs, ECG, detectors1):
    # IGNORE THIS FOR NOW, THIS IS FIXED IN FUTURE SCRIPTS

    # Do breathing rate over t=30 instead of t=15
    win2 = int(30 * fs)
    if (a+2) % 2 == 0:
        c2 = ECG[range(int((a+2)/2-1) * win2, int((a+2)/2-1) * win2 + win2)].T
    else:
        c2 = ECG[range(int((a+1)/2-1) * win2, int((a+1)/2-1) * win2 + win2)].T
    r_peaks2 = np.array(detectors1.pan_tompkins_detector(c2))
    # Calculate RRI
    rri2 = (np.diff(r_peaks2) / fs) * 10 ** 3
    # Filter RRI to remove big peaks
    filt_rri2 = np.array(quotient(rri2))
    b02 = np.median(filt_rri2)
    filt_rri2[np.where(filt_rri2 > 1200)] = b02
    filt_rri2[np.where(filt_rri2 < 600)] = b02

    ecg_rate2 = nk.ecg_rate(r_peaks2, sampling_rate=fs, desired_length=len(c2))

    # Breathing
    edr2 = nk.ecg_rsp(ecg_rate2, sampling_rate=fs)
    # Clean signal
    cleaned2 = nk.rsp_clean(edr2, sampling_rate=fs)
    # Extract peaks
    df2, peaks_dict2 = nk.rsp_peaks(cleaned2)
    info2 = nk.rsp_fixpeaks(peaks_dict2)
    formatted2 = pd.DataFrame({"RSP_Raw": edr2, "RSP_Clean": cleaned2})
    # Extract rate
    rsp_rate2 = nk.rsp_rate(formatted2, sampling_rate=fs)
    if math.isnan(rsp_rate2[0]) == False:
        #print(rsp_rate)
        rsp = np.nanmean(rsp_rate2)
        rrv = nk.rsp_rrv(rsp_rate2, info2, sampling_rate=fs, show=False)
        RRV_median = rrv['RRV_MedianBB'][0]
        RRV_mean = rrv['RRV_MeanBB'][0]
        RRV_ApEn = rrv['RRV_ApEn'][0]
    else:
        rsp = float('nan')
        RRV_median = float('nan')
        RRV_mean = float('nan')
        RRV_ApEn = float('nan')


    # Back to t=15 for other ECG
    c = ECG[range(a * win, a * win + win)].T
    r_peaks1 = np.array(detectors1.pan_tompkins_detector(c))
    # Calculate RRI
    rri1 = (np.diff(r_peaks1) / fs) * 10 ** 3
    # Filter RRI to remove big peaks
    filt_rri = np.array(quotient(rri1))
    d = np.std(filt_rri)
    b0 = np.median(filt_rri)
    filt_rri[np.where(filt_rri > 1200)] = b0
    filt_rri[np.where(filt_rri < 600)] = b0

    ecg_rate = nk.ecg_rate(r_peaks1, sampling_rate=fs, desired_length=len(c))

    # HR and Entropy
    HR = np.nanmedian(ecg_rate)

    if HR > 120:
        HR = np.NaN
        HR_std = np.NaN
        Sh_Ent = np.NaN
        Approx = np.NaN
        fuzzy = np.NaN
        wave_ent = np.NaN
        HRV_mean = np.NaN
        HRV_std = np.NaN
        HRV_kurt = np.NaN
        HRV_var = np.NaN
        HRV_skew = np.NaN
        VLF = np.NaN
        LF = np.NaN
        HF = np.NaN
        LF_HF = np.NaN
        Power = np.NaN
    elif HR < 40:
        HR = np.NaN
        HR_std = np.NaN
        Sh_Ent = np.NaN
        Approx = np.NaN
        fuzzy = np.NaN
        wave_ent = np.NaN
        HRV_mean = np.NaN
        HRV_std = np.NaN
        HRV_kurt = np.NaN
        HRV_var = np.NaN
        HRV_skew = np.NaN
        VLF = np.NaN
        LF = np.NaN
        HF = np.NaN
        LF_HF = np.NaN
        Power = np.NaN
    else:
        HR_std = np.std(ecg_rate)
        if len(filt_rri) < 4:
            Sh_Ent = np.NaN
            Approx = np.NaN
            fuzzy = np.NaN
            wave_ent = np.NaN
            HRV_mean = np.NaN
            HRV_std = np.NaN
            HRV_kurt = np.NaN
            HRV_var = np.NaN
            HRV_skew = np.NaN
            VLF = np.NaN
            LF = np.NaN
            HF = np.NaN
            LF_HF = np.NaN
            Power = np.NaN
        else:
            Sh_Ent = nk.entropy_shannon(filt_rri)
            Approx = nk.entropy_approximate(filt_rri)
            # sample = nk.entropy_sample(filt_rri) DOES NOT WORK
            fuzzy = nk.entropy_fuzzy(filt_rri)
            # Multiscale = nk.entropy_multiscale(filt_rri) DOES NOT WORK
            wave_ent = WE(filt_rri)
            HRV_mean = np.nanmedian(filt_rri)
            HRV_std = np.nanstd(filt_rri)
            HRV_kurt = stats.kurtosis(filt_rri, nan_policy='omit')
            HRV_var = np.nanvar(filt_rri)
            HRV_skew = stats.skew(filt_rri, nan_policy='omit')
            frequency_domain = hrvanalysis.get_frequency_domain_features(filt_rri)
            VLF = frequency_domain['vlf']
            LF = frequency_domain['lf']
            HF = frequency_domain['hf']
            LF_HF = frequency_domain['lf_hf_ratio']
            Power = frequency_domain['total_power']


    ECG_features_out = np.hstack((HR, VLF, LF, HF, LF_HF, Power, rsp, RRV_median, RRV_mean, RRV_ApEn, HR_std, Sh_Ent,
                                  Approx, fuzzy, wave_ent, HRV_mean, HRV_std, HRV_kurt, HRV_var, HRV_skew))
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


#######################################################################################################################
# Please ignore all ECG in this file. It get replaced with the features in the "Emma_ECG_Drozy.py" script in the future.
# This script takes a while to run.
# Window is 15 or 30
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

    # ECG data (CAN BE IGNORED IN THIS SCRIPT AS THE ECG IS REPLACED LATER)
    data_ECG = -(data[len(CH)-1, :])

    # Filter data
    M11 = ECG_lowpass_anti_aliasing(data_ECG, fs1, len(data_ECG))
    M12, time = processing.resample_sig(data_ECG, fs1, fs)

    M1 = ECG_lowpass(M12, fs, len(data_ECG))
    ECG = ECG_highpass(M1, fs, len(data_ECG))

    # EEG electrodes
    s = ['Fz', 'Cz', 'C3', 'C4', 'Pz']

    # Collect EEG data
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
    EEG_features_out, ECG_features_out = get_power(EEG, ECG, fs, t)

    # Make the arrays
    A_T = np.hstack((EEG_features_out, ECG_features_out))
    if c == 0:
        EEG_feat = EEG_features_out
        ECG_feat = ECG_features_out
        Subject = c * np.ones((len(ECG_features_out), 1))
    else:
        EEG_feat = np.vstack((EEG_feat, EEG_features_out))
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
A = np.hstack((EEG_feat, ECG_feat, Subject2))

# Make the names for the data frame
names = ['Delta', 'Theta', 'Alpha', 'Beta', 'ApEnt', 'SaEnt', 'Sh_Ent', 'fuzzy', 'Multiscale', 'Sp_ent', 'wave_ent',
         'EEG_mean', 'EEG_std', 'EEG_kurt', 'EEG_var', 'EEG_skew', 'HR', 'VLF', 'LF', 'HF', 'LF_HF', 'Power',
         'rsp', 'RRV_median', 'RRV_mean', 'RRV_ApEn', 'HR_std', 'Sh_Ent', 'Approx', 'fuzzy', 'wave_ent', 'HRV_mean',
         'HRV_std', 'HRV_kurt', 'HRV_var', 'HRV_skew', 'Subject']
cols0 = []
for a in range(5):
    cols0.append(names[0:16])

cols = cols0[0] + cols0[1] + cols0[2] + cols0[3] + cols0[4] + names[16:len(names)]

# Make a data frame with the names
df = pd.DataFrame(A, columns=cols)

# Save the features
df.to_pickle("DROZY_ECG_EEG" + str(t) + ".pkl")
