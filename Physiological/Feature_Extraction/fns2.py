"""
Created on Wed Mar 28 13:19:34 2018

@author: mohammad
"""

import os
import numpy as np
import pandas as pd
# from pylab import find was here but this is no longer supported.
import scipy.io
import joblib

def find(condition):
    """Returns indices where ravel(a) is true.
    Private implementation of deprecated matplotlib.mlab.find
    """
    return np.nonzero(np.ravel(condition))[0]
# -----------------------------------------------------------------------------
# returns a list of the training and testing file locations for easier import
# -----------------------------------------------------------------------------
def get_files():
    header_loc = []
    rootDir = '.'
    for dirName, subdirList, fileList in os.walk(rootDir, followlinks=True):
        if dirName != '.' and dirName != './training2':
            if dirName.startswith('.\\training2\\'):

                for fname in fileList:
                    if '.edf' in fname:
                        header_loc.append(dirName + '\\' + fname)


    # combine into a data frame
    data_locations = {'edf':      header_loc}

    # Convert to a data-frame
    training_files = pd.DataFrame(data=data_locations)

    ''''# Split the data frame into training and testing sets.
    tr_ind = list(find(df.is_training.values))

    training_files = df.loc[tr_ind, :]'''

    return training_files

# -----------------------------------------------------------------------------
# import the outcome vector, given the file name.
# e.g. /training/tr04-0808/tr04-0808-arousal.mat
# -----------------------------------------------------------------------------
def import_arousals(file_name):
    import h5py
    import numpy
    f = h5py.File(file_name, 'r')
    arousals = numpy.array(f['data']['arousals'])
    sleep1 = numpy.array(f['data']['sleep_stages']['nonrem1'])
    sleep2 = numpy.array(f['data']['sleep_stages']['nonrem2'])
    sleep3 = numpy.array(f['data']['sleep_stages']['nonrem3'])
    sleep4 = numpy.array(f['data']['sleep_stages']['rem'])
    sleep5 = numpy.array(f['data']['sleep_stages']['undefined'])
    sleep6 = numpy.array(f['data']['sleep_stages']['wake'])
    stage = []
    for a in range(len(sleep1[0])):
        if sleep1[0][a] == 1:
            stage.append(1)
        elif sleep2[0][a] == 1:
            stage.append(2)
        elif sleep3[0][a] == 1:
            stage.append(3)
        elif sleep4[0][a] == 1:
            stage.append(5)
        elif sleep5[0][a] == 1:
            stage.append(0.5)
        elif sleep6[0][a] == 1:
            stage.append(0.5)
    return arousals, stage

def import_signals(file_name):
    return np.transpose(scipy.io.loadmat(file_name)['val'])

# -----------------------------------------------------------------------------
# Take a header file as input, and returns the names of the signals
# For the corresponding .mat file containing the signals.
# -----------------------------------------------------------------------------
def import_signal_names(file_name):
    with open(file_name, 'r') as myfile:
        s = myfile.read()
        s = s.split('\n')
        s = [x.split() for x in s]

        n_signals = int(s[0][1])
        n_samples = int(s[0][3])
        Fs        = int(s[0][2])

        s = s[1:-1]
        s = [s[i][8] for i in range(0, n_signals)]
    return s, Fs, n_samples

# -----------------------------------------------------------------------------
# Get a given subject's data
# -----------------------------------------------------------------------------
def get_subject_data(signal_file, signal_names):
    #this_arousal, sleep   = import_arousals(arousal_file)
    this_signal    = import_signals(signal_file)
    #this_signal = this_signal[:, :-1] # This shouldn't have to  be here. This is removing signal info
    #this_data      = np.append(this_signal, this_arousal, axis=1)
    this_data      = pd.DataFrame(this_signal, index=None, columns=signal_names)
    return this_data

def get_subject_data_test(signal_file, signal_names):
    this_signal    = import_signals(signal_file)
    this_data      = this_signal
    this_data      = pd.DataFrame(this_data, index=None, columns=signal_names)
    return this_data
