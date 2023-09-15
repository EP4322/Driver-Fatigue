import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from scipy.stats import shapiro
from scipy.stats import normaltest
from scipy.stats import ttest_rel
from scipy.stats import ttest_ind
from scipy.stats import wilcoxon
from scipy.stats import mannwhitneyu
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from matplotlib import pyplot
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier


def data_load(file_names, EEG=True, ECG=True, EMG=True, EOG=True):
    # Loads the saved arrays of features
    # Currently at least EEG or ECG must equal to True!!!!!

    names = ['Delta1', 'Theta1', 'Alpha1', 'Beta1', 'ApEnt1', 'SaEnt1', 'Sh_Ent1', 'fuzzy1', 'Multiscal1', 'Sp_ent1',
             'wave_ent1', 'EEG_mean1', 'EEG_std1', 'EEG_kurt1', 'EEG_var1', 'EEG_skew1', 'Delta2', 'Theta2', 'Alpha2',
             'Beta2', 'ApEnt2', 'SaEnt2', 'Sh_Ent2', 'fuzzy2', 'Multiscal2', 'Sp_ent2', 'wave_ent2', 'EEG_mean2',
             'EEG_std2', 'EEG_kurt2', 'EEG_var2', 'EEG_skew2', 'Delta3', 'Theta3', 'Alpha3', 'Beta3', 'ApEnt3',
             'SaEnt3', 'Sh_Ent3', 'fuzzy3', 'Multiscal3', 'Sp_ent3', 'wave_ent3', 'EEG_mean3', 'EEG_std3',
             'EEG_kurt3', 'EEG_var3', 'EEG_skew3', 'Delta4', 'Theta4', 'Alpha4', 'Beta4', 'ApEnt4', 'SaEnt4',
             'Sh_Ent4', 'fuzzy4', 'Multiscal4', 'Sp_ent4', 'wave_ent4', 'EEG_mean4', 'EEG_std4', 'EEG_kurt4',
             'EEG_var4', 'EEG_skew4', 'Delta5', 'Theta5', 'Alpha5', 'Beta5', 'ApEnt5', 'SaEnt5', 'Sh_Ent5', 'fuzzy5',
             'Multiscal5', 'Sp_ent5', 'wave_ent5', 'EEG_mean5', 'EEG_std5', 'EEG_kurt5', 'EEG_var5', 'EEG_skew5',
             'HR', 'VLF', 'LF', 'HF', 'LF_HF', 'Power', 'rsp', 'RRV_median', 'RRV_mean', 'RRV_ApEn', 'HR_std',
             'Sh_Ent', 'Approx', 'fuzzy', 'wave_ent', 'HRV_mean', 'HRV_std', 'HRV_kurt', 'HRV_var', 'HRV_skew']
    names2 = ['DeltaA1', 'ThetaA1', 'AlphaA1', 'BetaA1', 'DeltaR1', 'ThetaR1', 'AlphaR1', 'BetaR1', 'DeltaA2',
              'ThetaA2', 'AlphaA2', 'BetaA2', 'DeltaR2', 'ThetaR2', 'AlphaR2', 'BetaR2', 'DeltaA3', 'ThetaA3',
              'AlphaA3', 'BetaA3', 'DeltaR3', 'ThetaR3', 'AlphaR3', 'BetaR3', 'DeltaA4', 'ThetaA4', 'AlphaA4', 'BetaA4',
              'DeltaR4', 'ThetaR4', 'AlphaR4', 'BetaR4', 'DeltaA5', 'ThetaA5', 'AlphaA5', 'BetaA5', 'DeltaR5',
              'ThetaR5', 'AlphaR5', 'BetaR5']

    if EEG == False:
        if ECG == True:
            if ECG == True:
                open_file = open(file_names[-1], "rb")
                DROZY_ECG_EEG = pickle.load(open_file)
                index = DROZY_ECG_EEG.iloc[:, -1:]
                DROZY_ECG_EEG = DROZY_ECG_EEG.iloc[:, 0:-1]
                DROZY_ECG_EEG.columns = ['HR', 'VLF', 'LF', 'HF', 'LF_HF', 'Power', 'rsp', 'RRV_median', 'RRV_mean',
                                         'RRV_ApEn', 'HR_std', 'Sh_Ent', 'Approx', 'fuzzy', 'wave_ent', 'HRV_mean',
                                         'HRV_std', 'HRV_kurt', 'HRV_var', 'HRV_skew']
                B = DROZY_ECG_EEG.columns
                open_file.close()

    if ECG == False:
        if EEG == True:
            open_file = open(file_names[0], "rb")
            DROZY_ECG_EEG = pickle.load(open_file)
            index = DROZY_ECG_EEG.iloc[:, -1:]
            DROZY_ECG_EEG = DROZY_ECG_EEG.iloc[:, 0:-21]
            DROZY_ECG_EEG.columns = ['Delta1', 'Theta1', 'Alpha1', 'Beta1', 'ApEnt1', 'SaEnt1', 'Sh_Ent1', 'fuzzy1',
                                     'Multiscal1', 'Sp_ent1', 'wave_ent1', 'EEG_mean1', 'EEG_std1', 'EEG_kurt1',
                                     'EEG_var1', 'EEG_skew1', 'Delta2', 'Theta2', 'Alpha2', 'Beta2', 'ApEnt2', 'SaEnt2',
                                     'Sh_Ent2', 'fuzzy2', 'Multiscal2', 'Sp_ent2', 'wave_ent2', 'EEG_mean2', 'EEG_std2',
                                     'EEG_kurt2', 'EEG_var2', 'EEG_skew2', 'Delta3', 'Theta3', 'Alpha3', 'Beta3',
                                     'ApEnt3', 'SaEnt3', 'Sh_Ent3', 'fuzzy3', 'Multiscal3', 'Sp_ent3', 'wave_ent3',
                                     'EEG_mean3', 'EEG_std3', 'EEG_kurt3', 'EEG_var3', 'EEG_skew3', 'Delta4', 'Theta4',
                                     'Alpha4', 'Beta4', 'ApEnt4', 'SaEnt4', 'Sh_Ent4', 'fuzzy4', 'Multiscal4',
                                     'Sp_ent4', 'wave_ent4', 'EEG_mean4', 'EEG_std4', 'EEG_kurt4', 'EEG_var4',
                                     'EEG_skew4', 'Delta5', 'Theta5', 'Alpha5', 'Beta5', 'ApEnt5', 'SaEnt5', 'Sh_Ent5',
                                     'fuzzy5', 'Multiscal5', 'Sp_ent5', 'wave_ent5', 'EEG_mean5', 'EEG_std5',
                                     'EEG_kurt5', 'EEG_var5']
            B = DROZY_ECG_EEG.columns
            open_file.close()
            open_file = open(file_names[2], "rb")
            DROZY_Power = pickle.load(open_file)
            DROZY_Power = DROZY_Power.iloc[:, 0:-1]
            DROZY_Power.columns = names2
            B = B.append(DROZY_Power.columns)
            open_file.close()

    if EEG == True:
        if ECG == True:
            open_file = open(file_names[0], "rb")
            DROZY_ECG_EEG = pickle.load(open_file)
            index = DROZY_ECG_EEG.iloc[:, -1:]
            DROZY_ECG_EEG = DROZY_ECG_EEG.iloc[:, 0:-1]
            DROZY_ECG_EEG.columns = names
            B = DROZY_ECG_EEG.columns
            open_file.close()

            open_file = open(file_names[-1], "rb")
            DROZY_ECG_EEG_2 = pickle.load(open_file)
            DROZY_ECG_EEG.iloc[:, -20:] = DROZY_ECG_EEG_2.iloc[:, :-1]
            open_file.close()

            open_file = open(file_names[2], "rb")
            DROZY_Power = pickle.load(open_file)
            DROZY_Power = DROZY_Power.iloc[:, 0:-1]
            DROZY_Power.columns = names2
            B = B.append(DROZY_Power.columns)
            open_file.close()

    if EMG == True:
        open_file = open(file_names[1], "rb")
        DROZY_EMG = pickle.load(open_file)
        DROZY_EMG = DROZY_EMG.iloc[:, 0:-1]
        B = B.append(DROZY_EMG.columns)
        open_file.close()

    if EOG == True:
        open_file = open(file_names[3], "rb")
        DROZY_EOG = pickle.load(open_file)
        DROZY_EOG = DROZY_EOG.iloc[:, 0:-1]
        B = B.append(DROZY_EOG.columns)
        open_file.close()

    if (EEG == True and EMG == True and EOG == True):
        DROZY = np.hstack((DROZY_ECG_EEG, DROZY_Power, DROZY_EMG, DROZY_EOG, index))
    elif (EEG == False and ECG == True and EMG == True and EOG == True):
        DROZY = np.hstack((DROZY_ECG_EEG, DROZY_EMG, DROZY_EOG, index))
    elif (EEG == True and EMG == True and EOG == False):
        DROZY = np.hstack((DROZY_ECG_EEG, DROZY_Power, DROZY_EMG, index))
    elif (EEG == True and EMG == False and EOG == True):
        DROZY = np.hstack((DROZY_ECG_EEG, DROZY_Power, DROZY_EOG, index))
    elif (EEG == False and ECG == True and EMG == False and EOG == True):
        DROZY = np.hstack((DROZY_ECG_EEG, DROZY_EOG, index))
    elif (EEG == False and ECG == True and EMG == True and EOG == False):
        DROZY = np.hstack((DROZY_ECG_EEG, DROZY_EMG, index))
    elif (EEG == False and ECG == True and EMG == False and EOG == False):
        DROZY = np.hstack((DROZY_ECG_EEG, index))
    elif (EEG == True and EMG == False and EOG == False):
        DROZY = np.hstack((DROZY_ECG_EEG, DROZY_Power, index))
    Drozy_ind = index

    return DROZY, Drozy_ind, B


def norm(Drozy_trials, Drozy, Drozy_ind, n):
    # Normality test for shapiro and normal test methods
    # p <= alpha: reject H0, not normal.
    # p > alpha: fail to reject H0, normal.

    alpha = 0.05

    stats = np.zeros((len(Drozy_trials), len(Drozy.T)))
    pvals = np.zeros((len(Drozy_trials), len(Drozy.T)))
    ERRs = np.zeros((len(Drozy_trials), len(Drozy.T)))

    for a in range(len(Drozy.T) - 1):
        for b in range(len(Drozy_trials)):
            temp = np.where(Drozy_ind == Drozy_trials[b])
            temp2 = Drozy[np.array(temp[0]), a]
            stats[b, a], pvals[b, a] = shapiro(temp2)
            if pvals[b, a] <= alpha:
                ERRs[b, a] = 1

    # Find the Gaussian distributed values
    norm = np.zeros(len(Drozy.T) - 1)

    reject = -1 * np.ones(len(Drozy.T) - 1)
    for a in range(len(Drozy.T) - 1):
        norm[a] = sum(ERRs[:, a])

        if norm[a] > n:
            reject[a] = 1

    rejections = np.where(reject > 0)
    keep = np.where(reject < 0)

    # "rejections" is for a rank test, "keep" are for a regular T-Test
    return rejections, keep


def KSS(Drozy, ind, a, b, p, Type, Type2):
    # Run correct test for each type
    if Type == "Rel":
        if Type2 == "Gauss":
            stat, pval = ttest(a[0], b[0], Drozy, ind, Type)
            g = len(stat)
            stat = np.zeros((len(a), g))
            pval = np.zeros((len(a), g))
        elif Type2 == "Rank":
            stat, pval = ranktest(a[0], b[0], Drozy, ind, Type)
            g = len(stat)
            stat = np.zeros((len(a), g))
            pval = np.zeros((len(a), g))

        for c in range(len(a)):
            if Type2 == "Gauss":
                stat[c, :], pval[c, :] = ttest(a[c], b[c], Drozy, ind, Type)
            elif Type2 == "Rank":
                stat[c, :], pval[c, :] = ranktest(a[c], b[c], Drozy, ind, Type)

            temp = np.where((pval[c, :] < 0.05))[0]

            if c == 0:
                good = temp
            else:
                good = np.hstack((good, temp))
        # Find how many indices are referenced and how many times
        ind, counts = np.unique(good, return_counts=True)

        # Gets where p% of values are significant
        use = np.where(counts > np.round(len(a) * p))[0]

    elif Type == "Independent":
        if Type2 == "Gauss":
            stat, pval = ttest(a, b, Drozy, ind, Type)
        elif Type2 == "Rank":
            stat, pval = ranktest(a, b, Drozy, ind, Type)

        use = np.where(pval < 0.05)[0]
    return use


def ttest(a, b, Drozy, ind, Type):
    if Type == "Rel":
        aa = Drozy[np.where(ind == a)[0], :]
        bb = Drozy[np.where(ind == b)[0], :]
        stat, pval = ttest_rel(aa, bb, nan_policy='omit')
    elif Type == "Independent":
        for pp in range(len(a)):
            temp = Drozy[np.where(ind == a[pp])[0], :]
            if pp == 0:
                aa = temp
            else:
                aa = np.vstack((aa, temp))
        for pp in range(len(b)):
            temp = Drozy[np.where(ind == b[pp])[0], :]
            if pp == 0:
                bb = temp
            else:
                bb = np.vstack((bb, temp))
        stat, pval = ttest_ind(aa, bb, nan_policy='omit')
    return stat, np.array(pval)


def ranktest(a, b, Drozy, ind, Type):
    if Type == "Rel":
        aa = Drozy[np.where(ind == a)[0], :]
        bb = Drozy[np.where(ind == b)[0], :]
        stat = np.zeros(len(aa.T))
        pval = np.zeros(len(aa.T))
        for d in range(len(aa.T)):
            # Remove errors
            if sum(aa[:, d]) - sum(bb[:, d]) != 0:
                stat[d], pval[d] = wilcoxon(aa[:, d], y=bb[:, d])
    elif Type == "Independent":
        for pp in range(len(a)):
            temp = Drozy[np.where(ind == a[pp])[0], :]
            if pp == 0:
                aa = temp
            else:
                aa = np.vstack((aa, temp))
        for pp in range(len(b)):
            temp = Drozy[np.where(ind == b[pp])[0], :]
            if pp == 0:
                bb = temp
            else:
                bb = np.vstack((bb, temp))
        stat, pval = mannwhitneyu(aa, y=bb)
    return stat, np.array(pval)


def DROZY_ML(Drozy, a, b, All, COLS, time, base):
    tick = 0
    aa = np.unique(a)
    for dd in aa:
        if tick == 0:
            Wake = np.where(Drozy[:, -1] == dd)[0]
            tick += 1
            new_ind = dd*np.ones(base-time)
        else:
            Wake = np.hstack((Wake, np.where(Drozy[:, -1] == dd)[0]))
            new_ind = np.hstack((new_ind, dd * np.ones(base-time)))
    tick = 0
    bb = np.unique(b)
    for dd in bb:
        if tick == 0:
            Drowsy = np.where(Drozy[:, -1] == dd)[0]
            tick += 1
            new_ind = np.hstack((new_ind, dd * np.ones(base-time)))
        else:
            Drowsy = np.hstack((Drowsy, np.where(Drozy[:, -1] == dd)[0]))
            new_ind = np.hstack((new_ind, dd * np.ones(base-time)))
    inda = Drozy[Wake, -1]
    Wake = Drozy[Wake, :]
    indd = Drozy[Drowsy, -1]
    Drowsy = Drozy[Drowsy, :]
    Tog_ind = np.hstack((inda, indd))
    Tog_ind = Tog_ind[:, np.newaxis]
    Tog = np.vstack((Wake[:, All], Drowsy[:, All]))
    Tog = np.hstack((Tog, Tog_ind))

    newcols = np.hstack((COLS[All], "Trials"))
    df2 = pd.DataFrame(Tog, columns=newcols)
    df2.fillna(value=0, inplace=True)
    b = np.hstack((np.zeros(len(Wake)), np.ones(len(Drowsy))))
    b = b.astype(int)

    return df2, b


######################################################################################################################
# This is a feature trends script to determine what trend the statistically significant features follow.
# These are copied to clipboard and can CTRL V into a spreadsheet.
# -1 = decrease, 0 = no change and 1 = increase (with increasing drowsiness)
######################################################################################################################
# Quick Initialisation

# Set window size (15 or 30)
window = 30

# True if looking at first half of the trial, false for whole trial
Half = False

# Type of drowsiness label: KSS789, KSSNO6, FL
Label = 'FL'

# What feature sets to use (Must include ECG or EEG)
EEG = True  # If false, change splitting data EEG to false
ECG = True
EMG = True
EOG = True

# Normalise??
Normal = True

######################################################################################################################
# I don't normally touch these.
######################################################################################################################

# Plot features used in classification (non-drowsy vs drowsy) - There are some errors if True
Plot_feat = False

# Percentage threshold for significant changes in data
p = 0.5

# Number of features to use
feats_testing = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

# Use a validation set? (ALWAYS TRUE)
ValTF = True

# Control randomness of subject selection
seeding = True

# Feature selection to include Ttest
Ttest = True

######################################################################################################################
# Main code
######################################################################################################################
# Set the number of samples dependent on window size and load the relevant data
if window == 15:
    base = 40
    file_names = ["DROZY_ECG_EEG15.pkl", "DROZY_EMG15.pkl", "DROZY_Power15.pkl", "DROZY_EOG15.pkl",
                  "DROZY_ECG_final15.pkl"]
    if Half:
        time = 20
    else:
        time = 0
elif window == 30:
    base = 20
    file_names = ["DROZY_ECG_EEG30.pkl", "DROZY_EMG30.pkl", "DROZY_Power30.pkl", "DROZY_EOG30.pkl",
                  "DROZY_ECG_final30.pkl"]
    if Half:
        time = 10
    else:
        time = 0

time2 = base-time

if Label == 'FL':
    from Splitting_Data_4 import Drowsy_training, Awake_training, Subjects, ind_sub, get_train_test
elif Label == 'KSS789':
    from Splitting_Data_2 import Drowsy_training, Awake_training, Subjects, ind_sub, get_train_test
elif Label == 'KSSNO6':
    from Splitting_Data_3 import Drowsy_training, Awake_training, Subjects, ind_sub, get_train_test

# Collect saved features
Drozy, Drozy_ind, COLSD = data_load(file_names, EEG=EEG, ECG=ECG, EMG=EMG, EOG=EOG)

# Trial Numbers,
if Label == 'FL':
    Drozy_trials = np.array([1.1, 1.3, 10.1, 10.3, 11.1, 11.3, 12.1, 13.1, 14.1, 14.3, 2.1, 2.3,
                             3.1, 3.3, 4.1, 4.3, 5.1, 5.3, 6.1, 6.3, 7.3, 8.1, 8.3, 9.3])

    # Initialisation of awake and drowsy
    awake = np.array([1.1, 10.1, 11.1, 12.1, 13.1, 14.1, 2.1, 3.1, 4.1, 5.1, 6.1, 8.1])
    drowsy = np.array([1.3, 10.3, 11.3, 14.3, 2.3, 3.3, 4.3, 5.3, 6.3, 8.3, 7.3, 9.3])

    # Independent T-tests
    awake_T = np.array([1.1, 10.1, 11.1, 12.1, 13.1, 14.1, 2.1, 3.1, 4.1, 5.1, 6.1, 8.1])
    drowsy_T = np.array([1.3, 10.3, 11.3, 14.3, 2.3, 3.3, 4.3, 5.3, 6.3, 8.3, 7.3, 9.3])

elif Label == 'KSS789':
    Drozy_trials = np.array(
        [1.1, 1.2, 1.3, 10.1, 10.3, 11.1, 11.2, 11.3, 12.1, 13.1, 13.2, 14.1, 14.2, 14.3, 2.1, 2.2,
         2.3, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 7.2, 7.3, 8.1, 8.2, 8.3,
         9.2, 9.3])

    # Initialisation of awake and drowsy
    awake = np.array([1.1, 1.2, 10.1, 11.1, 12.1, 13.1, 13.2, 14.1, 2.1, 2.3, 3.1, 3.2, 3.3, 4.1, 5.1, 6.1, 6.2,
                      7.2, 8.1, 8.2, 9.2])
    drowsy = np.array([1.3, 10.3, 11.2, 11.3, 14.2, 14.3, 2.2, 4.2, 4.3, 5.2, 5.3, 6.3, 8.3, 7.3, 9.3])

    # Independent T-tests
    awake_T = np.array([12.1, 13.1, 13.2, 3.1, 3.2, 3.3])
    drowsy_T = np.array([1.3, 10.3, 11.2, 11.3, 14.2, 14.3, 2.2, 4.2, 4.3, 5.2, 5.3, 6.3, 8.3, 7.3, 9.3])

elif Label == 'KSSNO6':
    Drozy_trials = np.array([1.1, 1.3, 10.1, 10.3, 11.1, 11.2, 11.3, 12.1, 13.2, 14.1, 14.2, 14.3, 2.1, 2.2,
                             3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 7.2, 7.3, 8.1, 8.3, 9.3])

    # Initialisation of awake and drowsy
    awake = np.array([1.1, 10.1, 11.1, 12.1, 13.2, 14.1, 2.1, 3.1, 3.2, 3.3, 4.1, 5.1, 6.1, 6.2, 7.2, 8.1])
    drowsy = np.array([1.3, 10.3, 11.2, 11.3, 14.2, 14.3, 2.2, 4.2, 4.3, 5.2, 5.3, 6.3, 8.3, 7.3, 9.3])

    # Independent T-tests
    awake_T = np.array([12.1, 13.2, 3.1, 3.2, 3.3])
    drowsy_T = np.array([1.3, 10.3, 11.2, 11.3, 14.2, 14.3, 2.2, 4.2, 4.3, 5.2, 5.3, 6.3, 8.3, 7.3, 9.3])

Drozy_n = np.zeros((int(len(Drozy)/(base/time2)), len(Drozy.T)))
tick = 1
for pp in Drozy_trials:
    temp = Drozy[np.where(Drozy[:, -1] == pp)[0], :]
    temp2 = temp[0:time2, :]
    Drozy_n[(time2*tick-time2):time2*tick, :] = temp2
    tick += 1
Drozy = Drozy_n
ind = Drozy[:, -1]

# Normality testing of data
Rank, Gaussian = norm(Drozy_trials, Drozy, ind, 18)
data_gaus = Drozy[:, Gaussian[0]]
data_rank = Drozy[:, Rank[0]]

# Calculate what features to use
ind_feat_G = KSS(data_gaus, ind, awake_T, drowsy_T, p, "Independent", "Gauss")
ind_feat_R = KSS(data_rank, ind, awake_T, drowsy_T, p, "Independent", "Rank")

Gaus_important2 = np.array(Gaussian[0][ind_feat_G])
Rank_important2 = np.array(Rank[0][ind_feat_R])
Important_feat = np.unique(np.hstack((Gaus_important2, Rank_important2)))

# Normalise data
if Normal == True:
    tick2 = 0
    for p in range(1, 15):
        trials = ind_sub(p)
        tick = 0
        for pp in trials:
            tempa = Drozy[np.where(ind == pp)[0]]
            if tick == 0:
                temp = tempa
                indt = np.ones(base-time)*pp
            else:
                temp = np.vstack((temp, tempa))
                indt = np.hstack((indt, np.ones(base-time)*pp))
            tick += 1
        #indt = indt.T
        normt = np.zeros((len(temp), len(temp.T)-1))
        for a in range(len(temp.T)-1):
            if np.nanmax(temp[:, a]) - np.nanmin(temp[:, a]) != 0:
                normt[:, a] = (temp[:, a] - np.nanmin(temp[:, a])) / (np.nanmax(temp[:, a]) - np.nanmin(temp[:, a]))
            else:
                normt[:, a] = np.nan
        if tick2 == 0:
            indt = indt[:, np.newaxis]
            Drozy_norm = np.hstack((normt, indt))
        else:
            indt = indt[:, np.newaxis]
            Drozy_norm = np.vstack((Drozy_norm, np.hstack((normt, indt))))
        tick2 += 1
else:
    Drozy_norm = Drozy.copy()

# Get data in correct format
A, b = DROZY_ML(Drozy_norm, awake, drowsy, Important_feat, COLSD, time, base)

#######################################################################################################################
# Feature trends: flagged as significant in t-testing
#######################################################################################################################
awake_data = np.zeros((len(awake)*time2, len(Important_feat)))
count = 0
for aw in awake:
    temp = Drozy_norm[np.where(Drozy_norm[:, -1] == aw)[0], :]
    awake_data[count * time2:count * time2 + time2, :] = temp[:time2, Important_feat]
    count += 1
count = 0
drowsy_data = np.zeros((len(drowsy) * time2, len(Important_feat)))
for dr in drowsy:
    temp = Drozy_norm[np.where(Drozy_norm[:, -1] == dr)[0], :]
    drowsy_data[count * time2:count * time2 + time2, :] = temp[:time2, Important_feat]
    count += 1
Feat_names = COLSD[Important_feat]

trends = np.zeros((len(Important_feat), 1))
for trend in range(len(Important_feat)):
    # Look at the differences between the median values for each feature
    temp = np.nanmedian(drowsy_data[:, trend]) - np.nanmedian(awake_data[:, trend])
    if temp > 0:
        # Feature increase
        trends[trend] = 1
    elif temp < 0:
        # Feature decrease
        trends[trend] = -1
feature_trends = pd.DataFrame(np.transpose(trends), columns=Feat_names)
#feat_trending_all.append(np.transpose(trends))
#feat_trending_all_names.append(Feat_names)
#print(feature_trends)
feature_trends.to_clipboard()


