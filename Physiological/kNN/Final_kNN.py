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
                                     'EEG_kurt5', 'EEG_var5', 'EEG_skew5']
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

        if EEG == False:
            index = DROZY_EOG.iloc[:, -1:]

        DROZY_EOG = DROZY_EOG.iloc[:, 0:-1]

        if EEG == False:
            B = DROZY_EOG.columns
        else:
            B = B.append(DROZY_EOG.columns)
        open_file.close()

    # Stack the required data
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
    elif (EEG == False and ECG == False and EOG == True and EMG == False):
        DROZY = np.hstack((DROZY_EOG, index))
    Drozy_ind = index

    return DROZY, Drozy_ind, B


def DROZY_ML(Drozy, a, b, All, COLS, time, base):
    # This function is to format the data into a dataframe for the features and "b" as the label
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


def get_features(A, b, sub_sep_awake, sub_sep_drowsy, Valid, seeding):
    # Split data into training and testing
    case, drowsy_subs = Drowsy_training(Valid, valid=True, seeding=seeding)
    if Label == 'FL':
        awake_subs, flag = Awake_training(Valid, case, seeding=seeding)
        X_train, X_validation, Y_train, Y_validation = get_train_test(awake_subs, drowsy_subs, A.to_numpy(),
                                                                      sub_sep_awake, sub_sep_drowsy, time, base, flag)
    else:
        awake_subs = Awake_training(Valid, case, seeding=seeding)
        X_train, X_validation, Y_train, Y_validation = get_train_test(awake_subs, drowsy_subs, A.to_numpy(),
                                                                      sub_sep_awake, sub_sep_drowsy, time, base)

    X_train = X_train[:, :-1]
    X_validation = X_validation[:, :-1]

    Y_train = Y_train.astype(int)
    Y_validation = Y_validation.astype(int)

    return X_train, X_validation, Y_train, Y_validation


#######################################################################################################################
# This is the kNN Final Script to make the final models with preset features. Model results are output.
#######################################################################################################################
# Quick Initialisation

# Set window size (15 or 30)
window = 30

# True if looking at first half of the trial, false for whole trial
Half = False

# Type of drowsiness label: KSS789, KSSNO6, FL
Label = 'KSS789'

# Normalise??
Normal = True

# Number of k-Nearest Neighbours
k = 20

# Features to use
Features = ['AlphaA5', 'AlphaR5', 'Alpha5', 'AlphaR4', 'AlphaR3']

# Save the model?
save_model = False
Name = 'EEG_30s_model_val'  # Script adds the left out subject number to the end

######################################################################################################################
# I don't normally touch these.
######################################################################################################################
# Use a validation set? - to compare features vs accuracy
ValTF = True

# Control randomness of subject selection
seeding = True

# Feature selection to include Ttest
Ttest = True

# Plot features used in classification (non-drowsy vs drowsy)
Plot_feat = False

# What feature sets to use (Must include ECG or EEG)
EEG = True  # If only ECG, change splitting data EEG to false
ECG = True
EMG = True
EOG = True
#######################################################################################################################
# Main code
#######################################################################################################################
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

# Collect data Partitioning
if Label == 'FL':
    from Splitting_Data_4 import Drowsy_training, Awake_training, Subjects, ind_sub, get_train_test
elif Label == 'KSS789':
    from Splitting_Data_2 import Drowsy_training, Awake_training, Subjects, ind_sub, get_train_test
elif Label == 'KSSNO6':
    from Splitting_Data_3 import Drowsy_training, Awake_training, Subjects, ind_sub, get_train_test


# Run for each subject left out
for Valid in range(1, 15):
    print('\n SUBJECT: ', Valid)

    # Collect saved features
    Drozy, Drozy_ind, COLSD = data_load(file_names, EEG=EEG, ECG=ECG, EMG=EMG, EOG=EOG)
    Important_feat = np.empty(len(Features))
    for f in range(0, len(Features)):
        Important_feat[f] = np.where(COLSD == Features[f])[0]
    Important_feat = Important_feat.astype(int)

    # Trial Numbers,
    if Label == 'FL':
        Drozy_trials = np.array([1.1, 1.3, 10.1, 10.3, 11.1, 11.3, 12.1, 13.1, 14.1, 14.3, 2.1, 2.3,
                                 3.1, 3.3, 4.1, 4.3, 5.1, 5.3, 6.1, 6.3, 7.3, 8.1, 8.3, 9.3])

        # Removal of validation data
        sub_sep_awake, sub_sep_drowsy = Subjects(Valid)
        sub = ind_sub(Valid)

        # Initialisation of awake and drowsy
        awake = np.array([1.1, 10.1, 11.1, 12.1, 13.1, 14.1, 2.1, 3.1, 4.1, 5.1, 6.1, 8.1])
        awake = np.unique(np.append(awake, sub_sep_awake))
        drowsy = np.array([1.3, 10.3, 11.3, 14.3, 2.3, 3.3, 4.3, 5.3, 6.3, 8.3, 7.3, 9.3])

        # Independent T-tests
        awake_T = np.array([1.1, 10.1, 11.1, 12.1, 13.1, 14.1, 2.1, 3.1, 4.1, 5.1, 6.1, 8.1])
        drowsy_T = np.array([1.3, 10.3, 11.3, 14.3, 2.3, 3.3, 4.3, 5.3, 6.3, 8.3, 7.3, 9.3])

    elif Label == 'KSS789':
        Drozy_trials = np.array(
            [1.1, 1.2, 1.3, 10.1, 10.3, 11.1, 11.2, 11.3, 12.1, 13.1, 13.2, 14.1, 14.2, 14.3, 2.1, 2.2,
             2.3, 3.1, 3.2, 3.3, 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 6.1, 6.2, 6.3, 7.2, 7.3, 8.1, 8.2, 8.3,
             9.2, 9.3])

        # Removal of validation data
        sub_sep_awake, sub_sep_drowsy = Subjects(Valid)
        sub = ind_sub(Valid)

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

        # Removal of validation data
        sub_sep_awake, sub_sep_drowsy = Subjects(Valid)
        sub = ind_sub(Valid)

        # Initialisation of awake and drowsy
        awake = np.array([1.1, 10.1, 11.1, 12.1, 13.2, 14.1, 2.1, 3.1, 3.2, 3.3, 4.1, 5.1, 6.1, 6.2, 7.2, 8.1])
        drowsy = np.array([1.3, 10.3, 11.2, 11.3, 14.2, 14.3, 2.2, 4.2, 4.3, 5.2, 5.3, 6.3, 8.3, 7.3, 9.3])

        # Independent T-tests
        awake_T = np.array([12.1, 13.2, 3.1, 3.2, 3.3])
        drowsy_T = np.array([1.3, 10.3, 11.2, 11.3, 14.2, 14.3, 2.2, 4.2, 4.3, 5.2, 5.3, 6.3, 8.3, 7.3, 9.3])

    # Format the data
    Drozy_n = np.zeros((int(len(Drozy)/(base/time2)), len(Drozy.T)))
    tick = 1
    for pp in Drozy_trials:
        temp = Drozy[np.where(Drozy[:, -1] == pp)[0], :]
        temp2 = temp[0:time2, :]
        Drozy_n[(time2*tick-time2):time2*tick, :] = temp2
        tick += 1
    Drozy = Drozy_n
    ind = Drozy[:, -1]

    # Remove validation subject from data
    if ValTF == True:
        for pp in sub:
            # Remove from indexing trial sets - awake vs drowsy
            awake = awake[np.where(awake != pp)[0]]
            drowsy = drowsy[np.where(drowsy != pp)[0]]

    # Find the relevant features using statistical means
    if Ttest == True:
        # Normalise the data
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
        # Validation set
        if ValTF == True:
            A2, b2 = DROZY_ML(Drozy_norm, sub_sep_awake, sub_sep_drowsy, Important_feat, COLSD, time, base)



        # Get data in the right format (split data)
        X_train, X_validation, Y_train, Y_validation = get_features(A, b, sub_sep_awake, sub_sep_drowsy, Valid, seeding)

        model = KNeighborsClassifier(n_neighbors=k)
        # Fit the model and make predictions
        model.fit(X_train, Y_train)
        predictions = model.predict(X_validation)
        acc1 = accuracy_score(Y_validation, predictions)
        print(acc1)

        if save_model == True:
            model_save = Name + str(Valid) + '.sav'
            pickle.dump(model, open(model_save, 'wb'))

    # Classify left out subject
    if ValTF == True:
        # Validation set features we want
        X2 = A2.iloc[:, :-1]
        print(X2.columns)

        predictions2 = model.predict(X2.iloc[0:base - time, :])
        # Print validation subject predictions
        print("Validating Subject")
        print(accuracy_score(b2[0:base - time], predictions2))

        if len(sub) > 1:
            predictions2 = model.predict(X2.iloc[base - time:base * 2 - time * 2, :])
            # Print validation subject predictions
            print(accuracy_score(b2[base - time:base * 2 - time * 2], predictions2))
        if len(sub) > 2:
            predictions2 = model.predict(X2.iloc[base * 2 - time * 2:base * 3 - time * 3, :])
            # Print validation subject predictions
            print(accuracy_score(b2[base * 2 - time * 2:base * 3 - time * 3], predictions2))

    # Plot non drowsy vs drowsy (Not used in model development, just if interested - not fully commented)
    if Plot_feat == True:
        # Plots the validation features and their relationship to drowsiness
        X = X_validation  # np.hstack([X_train, X_validation])
        y = Y_validation  # np.hstack([Y_train, Y_validation])
        Awake_Compare = X[np.where(y == 0)[0], :]
        Drowsy_Compare = X[np.where(y == 1)[0], :]
        Types = X2.columns

        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
        count = 0
        axer = [ax1, ax2, ax3, ax4, ax5, ax6]
        while count < len(Features) and count < 6:
            axer[count].boxplot([Awake_Compare[:, count], Drowsy_Compare[:, count]])
            axer[count].set_title(Types[count])

            count += 1

            fig.suptitle("Features")
            fig.show()

