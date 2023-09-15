import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import pickle
from random import seed
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import os


def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def lstm_model(lr, input, start, input_size):
    model = Sequential()
    model.add(LSTM(start, return_sequences=True, input_shape=(input, input_size),
                   kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)))
    model.add(Dropout(0.5))

    '''model.add(LSTM(start, return_sequences=True, input_shape=(input, input_size),
                   kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)))
    model.add(Dropout(0.5))
    
    model.add(LSTM(start, return_sequences=True, input_shape=(input, input_size),
                   kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)))
    model.add(Dropout(0.5))'''

    model.add(LSTM(start, return_sequences=False, input_shape=(input, input_size),
                   kernel_regularizer=l2(0.01), recurrent_regularizer=l2(0.01)))

    model.add(Dense(2, activation='softmax'))
    opt = Adam(lr=lr)  # lr=lr, decay=1e-6
    model.compile(loss='MSE', optimizer=opt, metrics=['accuracy', f1_m])

    #print(model.summary())
    return model


def step_decay(epoch):
    initial_lrate = 0.0001
    drop = 0.5  # in 0.5 it provided an accuracy of 80%+
    epochs_drop = 5.0  # 5.0 gives an optimal epochs_drop
    lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
    return lrate


def load_data(name):
    with open(name, 'rb') as f:
        loaded = pickle.load(f)
    return loaded


def get_data(case, data, KSS, subs_data):
    # Initialise data
    a_sub_num = []
    d_sub_num = []
    a_sub_trials = []
    d_sub_trials = []
    a = []
    d = []
    tr2_sub_num = []
    tr2_sub_trials = []
    tr2 = []

    if case == 3:
        from Splitting_Data_F_L import get_train_test_Behavioural, Awake_training, Drowsy_training

        # Subject data
        TEMP = np.zeros(len(subs_data))
        count = 0
        for abc in subs_data:
            # TEMP.append(abc.split('-')[1])
            TEMP[count] = int(abc.split('-')[1])
            count += 1
        a_data = data[np.where(TEMP == 1)[0], :, :]
        tr2_data = data[np.where(TEMP == 2)[0], :, :]
        d_data = data[np.where(TEMP == 3)[0], :, :]

        # Subject number and  drowsiness label
        temp = subs_data[np.where(TEMP == 1)[0]]
        temp_tr2 = subs_data[np.where(TEMP == 2)[0]]
        temp2 = subs_data[np.where(TEMP == 3)[0]]
        for abc in temp:
            a_sub_num.append(abc.split('-')[0])
            a_sub_trials.append(abc)
            a.append(0)
        for abc in temp_tr2:
            tr2_sub_num.append(abc.split('-')[0])
            tr2_sub_trials.append(abc)
            tr2.append(0)
        for abc in temp2:
            d_sub_num.append(abc.split('-')[0])
            d_sub_trials.append(abc)
            d.append(1)

    elif case == 2:
        from Splitting_Data_789_6 import get_train_test_Behavioural, Awake_training, Drowsy_training

        # Subject data
        a_data = data[np.where(KSS < 6)[0]]
        d_data = data[np.where(KSS > 6)[0]]

        # Subject number and  drowsiness label
        temp = subs_data[np.where(KSS < 6)[0]]
        temp2 = subs_data[np.where(KSS > 6)[0]]
        for abc in temp:
            a_sub_num.append(abc.split('-')[0])
            a_sub_trials.append(abc)
            a.append(0)
        for abc in temp2:
            d_sub_num.append(abc.split('-')[0])
            d_sub_trials.append(abc)
            d.append(1)

    elif case == 1:
        from Splitting_Data_789 import get_train_test_Behavioural, Awake_training, Drowsy_training

        # Subject data
        a_data = data[np.where(KSS < 7)[0]]
        d_data = data[np.where(KSS > 6)[0]]

        # Subject number and  drowsiness label
        temp = subs_data[np.where(KSS < 7)[0]]
        temp2 = subs_data[np.where(KSS > 6)[0]]
        for abc in temp:
            a_sub_num.append(abc.split('-')[0])
            a_sub_trials.append(abc)
            a.append(0)
        for abc in temp2:
            d_sub_num.append(abc.split('-')[0])
            d_sub_trials.append(abc)
            d.append(1)

    elif case == 4:
        from Splitting_Data_B89 import get_train_test, Awake_training, Drowsy_training

        # Subject data
        a_data = data[np.where(KSS < 8)[0]]
        d_data = data[np.where(KSS > 7)[0]]

        # Subject number and  drowsiness label
        temp = subs_data[np.where(KSS < 8)[0]]
        temp2 = subs_data[np.where(KSS > 7)[0]]
        for abc in temp:
            a_sub_num.append(abc.split('-')[0])
            a_sub_trials.append(abc)
            a.append(0)
        for abc in temp2:
            d_sub_num.append(abc.split('-')[0])
            d_sub_trials.append(abc)
            d.append(1)

    elif case == 5:
        from Splitting_Data_B89_7 import get_train_test, Awake_training, Drowsy_training

        # Subject data
        a_data = data[np.where(KSS < 7)[0]]
        d_data = data[np.where(KSS > 7)[0]]

        # Subject number and  drowsiness label
        temp = subs_data[np.where(KSS < 7)[0]]
        temp2 = subs_data[np.where(KSS > 7)[0]]
        for abc in temp:
            a_sub_num.append(abc.split('-')[0])
            a_sub_trials.append(abc)
            a.append(0)
        for abc in temp2:
            d_sub_num.append(abc.split('-')[0])
            d_sub_trials.append(abc)
            d.append(1)

    elif case == 6:
        from Splitting_Data_B89_67 import get_train_test, Awake_training, Drowsy_training

        # Subject data
        a_data = data[np.where(KSS < 6)[0]]
        d_data = data[np.where(KSS > 7)[0]]

        # Subject number and  drowsiness label
        temp = subs_data[np.where(KSS < 6)[0]]
        temp2 = subs_data[np.where(KSS > 7)[0]]
        for abc in temp:
            a_sub_num.append(abc.split('-')[0])
            a_sub_trials.append(abc)
            a.append(0)
        for abc in temp2:
            d_sub_num.append(abc.split('-')[0])
            d_sub_trials.append(abc)
            d.append(1)

    if case != 3:
        return a_data, a_sub_num, a, d_data, d_sub_num, d, get_train_test_Behavioural, Awake_training, Drowsy_training, \
               a_sub_trials, d_sub_trials
    else:
        return a_data, a_sub_num, a, d_data, d_sub_num, d, get_train_test_Behavioural, Awake_training, Drowsy_training, \
               a_sub_trials, d_sub_trials, tr2_data, tr2, tr2_sub_num, tr2_sub_trials


def get_callbacks(path):
    #EarlyStopping(monitor='my_metric', mode='max')
    return [tf.keras.callbacks.ModelCheckpoint(filepath=path, monitor="val_f1_m", save_weights_only=True,
                                               save_best_only=True, mode='max', verbose=1),
            tf.keras.callbacks.EarlyStopping(monitor="val_f1_m", patience=15, mode='max')]


def save_data(file_to_save, name):
    with open(name, 'wb') as f:
        pickle.dump(file_to_save, f)


#######################################################################################################################
# This script performs classification using LSTM
#######################################################################################################################
# These paths need amending
# The directory of sequences
load_directory = './Behavioural/Our_Method/Behavioural_Feature_Files/' # May need adjusting

# The directory to save weights + run_through (ie .../Weights' saves to the '.../Weights1' location on run_through = 1)
Weights_Path = './Weights'

# Run for both True and False
Half = True

# The number of images in the sequence (10)
sequence_length = 10

# The drowsiness label to use: 1 for KSS789, 2 for KSSNO6 and 3 for FL
case = 2

# When running multiple times, save the weights:
run_through = 1

#######################################################################################################################
os.chdir(load_directory)

# Load the existing features data
if Half:
    prefix = "_Half_"
else:
    prefix = "_"

loaded_data = load_data("ResNet101" + prefix + "Combined_features_s" + str(sequence_length) + ".pkl")
subs_data = load_data("ResNet101" + prefix + "Combined_subs_s" + str(sequence_length) + ".pkl")
KSS = load_data("ResNet101" + prefix + "Combined_KSS_s" + str(sequence_length) + ".pkl")

# Change this for testing removal of features
Data_use = 'All'
data = []
for a in range(len(loaded_data)):
    if Data_use == 'ResNet':
        data.append(loaded_data[a][:, :2048])
    elif Data_use == 'HC':
        data.append(loaded_data[a][:, 2048:2052])
    elif Data_use == 'MediaPipe':
        data.append(loaded_data[a][:, 2052:])
    elif Data_use == 'No Z':
        data.append(loaded_data[a][:, 2052:-468])
    elif Data_use == 'ResHC':
        data.append(loaded_data[a][:, :2052])
    elif Data_use == 'HCMedia':
        data.append(loaded_data[a][:, 2048:])
    elif Data_use == 'ResMedia':
        data.append(np.hstack((loaded_data[a][:, :2048], loaded_data[a][:, 2052:])))
    elif Data_use == 'All':
        data = loaded_data

input_size = len(data[1][1, :])

# Convert data to numpy array for easier formatting
data = np.array(data)
KSS = np.array(KSS)
subs_data = np.array(subs_data)

# Load data splits
if case != 3:
    a_data, a_sub_num, a, d_data, d_sub_num, d, get_train_test_Behavioural, Awake_training, Drowsy_training, a_sub_trials, \
    d_sub_trials = get_data(case, data, KSS, subs_data)
else:
    a_data, a_sub_num, a, d_data, d_sub_num, d, get_train_test_Behavioural, Awake_training, Drowsy_training, a_sub_trials, \
    d_sub_trials, tr2_data, tr2, tr2_sub_num, tr2_sub_trials = get_data(case, data, KSS, subs_data)

# Initialise variables
data_accuracy, trial_num, sub_num = [], [], []
val_acc = 0

# Run for each left out subject
for k in range(1, 15):
    print('Testing subject: ' + str(k) + '\n')

    # Make copies of the data for indexing
    a_data_copy = a_data.copy().tolist()
    d_data_copy = d_data.copy().tolist()
    a_copy = a.copy()
    d_copy = d.copy()
    a_sub_copy = a_sub_num.copy()
    d_sub_copy = d_sub_num.copy()
    a_sub_trials_copy = a_sub_trials.copy()
    d_sub_trials_copy = d_sub_trials.copy()

    # Store validation data
    a_valid_data = []
    d_valid_data = []
    a_valid_label = []
    d_valid_label = []
    a_valid_sub = []
    d_valid_sub = []
    a_valid_trials = []
    d_valid_trials = []

    # Repeat for trial two in the case of first vs last
    if case == 3:
        tr2_valid_sub = []
        tr2_valid_label = []
        tr2_valid_trials = []
        tr2_valid_data = []
        tr2_data_copy = tr2_data.copy().tolist()
        tr2_copy = tr2.copy()
        tr2_sub_copy = tr2_sub_num.copy()
        tr2_sub_trials_copy = tr2_sub_trials.copy()

    # Delete the leave one out subject data - may have issues if no drowsy data for subject?
    c = a_sub_copy.count(str(k))
    for abc in range(0, c):
        a_valid_data.append(a_data_copy[a_sub_copy.index(str(k))])
        del (a_data_copy[a_sub_copy.index(str(k))])
        a_valid_label.append(a_copy[a_sub_copy.index(str(k))])
        del (a_copy[a_sub_copy.index(str(k))])
        a_valid_trials.append(a_sub_trials_copy[a_sub_copy.index(str(k))])
        del (a_sub_trials_copy[a_sub_copy.index(str(k))])
        a_valid_sub.append(a_sub_copy[a_sub_copy.index(str(k))])
        del (a_sub_copy[a_sub_copy.index(str(k))])

    a_data_copy = np.asarray(a_data_copy)
    a_copy = np.asarray(a_copy)
    a_sub_copy = np.asarray(a_sub_copy)

    a_valid_data = np.asarray(a_valid_data)
    a_valid_label = np.asarray(a_valid_label)
    a_valid_sub = np.asarray(a_valid_sub)
    a_valid_trials = np.asarray(a_valid_trials)

    if case == 3:
        c = tr2_sub_copy.count(str(k))
        for abc in range(0, c):
            tr2_valid_data.append(tr2_data_copy[tr2_sub_copy.index(str(k))])
            del (tr2_data_copy[tr2_sub_copy.index(str(k))])
            tr2_valid_label.append(tr2_copy[tr2_sub_copy.index(str(k))])
            del (tr2_copy[tr2_sub_copy.index(str(k))])
            tr2_valid_trials.append(tr2_sub_trials_copy[tr2_sub_copy.index(str(k))])
            del (tr2_sub_trials_copy[tr2_sub_copy.index(str(k))])
            tr2_valid_sub.append(tr2_sub_copy[tr2_sub_copy.index(str(k))])
            del (tr2_sub_copy[tr2_sub_copy.index(str(k))])

        tr2_data_copy = np.asarray(tr2_data_copy)
        tr2_copy = np.asarray(tr2_copy)
        tr2_sub_copy = np.asarray(tr2_sub_copy)

        tr2_valid_data = np.asarray(tr2_valid_data)
        tr2_valid_label = np.asarray(tr2_valid_label)
        tr2_valid_sub = np.asarray(tr2_valid_sub)
        tr2_valid_trials = np.asarray(tr2_valid_trials)

    c = d_sub_copy.count(str(k))
    for abc in range(0, c):
        d_valid_data.append(d_data_copy[d_sub_copy.index(str(k))])
        del (d_data_copy[d_sub_copy.index(str(k))])
        d_valid_label.append(d_copy[d_sub_copy.index(str(k))])
        del (d_copy[d_sub_copy.index(str(k))])
        d_valid_trials.append(d_sub_trials_copy[d_sub_copy.index(str(k))])
        del (d_sub_trials_copy[d_sub_copy.index(str(k))])
        d_valid_sub.append(d_sub_copy[d_sub_copy.index(str(k))])
        del (d_sub_copy[d_sub_copy.index(str(k))])

    d_data_copy = np.asarray(d_data_copy)
    d_copy = np.asarray(d_copy)
    d_sub_copy = np.asarray(d_sub_copy)

    d_valid_data = np.asarray(d_valid_data)
    d_valid_label = np.asarray(d_valid_label)
    d_valid_sub = np.asarray(d_valid_sub)
    d_valid_trials = np.asarray(d_valid_trials)

    trainX = np.concatenate((a_data_copy, d_data_copy), axis=0)
    trainY = np.concatenate((a_copy, d_copy), axis=0)
    subs_stack = np.concatenate((a_sub_copy, d_sub_copy), axis=0)

    lb = LabelEncoder()

    case2, drowsy_subs = Drowsy_training(k, seeding=True)
    if case == 3:
        awake_subs, flag = Awake_training(k, case2, seeding=True)  # Flag required for 1st vs last
        train_x, test_x, train_y, test_y = get_train_test_Behavioural(awake_subs, drowsy_subs, trainX, trainY, k, subs_stack, flag)
    else:
        awake_subs = Awake_training(k, case2, seeding=True)
        train_x, test_x, train_y, test_y = get_train_test_Behavioural(awake_subs, drowsy_subs, trainX, trainY, k, subs_stack)

    train_x = np.concatenate(train_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    test_x = np.concatenate(test_x, axis=0)
    test_y = np.concatenate(test_y, axis=0)

    train_y = to_categorical(lb.fit_transform(train_y))
    test_y = to_categorical(lb.fit_transform(test_y))

    # Shuffle the inputs
    seed(1)
    shuffled_indices = np.random.permutation(len(train_y))  # return a permutation of the indices
    train_x = train_x[shuffled_indices, :, :]
    train_y = train_y[shuffled_indices, :]
    shuffled_indices = np.random.permutation(len(test_y))
    test_x = test_x[shuffled_indices, :, :]
    test_y = test_y[shuffled_indices, :]

    path = Weights_Path + str(run_through) + '/sub' + str(k) + 'SL' + str(sequence_length) + 'DrowsyC' + str(case)

    # Model Variables
    lr = 0.001
    initial = 4

    if case == 3:
        class_weight = {0: 1., 1: 1.}
    else:
        class_weight = {0: 1., 1: 30.}   # Worked poor for drowsy label when 1:1

    model = lstm_model(lr, sequence_length, initial, input_size)
    batch = 64
    epochs = 30

    history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                        epochs=epochs, batch_size=batch, class_weight=class_weight,
                        callbacks=get_callbacks(path))
    # Packages.plot_history(history)
    plt.plot(history.history['accuracy'], label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], label="Val Accuracy")
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Val Loss")
    plt.title("Training Accuracy/Loss")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.show()

    print("Drowsy Case: ", case)
    print("Sequence length: ", sequence_length)
    print("Batch size: ", batch)
    print("Initial Learning Rate: ", lr)

    save_data(a_valid_data, 'Eval' + str(k) + 'a.pkl')
    save_data(d_valid_data, 'Eval' + str(k) + 'd.pkl')
    model.load_weights(path)
    if a_valid_data.size > 0:
        predictionsA = (model.predict(a_valid_data) > 0.5).astype(int)
        trialA = np.unique(a_valid_trials)
        for tri in range(len(trialA)):
            temp = predictionsA[np.where(a_valid_trials == trialA[tri])[0]][:, 0]
            print('Trial: ', trialA[tri])
            print('Accuracy: ', sum(temp) / len(temp) * 100)
            data_accuracy.append([sum(temp) / len(temp) * 100])
            trial_num.append(trialA[tri])
            sub_num.append(max(history.history['val_f1_m']))

    if case == 3:
        if tr2_valid_data.size > 0:
            predictionsA = (model.predict(tr2_valid_data) > 0.5).astype(int)
            trialA = np.unique(tr2_valid_trials)
            for tri in range(len(trialA)):
                temp = predictionsA[np.where(tr2_valid_trials == trialA[tri])[0]][:, 0]
                print('Trial: ', trialA[tri])
                print('Accuracy: ', sum(temp) / len(temp) * 100)
                data_accuracy.append([sum(temp) / len(temp) * 100])
                trial_num.append(trialA[tri])
                sub_num.append(max(history.history['val_f1_m']))

    if d_valid_data.size > 0:
        predictionsD = (model.predict(d_valid_data) > 0.5).astype(int)
        trialD = np.unique(d_valid_trials)
        for tri in range(len(trialD)):
            temp = predictionsD[np.where(d_valid_trials == trialD[tri])[0]][:, 1]
            print('Trial: ', trialD[tri])
            print('Accuracy: ', sum(temp) / len(temp) * 100)
            data_accuracy.append([sum(temp) / len(temp) * 100])
            trial_num.append(trialD[tri])
            sub_num.append(max(history.history['val_f1_m']))
    print('NEXT')


for a in range(len(trial_num)):
    print("\n Trial:", trial_num[a])
    print("val_f1: ", sub_num[a]*100)
    print("Accuracy: ", data_accuracy[a][0])
