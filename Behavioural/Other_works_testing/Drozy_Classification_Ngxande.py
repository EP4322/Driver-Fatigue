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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

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
            tr2.append(1)
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
# Code to test Ngxande method with FL labelling, 55% training, 15% validation and 30% held out
#######################################################################################################################
load_directory = './Behavioural/Our_Method/Behavioural_Feature_Files/' # May need adjusting
Weights_path = './OtherWeights' # May need adjusting

# Load the existing features data
sequence_length = 10
case = 3  # 1 for 789 drowsy, 2 for 6 removed and 789 drowsy and 3 for first vs last
run_through = 1  # Keeping track of best weights
Half = False

#######################################################################################################################
os.chdir(load_directory)

if Half:
    prefix = "_Half_"
else:
    prefix = "_"

loaded_data = load_data("ResNet101" + prefix + "Combined_features_s" + str(sequence_length) + ".pkl")
subs_data = load_data("ResNet101" + prefix + "Combined_subs_s" + str(sequence_length) + ".pkl")
KSS = load_data("ResNet101" + prefix + "Combined_KSS_s" + str(sequence_length) + ".pkl")

data = loaded_data

input_size = len(data[1][1, :])

# Convert data to numpy array for easier formatting
data = np.array(data)
KSS = np.array(KSS)
subs_data = np.array(subs_data)
data_accuracy, trial_num, sub_num = [], [], []

val_acc = 0

# Split train and test
tr1_data, tr1_sub_num, tr1, tr3_data, tr3_sub_num, tr3, get_train_test_Behavioural, TR1_training, TR2_training,\
tr1_sub_trials, tr3_sub_trials, tr2_data, tr2, tr2_sub_num, tr2_sub_trials = get_data(case, data, KSS, subs_data)

tr1_data = tr1_data.tolist()
tr2_data = tr2_data.tolist()
tr3_data = tr3_data.tolist()

model_data = np.concatenate([tr2_data, tr1_data], axis=0)
model_labels = np.concatenate([tr2, tr1], axis=0)
Train_x, Test_x, Train_y, Test_y = train_test_split(model_data, model_labels, test_size=0.3)
train_x, test_x, train_y, test_y = train_test_split(Train_x, Train_y, test_size=0.15)

lb = LabelEncoder()
train_y = to_categorical(lb.fit_transform(train_y))
test_y = to_categorical(lb.fit_transform(test_y))

path = Weights_path + str(run_through) + '/SL' + str(sequence_length) + 'DrowsyC' + str(case)

# Model Variables
lr = 0.001
initial = 4

class_weight = {0: 1., 1: 1.}

model = lstm_model(lr, sequence_length, initial, input_size)
batch = 64
epochs = 30

history = model.fit(train_x, train_y, validation_data=(test_x, test_y),
                    epochs=epochs, batch_size=batch, class_weight=class_weight,
                    callbacks=get_callbacks(path))

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


model.load_weights(path)
predictions = (model.predict(np.asarray(Test_x)) > 0.5).astype(int)
temp = predictions[:][:, 1]
accuracy = accuracy_score(Test_y, temp)
print("Confusion matrix for remaining data: ")
print(confusion_matrix(Test_y, temp))
print("Accuracy on remaining data: ", accuracy)


predictions = (model.predict(np.asarray(tr3_data)) > 0.5).astype(int)
temp = predictions[:][:, 1]
accuracy = accuracy_score(tr3, temp)
print("Accuracy on TR3: ", accuracy)

