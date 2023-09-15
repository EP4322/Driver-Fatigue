import numpy as np
import random
from random import seed


# Switch for ECG only
EEG = True

# This function generates a random set of drowsy training subjects to keep 30% testing
def Drowsy_training(a, valid=True, seeding=True):
    # For repetition
    if seeding == True:
        seed(1)

    # Subjects with drowsy data
    subs_array = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14])
    sub1 = a
    sub2 = a
    sub3 = a
    while a == sub1:
        temp = random.randint(0, 11)
        sub1 = subs_array[temp]
        while a == sub2 or sub1 == sub2 or ((sub1 == 7 or sub1 == 9) and (sub2 == 7 or sub2 == 9)):
            temp = random.randint(0, 11)
            sub2 = subs_array[temp]
            while a == sub3 or sub3 == sub2 or sub3 == sub1 or ((sub1 == 7 or sub1 == 9) and (sub3 == 7 or sub3 == 9))\
                    or ((sub2 == 7 or sub2 == 9) and (sub3 == 7 or sub3 == 9)):
                temp = random.randint(0, 11)
                sub3 = subs_array[temp]
    subs = np.hstack((sub1, sub2, sub3))

    if sub1 == 7 or sub1 == 9 or sub2 == 7 or sub2 == 9 or sub3 == 7 or sub3 == 9:
        case2 = 1
    else:
        case2 = 0

    return case2, subs


# This function generates a random set of non drowsy training subjects to keep 30% testing
def Awake_training(a, case2, seeding=True):
    # For repetition
    if seeding == True:
        seed(1)

    flag = False
    if case2 == 1:
        subs = a
        sub = np.array([12, 13])
        while a == subs:
            temp = random.randint(0, 1)
            subs = sub[temp]
    else:
        subs = np.empty(1)
        flag = True
    # The non drowsy subjects that contribute to 30% testing is generated
    return subs, flag


# This function generates the trial numbers (non drowsy and drowsy) from the subject number and accepts an array or
# an integer
def Subjects(nums):
    idx = np.zeros(1)
    idx2 = np.zeros(1)
    if np.isscalar(nums):
        a = nums
        if a == 1:
            temp = np.array([1.1])
            idx = np.hstack((idx, temp))
            temp2 = np.array([1.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 2:
            temp = np.array([2.1])
            idx = np.hstack((idx, temp))
            temp2 = np.array([2.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 3:
            temp = np.array([3.1])
            idx = np.hstack((idx, temp))
            temp2 = np.array([3.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 4:
            temp = np.array([4.1])
            temp2 = np.array([4.3])
            idx = np.hstack((idx, temp))
            idx2 = np.hstack((idx2, temp2))
        elif a == 5:
            temp = np.array([5.1])
            temp2 = np.array([5.3])
            idx = np.hstack((idx, temp))
            idx2 = np.hstack((idx2, temp2))
        elif a == 6:
            if EEG:
                temp = np.array([6.1])
                idx = np.hstack((idx, temp))
            temp2 = np.array([6.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 7:
            temp2 = np.array([7.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 8:
            temp = np.array([8.1])
            temp2 = np.array([8.3])
            idx = np.hstack((idx, temp))
            idx2 = np.hstack((idx2, temp2))
        elif a == 9:
            temp2 = np.array([9.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 10:
            temp = np.array([10.1])
            idx = np.hstack((idx, temp))
            temp2 = np.array([10.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 11:
            temp = np.array([11.1])
            idx = np.hstack((idx, temp))
            temp2 = np.array([11.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 12:
            temp = np.array([12.1])
            idx = np.hstack((idx, temp))
        elif a == 13:
            temp = np.array([13.1])
            idx = np.hstack((idx, temp))
        elif a == 14:
            temp = np.array([14.1])
            temp2 = np.array([14.3])
            idx = np.hstack((idx, temp))
            idx2 = np.hstack((idx2, temp2))

    else:
        for a in nums:
            if a == 1:
                temp = np.array([1.1])
                idx = np.hstack((idx, temp))
                temp2 = np.array([1.3])
                idx2 = np.hstack((idx2, temp2))
            elif a == 2:
                temp = np.array([2.1])
                idx = np.hstack((idx, temp))
                temp2 = np.array([2.3])
                idx2 = np.hstack((idx2, temp2))
            elif a == 3:
                temp = np.array([3.1])
                idx = np.hstack((idx, temp))
                temp2 = np.array([3.3])
                idx2 = np.hstack((idx2, temp2))
            elif a == 4:
                temp = np.array([4.1])
                temp2 = np.array([4.3])
                idx = np.hstack((idx, temp))
                idx2 = np.hstack((idx2, temp2))
            elif a == 5:
                temp = np.array([5.1])
                temp2 = np.array([5.3])
                idx = np.hstack((idx, temp))
                idx2 = np.hstack((idx2, temp2))
            elif a == 6:
                if EEG:
                    temp = np.array([6.1])
                    idx = np.hstack((idx, temp))
                temp2 = np.array([6.3])
                idx2 = np.hstack((idx2, temp2))
            elif a == 7:
                temp2 = np.array([7.3])
                idx2 = np.hstack((idx2, temp2))
            elif a == 8:
                temp = np.array([8.1])
                temp2 = np.array([8.3])
                idx = np.hstack((idx, temp))
                idx2 = np.hstack((idx2, temp2))
            elif a == 9:
                temp2 = np.array([9.3])
                idx2 = np.hstack((idx2, temp2))
            elif a == 10:
                temp = np.array([10.1])
                idx = np.hstack((idx, temp))
                temp2 = np.array([10.3])
                idx2 = np.hstack((idx2, temp2))
            elif a == 11:
                temp = np.array([11.1])
                idx = np.hstack((idx, temp))
                temp2 = np.array([11.3])
                idx2 = np.hstack((idx2, temp2))
            elif a == 12:
                temp = np.array([12.1])
                idx = np.hstack((idx, temp))
            elif a == 13:
                temp = np.array([13.1])
                idx = np.hstack((idx, temp))
            elif a == 14:
                temp = np.array([14.1])
                temp2 = np.array([14.3])
                idx = np.hstack((idx, temp))
                idx2 = np.hstack((idx2, temp2))
    if len(idx) != 1:
        idx = np.delete(idx, 0)
    else:
        idx = np.empty(1)

    if len(idx2) > 1:
        idx2 = np.delete(idx2, 0)
    else:
        idx2 = np.empty(1)
    # The awake subjects and drowsy subject arrays are returned
    return idx, idx2


# This generates the array based on the subject integer
def ind_sub(Valid):
    if Valid == 1:
        sub = np.array([1.1, 1.3])
    elif Valid == 2:
        sub = np.array([2.1, 2.3])
    elif Valid == 3:
        sub = np.array([3.1, 3.3])
    elif Valid == 4:
        sub = np.array([4.1, 4.3])
    elif Valid == 5:
        sub = np.array([5.1, 5.3])
    elif Valid == 6:
        if EEG:
            sub = np.array([6.1, 6.3])
        else:
            sub = np.array([6.3])
    elif Valid == 7:
        sub = np.array([7.3])
    elif Valid == 8:
        sub = np.array([8.1, 8.3])
    elif Valid == 9:
        sub = np.array([9.3])
    elif Valid == 10:
        sub = np.array([10.1, 10.3])
    elif Valid == 11:
        sub = np.array([11.1, 11.3])
    elif Valid == 12:
        sub = np.array([12.1])
    elif Valid == 13:
        sub = np.array([13.1])
    elif Valid == 14:
        sub = np.array([14.1, 14.3])
    return sub


# This function splits the data into training and testing data (referred to as "validation")
def get_train_test_Physiological(awake_subs, drowsy_subs, data, valid_trials_awake, valid_trials_drowsy, time, base,
                                 flag):
    ind = data[:, -1]

    if flag == True:
        # Get trials of subjects
        D_awake_subs, drowsy_subs_idx = Subjects(drowsy_subs)
        # Get the validation data subjects
        X_validation_idx = np.hstack((D_awake_subs, drowsy_subs_idx))
        Y_validation = np.hstack((np.zeros(len(D_awake_subs) * (base - time)),
                                  np.ones(len(drowsy_subs_idx) * (base - time))))
    else:
        # Get trials of subjects
        awake_subs_idx, temp = Subjects(awake_subs)
        D_awake_subs, drowsy_subs_idx = Subjects(drowsy_subs)
        # Get the validation data subjects
        X_validation_idx = np.hstack((awake_subs_idx, D_awake_subs, drowsy_subs_idx))
        Y_validation = np.hstack((np.zeros(len(awake_subs_idx)*(base-time)), np.zeros(len(D_awake_subs)*(base-time)),
                             np.ones(len(drowsy_subs_idx)*(base-time))))

    # Set our awake and asleep trials
    if EEG:
        awake = np.array([1.1, 10.1, 11.1, 12.1, 13.1, 14.1, 2.1, 3.1, 4.1, 5.1, 6.1, 8.1])
    else:
        awake = np.array([1.1, 10.1, 11.1, 12.1, 13.1, 14.1, 2.1, 3.1, 4.1, 5.1, 8.1])
    drowsy = np.array([1.3, 10.3, 11.3, 14.3, 2.3, 3.3, 4.3, 5.3, 6.3, 8.3, 7.3, 9.3])

    # Remove the testing/final validation indices
    Trials_awake = awake
    Trials_drowsy = drowsy
    for a in valid_trials_awake:
        Trials_awake = Trials_awake[np.where(Trials_awake != a)[0]]
    for a in valid_trials_drowsy:
        Trials_drowsy = Trials_drowsy[np.where(Trials_drowsy != a)[0]]

    # Remove the subjects used for validation/testing
    X_train_awake = Trials_awake
    X_train_drowsy = Trials_drowsy
    for a in X_validation_idx:
        X_train_awake = X_train_awake[np.where(X_train_awake != a)[0]]
    for a in X_validation_idx:
        X_train_drowsy = X_train_drowsy[np.where(X_train_drowsy != a)[0]]

    # Stack the remaining data
    X_train_idx = np.hstack((X_train_awake, X_train_drowsy))
    Y_train = np.hstack((np.zeros(len(X_train_awake)*(base-time)), np.ones(len(X_train_drowsy)*(base-time))))

    for a in range(len(X_train_idx)):
        if a == 0:
            X_train = data[np.where(ind == X_train_idx[a])[0], :]

        else:
            X_train = np.vstack((X_train, data[np.where(ind == X_train_idx[a])[0], :]))

    for a in range(len(X_validation_idx)):
        if a == 0:
            X_validation = data[np.where(ind == X_validation_idx[a])[0], :]
        else:
            X_validation = np.vstack((X_validation, data[np.where(ind == X_validation_idx[a])[0], :]))

    return X_train, X_validation, Y_train.T, Y_validation.T


def Valid_Subjects(nums):
    idx = np.zeros(1)
    idx2 = np.zeros(1)
    if np.isscalar(nums):
        a = nums
        if a == 1:
            temp = np.array([1.1, 1.2])
            idx = np.hstack((idx, temp))
            temp2 = np.array([1.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 2:
            temp = np.array([2.1, 2.2])
            idx = np.hstack((idx, temp))
            temp2 = np.array([2.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 3:
            temp = np.array([3.1, 3.2])
            idx = np.hstack((idx, temp))
            temp2 = np.array([3.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 4:
            temp = np.array([4.1, 4.2])
            temp2 = np.array([4.3])
            idx = np.hstack((idx, temp))
            idx2 = np.hstack((idx2, temp2))
        elif a == 5:
            temp = np.array([5.1, 5.2])
            temp2 = np.array([5.3])
            idx = np.hstack((idx, temp))
            idx2 = np.hstack((idx2, temp2))
        elif a == 6:
            if EEG:
                temp = np.array([6.1, 6.2])
            else:
                temp = np.array([6.2])
            idx = np.hstack((idx, temp))
            temp2 = np.array([6.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 7:
            temp = np.array([7.2])
            idx = np.hstack((idx, temp))
            temp2 = np.array([7.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 8:
            temp = np.array([8.1, 8.2])
            temp2 = np.array([8.3])
            idx = np.hstack((idx, temp))
            idx2 = np.hstack((idx2, temp2))
        elif a == 9:
            temp = np.array([9.2])
            idx = np.hstack((idx, temp))
            temp2 = np.array([9.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 10:
            temp = np.array([10.1])
            idx = np.hstack((idx, temp))
            temp2 = np.array([10.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 11:
            temp = np.array([11.1, 11.2])
            idx = np.hstack((idx, temp))
            temp2 = np.array([11.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 12:
            temp = np.array([12.1])
            idx = np.hstack((idx, temp))
        elif a == 13:
            temp = np.array([13.1, 13.2])
            idx = np.hstack((idx, temp))
        elif a == 14:
            temp = np.array([14.1, 14.2])
            temp2 = np.array([14.3])
            idx = np.hstack((idx, temp))
            idx2 = np.hstack((idx2, temp2))

    if len(idx) != 1:
        idx = np.delete(idx, 0)
    else:
        idx = np.empty(1)

    if len(idx2) > 1:
        idx2 = np.delete(idx2, 0)
    else:
        idx2 = np.empty(1)
    # The awake subjects and drowsy subject arrays are returned
    return idx, idx2

def Valid_ind_sub(Valid):
    if Valid == 1:
        sub = np.array([1.1, 1.2, 1.3])
    elif Valid == 2:
        sub = np.array([2.1, 2.2, 2.3])
    elif Valid == 3:
        sub = np.array([3.1, 3.2, 3.3])
    elif Valid == 4:
        sub = np.array([4.1, 4.2, 4.3])
    elif Valid == 5:
        sub = np.array([5.1, 5.2, 5.3])
    elif Valid == 6:
        if EEG:
            sub = np.array([6.1, 6.2, 6.3])
        else:
            sub = np.array([6.2, 6.3])
    elif Valid == 7:
        sub = np.array([7.2, 7.3])
    elif Valid == 8:
        sub = np.array([8.1, 8.2, 8.3])
    elif Valid == 9:
        sub = np.array([9.2, 9.3])
    elif Valid == 10:
        sub = np.array([10.1, 10.3])
    elif Valid == 11:
        sub = np.array([11.1, 11.2, 11.3])
    elif Valid == 12:
        sub = np.array([12.1])
    elif Valid == 13:
        sub = np.array([13.1, 13.2])
    elif Valid == 14:
        sub = np.array([14.1, 14.2, 14.3])
    return sub



def get_train_test_Behavioural(awake_subs, drowsy_subs, data, label, valid, subs_array, flag):

    # Testing subs = awake_subs, drowsy_subs
    # Train subs = remainder
    sub = list(range(1, 15))

    # remove validation from the list
    sub.remove(valid)

    # Intialise training
    training = sub.copy()

    # Get training subs
    if flag == True:
        testing = drowsy_subs
    else:
        testing = np.hstack((awake_subs, drowsy_subs))

    for a in testing:
        training.remove(a)

    X_train = []
    Y_train = []
    subs_train = []  # Check is subs correct
    X_validation = []
    Y_validation = []
    subs_test = []  # Check is subs correct

    # data = np.asarray(data, dtype=np.float32)
    subs_array = np.asarray(subs_array, dtype=np.float32)

    for a in testing:
        X_validation.append(data[np.where(subs_array == a)[0], :, :])
        Y_validation.append(label[np.where(subs_array == a)[0]])
        subs_test.append(subs_array[np.where(subs_array == a)[0]])

    for a in training:
        X_train.append(data[np.where(subs_array == a)[0], :, :])
        Y_train.append(label[np.where(subs_array == a)[0]])
        subs_train.append(subs_array[np.where(subs_array == a)[0]])

    return X_train, X_validation, Y_train, Y_validation
