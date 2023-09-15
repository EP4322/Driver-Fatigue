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

    if (a == 1 or  a == 2 or a == 7 or a == 8 or a == 10):
        case = 1
    elif (a == 3):
        case = 2
    elif (a == 4 or a == 5 or a == 11 or a == 14):
        case = 3
    elif (a == 6):
        case = 4
    elif (a == 9):
        case = 5
    elif (a == 13 or a == 12):
        case = 6
    G = np.array([1, 2, 7, 8, 10])
    I = np.array([4, 5, 11, 14])
    J = 6
    K = 9
    sub1 = a
    sub2 = a
    sub3 = a
    sub4 = a

    if (case == 1 or case == 3 or case == 4):
        temp = random.randint(0, 5)
        if (temp == 0):
            temp = random.randint(0, 4)
            subs = G
            if case == 1:
                np.delete(subs, np.where(subs == a))
            else:
                np.delete(subs, temp)
            case2 = 1
        elif (temp == 1):
            while a == sub1:
                temp = random.randint(0, 3)
                sub1 = I[temp]
            while a == sub2:
                temp = random.randint(0, 4)
                sub2 = G[temp]
                while a == sub3 or sub2 == sub3:
                    temp = random.randint(0, 4)
                    sub3 = G[temp]
            subs = np.hstack((sub1, sub2, sub3))
            case2 = 2
        elif (temp == 2):
            while a == sub1:
                temp = random.randint(0, 3)
                sub1 = I[temp]
            sub2 = J
            while a == sub3:
                temp = random.randint(0, 4)
                sub3 = G[temp]
            subs = np.hstack((sub1, sub2, sub3))
            case2 = 1
        elif (temp == 3):
            sub1 = K
            while a == sub2:
                temp = random.randint(0, 4)
                sub2 = G[temp]
                while a == sub3 or sub2 == sub3:
                    temp = random.randint(0, 4)
                    sub3 = G[temp]
                    while a == sub4 or sub4 == sub3 or sub4 == sub2:
                        temp = random.randint(0, 4)
                        sub4 = G[temp]
            subs = np.hstack((sub1, sub2, sub3, sub4))
            case2 = 2
        elif (temp == 4):
            sub1 = K
            sub2 = J
            while a == sub3:
                temp = random.randint(0, 4)
                sub2 = G[temp]
                while a == sub4 or sub3 == sub4:
                    temp = random.randint(0, 4)
                    sub4 = G[temp]
            subs = np.hstack((sub1, sub2, sub3, sub4))
            case2 = 1
        elif (temp == 5):
            while a == sub1:
                temp = random.randint(0, 3)
                sub1 = I[temp]
                while a == sub2 or sub2 == sub1:
                    temp = random.randint(0, 3)
                    sub2 = I[temp]
            sub34 = L
            subs = np.hstack((sub1, sub2, sub34))
            case2 = 3

    elif (case == 2 or case == 6):
        if case == 2:
            temp = random.randint(1, 6)
        else:
            temp = random.randint(0, 5)
        if (temp == 0):
            temp = random.randint(0, 4)
            subs = G
            if case == 1:
                np.delete(subs, np.where(subs == a))
            else:
                np.delete(subs, temp)
            case2 = 2
        elif (temp == 1):
            while a == sub1:
                temp = random.randint(0, 3)
                sub1 = I[temp]
            sub2 = J
            while a == sub3:
                temp = random.randint(0, 4)
                sub3 = G[temp]
            subs = np.hstack((sub1, sub2, sub3))
            case2 = 2
        elif (temp == 6):
            while a == sub1:
                temp = random.randint(0, 3)
                sub1 = I[temp]
                while a == sub2 or sub2 == sub1:
                    temp = random.randint(0, 3)
                    sub2 = I[temp]
            subs = np.hstack((sub1, sub2))
            case2 = 4
        elif (temp == 3):
            while a == sub2:
                temp = random.randint(0, 4)
                sub2 = G[temp]
                while a == sub3 or sub2 == sub3:
                    temp = random.randint(0, 4)
                    sub3 = G[temp]
            sub1 = K
            sub4 = J
            subs = np.hstack((sub1, sub2, sub3, sub4))
            case2 = 2
        elif (temp == 4):
            while a == sub1:
                temp = random.randint(0, 3)
                sub1 = I[temp]
            sub2 = K
            sub3 = J
            subs = np.hstack((sub1, sub2, sub3))
            case2 = 2
        elif (temp == 5):
            while a == sub2:
                temp = random.randint(0, 4)
                sub2 = G[temp]
                while a == sub3 or sub2 == sub3:
                    temp = random.randint(0, 4)
                    sub3 = G[temp]
                    while a == sub4 or sub4 == sub3 or sub4 == sub2:
                        temp = random.randint(0, 4)
                        sub4 = G[temp]
            sub1 = J
            subs = np.hstack((sub1, sub2, sub3, sub4))
            case2 = 1
        elif (temp == 2):
            while a == sub2:
                temp = random.randint(0, 4)
                sub2 = G[temp]
                while a == sub3 or sub2 == sub3:
                    temp = random.randint(0, 4)
                    sub3 = G[temp]
                    while a == sub4 or sub4 == sub3 or sub4 == sub2:
                        temp = random.randint(0, 4)
                        sub4 = G[temp]
            sub1 = K
            subs = np.hstack((sub1, sub2, sub3))
            case2 = 3

    elif (case == 5):
        temp = random.randint(0, 1)
        if (temp == 0):
            while a == sub1:
                temp = random.randint(0, 3)
                sub1 = I[temp]
                while a == sub2 or sub2 == sub1:
                    temp = random.randint(0, 3)
                    sub2 = I[temp]
            while a == sub3:
                temp = random.randint(0, 4)
                sub3 = G[temp]
            subs = np.hstack((sub1, sub2, sub3))
            case2 = 2
        elif (temp == 1):
            while a == sub1:
                temp = random.randint(0, 3)
                sub1 = I[temp]
            while a == sub2:
                temp = random.randint(0, 4)
                sub2 = G[temp]
                while a == sub3 or sub2 == sub3:
                    temp = random.randint(0, 4)
                    sub3 = G[temp]
                    while a == sub4 or sub4 == sub3 or sub4 == sub2:
                        temp = random.randint(0, 4)
                        sub4 = G[temp]
            subs = np.hstack((sub1, sub2, sub3, sub4))
            case2 = 1

    return case2, subs


# This function generates a random set of non drowsy training subjects to keep 30% testing
def Awake_training(a, case2, seeding=True):
    # For repetition
    if seeding == True:
        seed(1)
    H = 3
    L = np.array([12, 13])
    if (case2 == 1):
        subs = np.empty(1)
    elif (case2 == 2):
        subs = a
        while a == subs:
            temp = random.randint(0, 1)
            subs = L[temp]
    elif (case2 == 3):
        subs = L
    elif (case2 == 4):
        subs = H

    # The non drowsy subjects that contribute to 30% testing is generated
    return subs


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
            temp2 = np.array([2.2])
            idx2 = np.hstack((idx2, temp2))
        elif a == 3:
            temp = np.array([3.1, 3.2, 3.3])
            idx = np.hstack((idx, temp))
        elif a == 4:
            temp = np.array([4.1])
            temp2 = np.array([4.2, 4.3])
            idx = np.hstack((idx, temp))
            idx2 = np.hstack((idx2, temp2))
        elif a == 5:
            temp = np.array([5.1])
            temp2 = np.array([5.2, 5.3])
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
            temp2 = np.array([7.3])
            idx = np.hstack((idx, temp))
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
            temp2 = np.array([11.2, 11.3])
            idx2 = np.hstack((idx2, temp2))
        elif a == 12:
            temp = np.array([12.1])
            idx = np.hstack((idx, temp))
        elif a == 13:
            temp = np.array([13.2])
            idx = np.hstack((idx, temp))
        elif a == 14:
            temp = np.array([14.1])
            temp2 = np.array([14.2, 14.3])
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
                temp2 = np.array([2.2])
                idx2 = np.hstack((idx2, temp2))
            elif a == 3:
                temp = np.array([3.1, 3.2, 3.3])
                idx = np.hstack((idx, temp))
            elif a == 4:
                temp = np.array([4.1])
                temp2 = np.array([4.2, 4.3])
                idx = np.hstack((idx, temp))
                idx2 = np.hstack((idx2, temp2))
            elif a == 5:
                temp = np.array([5.1])
                temp2 = np.array([5.2, 5.3])
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
                temp2 = np.array([7.3])
                idx = np.hstack((idx, temp))
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
                temp2 = np.array([11.2, 11.3])
                idx2 = np.hstack((idx2, temp2))
            elif a == 12:
                temp = np.array([12.1])
                idx = np.hstack((idx, temp))
            elif a == 13:
                temp = np.array([13.2])
                idx = np.hstack((idx, temp))
            elif a == 14:
                temp = np.array([14.1])
                temp2 = np.array([14.2, 14.3])
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
        sub = np.array([2.1, 2.2])
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
        sub = np.array([8.1, 8.3])
    elif Valid == 9:
        sub = np.array([9.3])
    elif Valid == 10:
        sub = np.array([10.1, 10.3])
    elif Valid == 11:
        sub = np.array([11.1, 11.2, 11.3])
    elif Valid == 12:
        sub = np.array([12.1])
    elif Valid == 13:
        sub = np.array([13.2])
    elif Valid == 14:
        sub = np.array([14.1, 14.2, 14.3])
    return sub


# This function splits the data into training and testing data (referred to as "validation")
def get_train_test(awake_subs, drowsy_subs, data, valid_trials_awake, valid_trials_drowsy, time, base):
    ind = data[:, -1]
    # Get trials of subjects
    awake_subs_idx, temp = Subjects(awake_subs)
    D_awake_subs, drowsy_subs_idx = Subjects(drowsy_subs)

    # Get the validation data subjects
    X_validation_idx = np.hstack((awake_subs_idx, D_awake_subs, drowsy_subs_idx))
    Y_validation = np.hstack((np.zeros(len(awake_subs_idx)*(base-time)), np.zeros(len(D_awake_subs)*(base-time)),
                             np.ones(len(drowsy_subs_idx)*(base-time))))

    # Set our awake and asleep trials
    if EEG:
        awake = np.array([1.1, 10.1, 11.1, 12.1, 13.2, 14.1, 2.1, 3.1, 3.2, 3.3, 4.1, 5.1, 6.1, 6.2, 7.2, 8.1])
    else:
        awake = np.array([1.1, 10.1, 11.1, 12.1, 13.2, 14.1, 2.1, 3.1, 3.2, 3.3, 4.1, 5.1, 6.2, 7.2, 8.1])
    drowsy = np.array([1.3, 10.3, 11.2, 11.3, 14.2, 14.3, 2.2, 4.2, 4.3, 5.2, 5.3, 6.3, 8.3, 7.3, 9.3])

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



