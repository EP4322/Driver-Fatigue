import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")

import cv2
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle
import natsort
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt


def save_data(file_to_save, name):
    with open(name, 'wb') as f:
        pickle.dump(file_to_save, f)


def load_data(name1, directory):
    name = directory + name1
    with open(name, 'rb') as f:
        loaded = pickle.load(f)
    return loaded


#######################################################################################################################
# This script orders and combines the features for classification. The sequence length is altered here.
# This current script version only runs for ResNet101.
#######################################################################################################################
# These paths need amending
# The directory of the extracted features
load_directory = './Behavioural_Extraction_Files/' # May need adjusting

# The directory to save the ordered features into
save_directory = './Behavioural_Feature_Files/'

# Run for both True and False
Half = True

# The number of images in the sequence (10 determined as the best)
seq = 10

#######################################################################################################################
# Pickle will come in as a list of numpy arrays, sequence * 6, where 6 is:
index = ['idx', 'MAR', 'EAR', 'HeadTopZ', 'HeadBotZ', 'Head_leftZ', 'Head_rightZ']

# KSS Values
Drozy_KSS = np.array([3, 6, 7, 3, 7, 4, 7, 7, 2, 6, 3, 5, 7, 8, 3, 7, 6, 2, 3, 4, 4, 8, 9, 3, 7, 8, 2, 3, 7, 4, 9, 2,
                      6, 8, 6, 8])

# FPS (from the Number_Frames)
fps = [30, 15, 15, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 15, 15, 30, 15, 15, 30, 15, 15, 30, 15, 15, 30, 15,
       15, 15, 15, 30, 15, 15, 30, 30]


# Load pickles
Frame_numbers = load_data("Number_frames.pkl", load_directory)
data = load_data("MP_HC_data.pkl", load_directory)
subs = load_data("MP_HC_subs_data.pkl", load_directory)
KSS = load_data("MP_HC_KSS.pkl", load_directory)
Landmarks = load_data("MP_HC_Landmark_data.pkl", load_directory)

# Ignore the variable names (I did VGG first so this is the result)
VGG = load_data("data_ResNet101.pkl", load_directory)
VGG_Frames = load_data("Frames_ResNet101.pkl", load_directory)
VGG_KSS = load_data("KSS_ResNet101.pkl", load_directory)
VGG_subs = load_data("subs_data_ResNet101.pkl", load_directory)

# Format (some) handcrafted features
EAR = []
Nod = []
Side = []
MAR = []
for a in range(0, len(data)):
    EAR.append(data[a][2, :])
    MAR.append(data[a][1, :])
    Nod2 = []
    Side2 = []
    for b in range(0, len(data[a][0])):
        Side2.append(abs(float(data[a][5, b]) - float(data[a][6, b])))
        Nod2.append(float(data[a][4, b]) - float(data[a][3, b]))
    Side.append(np.asarray(Side2))
    Nod.append(np.asarray(Nod2))


# Make the sequence arrays:
# All data together
combined = []

# KSS labels
VGG_KSS_Combined = []

# Subs data
VGG_Subs_data = []

# Run for all subjects
for a in range(0, len(VGG)):
    # Print subject number
    print(a)

    # Every second or every image taken
    if fps[a] == 30:
        frame_skip = True
    elif fps[a] == 15:
        frame_skip = False

    # The total number of frames if all frames were extracted
    frame_num = int(Frame_numbers[a])

    # The frame numbers actually extracted
    arr = np.asarray(VGG_Frames[a], dtype=float)

    if Half:
        half_trial = np.where(arr > int(frame_num/2))[0][0]
    else:
        # Run for if all frames were extracted
        half_trial = len(VGG[a])

    for b in range(0, half_trial, seq):
        if b + seq > half_trial:
            print('new')
        else:
            temp1 = np.expand_dims(MAR[a][b:b+seq].astype('float64'), axis=1)
            temp2 = np.expand_dims(EAR[a][b:b + seq].astype('float64'), axis=1)
            temp3 = np.expand_dims(Side[a][b:b + seq].astype('float64'), axis=1)
            temp4 = np.expand_dims(Nod[a][b:b + seq].astype('float64'), axis=1)

            # Landmark extraction
            for sss in range(b, b+seq):
                temp5a = np.asarray(Landmarks[a][1][sss])
                temp5b = np.asarray(Landmarks[a][2][sss])
                temp5c = np.asarray(Landmarks[a][3][sss])
                temp5aa = np.hstack((temp5a, temp5b, temp5c))
                if sss == b:
                    temp5 = temp5aa
                else:
                    temp5 = np.vstack((temp5, temp5aa))
            combined.append(np.hstack((VGG[a][b:b+seq, :].astype('float64'), temp1, temp2, temp3, temp4, temp5)))
            VGG_KSS_Combined.append(VGG_KSS[a])
            VGG_Subs_data.append(VGG_subs[a])

os.chdir(save_directory)
if Half:
    save_data(combined, "ResNet101_Half_Combined_features_s" + str(seq) + ".pkl")
    save_data(VGG_Subs_data, "ResNet101_Half_Combined_subs_s" + str(seq) + ".pkl")
    save_data(VGG_KSS_Combined, "ResNet101_Half_Combined_KSS_s" + str(seq) + ".pkl")
else:
    save_data(combined, "ResNet101_Combined_features_s" + str(seq) + ".pkl")
    save_data(VGG_Subs_data, "ResNet101_Combined_subs_s" + str(seq) + ".pkl")
    save_data(VGG_KSS_Combined, "ResNet101_Combined_KSS_s" + str(seq) + ".pkl")


