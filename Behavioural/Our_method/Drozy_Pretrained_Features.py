import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")

import cv2
import numpy as np
from tensorflow.keras.applications import resnet50, vgg16, vgg19, resnet
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications.vgg16 import preprocess_input
import pickle
import natsort


def get_deep_features(input_img, model):
    image = np.expand_dims(input_img, axis=0)
    image = preprocess_input(image)
    deep_features = model.predict(image)
    d_features = deep_features.reshape(-1)
    d_feat = d_features.tolist()
    return d_feat


def save_data(file_to_save, name):
    with open(name, 'wb') as f:
        pickle.dump(file_to_save, f)


#######################################################################################################################
# This script extracts and saves the pre-trained model features (ResNet/VGG)
#######################################################################################################################
# These paths need amending

# Location of cropped images
cropping_path = './Data/Cropping/'

# The path where the videos are stored
videos_path = './Data/videos_i8/'

# The path to save the feature files output
save_path = './Behavioural_Extraction_Files/'

# Pre-trained model:
pre_train = 'ResNet101'

#######################################################################################################################
if pre_train == 'ResNet101':
    baseModel = resnet.ResNet101(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
elif pre_train == 'VGG16':
    baseModel = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
elif pre_train == 'ResNet50':
    baseModel = resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))
elif pre_train == 'VGG19':
    baseModel = vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=Input(shape=(224, 224, 3)))

#######################################################################################################################
# KSS Values
Drozy_KSS = np.array([3, 6, 7, 3, 7, 4, 7, 7, 2, 6, 3, 5, 7, 8, 3, 7, 6, 2, 3, 4, 4, 8, 9, 3, 7, 8, 2, 3, 7, 4, 9, 2,
                      6, 8, 6, 8])

# FPS (from the Number_Frames)
fps = [30, 15, 15, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 15, 15, 30, 15, 15, 30, 15, 15, 30, 15, 15, 30, 15,
       15, 15, 15, 30, 15, 15, 30, 30]


# First collect the trial names
names = os.listdir(videos_path)
names = names[:-1]
trial = []
for a in names:
    trial.append(a.split('.')[0])

#print(baseModel.summary())
for layer in baseModel.layers:
    layer.trainable = False
#print(baseModel.summary())

gap = GlobalAveragePooling2D()(baseModel.output)
model = Model(inputs=baseModel.inputs, outputs=gap)

# Run the images through the model
subs_data = []  # Store subject number and trial
KSS = []
data = []
frames = []

sub = list(range(1, 15))

# Run through all trials
for abc in range(0, len(Drozy_KSS)):
    p1 = os.listdir(cropping_path + trial[abc])
    p1 = natsort.natsorted(p1)  # Sort the images
    print(trial[abc])

    frame = []
    tensor = []
    for a in p1:
        img = cv2.imread(cropping_path + trial[abc] + '/' + a)

        frame1 = a.split('crop')[1]
        frame.append(frame1.split('.')[0])
        img = get_deep_features(img, model)
        tensor.append(img)

    tensor = np.array(tensor)
    KSS.append(Drozy_KSS[abc])
    data.append(tensor)
    subs_data.append(trial[abc])
    frames.append(frame)

# Save data
os.chdir(save_path)
save_data(subs_data, "subs_data_" + pre_train + ".pkl")
save_data(data, "data_" + pre_train + ".pkl")
save_data(KSS, "KSS_" + pre_train + ".pkl")
save_data(frames, "Frames_" + pre_train + ".pkl")


