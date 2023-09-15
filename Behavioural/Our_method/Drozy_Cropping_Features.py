# import libraries
import os
os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")

import cv2
import face_recognition
from mtcnn import MTCNN
import mediapipe as mp
import numpy as np
import pickle
import natsort
from scipy.spatial import distance as dist


def Create_folders(videos_path, destination):
    files = os.listdir(videos_path)
    for f in files:
        f_ = f.split('.')[0]
        if f_ not in os.listdir(destination):
            os.mkdir(destination + f_)
            print('Folder created' + str(f_) + '\n')

        else:
            print('Already exists' + str(f_) + '\n')
    return files


def get_features(x):
    IMAGE_FILES = x
    MAR = []
    EAR = []
    HeadTopZ = []
    HeadBotZ = []
    Head_leftZ, Head_rightZ = [], []
    landmarksx = []
    landmarksy = []
    landmarksz = []
    idx3 = []
    with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            min_detection_confidence=0.5) as face_mesh:
        for idx, file in enumerate(IMAGE_FILES):
            if file != 'desktop.ini':
                image = cv2.imread(file)
                # Convert the BGR image to RGB before processing.
                results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Print and draw face mesh landmarks on the image.
                if not results.multi_face_landmarks:
                    continue

                for face in results.multi_face_landmarks:
                    frame = x[idx].split('/frame')[1]
                    f = frame.split('.')[0]

                    # Cropping
                    h, w, c = image.shape
                    cx_min = w
                    cy_min = h
                    cx_max = cy_max = 0
                    for id, lm in enumerate(face.landmark):
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        if cx < cx_min:
                            cx_min = cx
                        if cy < cy_min:
                            cy_min = cy
                        if cx > cx_max:
                            cx_max = cx
                        if cy > cy_max:
                            cy_max = cy
                    box1 = cy_min
                    box0 = cx_min
                    box2 = cx_max - cx_min
                    box3 = cy_max - cy_min
                    img = image[box1: box1 + box3, box0: box0 + box2]

                    # Resize and saved cropped image
                    img = cv2.resize(img, (224, 224))
                    cv2.imwrite("crop%d.jpg" % int(f), img)

                    # Calculate MAR
                    MouthsumUpperY = 0.0
                    MouthsumLowerY = 0.0
                    for idx22 in range(0, len(Mouth_Upper)):
                        temp = face.landmark[Mouth_Upper[idx22]]
                        MouthsumUpperY += temp.y
                        temp2 = face.landmark[Mouth_Lower[idx22]]
                        MouthsumLowerY += temp2.y

                    MAR_TEMP = (MouthsumLowerY - MouthsumUpperY) / (face.landmark[Mouth_across[1]].x -
                                                                    face.landmark[Mouth_across[0]].x)

                    # Calculate EAR
                    LLeye = []
                    RReye = []
                    for va in Leye:
                        temp = [face.landmark[va].x, face.landmark[va].y]
                        LLeye.append(temp)
                    for va in Reye:
                        temp = [face.landmark[va].x, face.landmark[va].y]
                        RReye.append(temp)

                    LEAR = eye_aspect_ratio(LLeye)
                    REAR = eye_aspect_ratio(RReye)
                    # average the eye aspect ratio together for both eyes
                    EAR_TEMP = (LEAR + REAR) / 2.0

                    # Append landmarks for HC features
                    HeadTopZ.append(face.landmark[HeadTop[0]].z)
                    HeadBotZ.append(face.landmark[HeadBot[0]].z)
                    Head_leftZ.append(face.landmark[Side_Left[0]].z)
                    Head_rightZ.append(face.landmark[Side_Right[0]].z)

                    idx3.append(f)
                    MAR.append(MAR_TEMP)
                    EAR.append(EAR_TEMP)
                    #Frame_index.append(file)

                    # Landmarks
                    xx, yy, zz = [], [], []
                    for a in range(468):
                        xx.append(face.landmark[a].x)
                        yy.append(face.landmark[a].y)
                        zz.append(face.landmark[a].z)

                    landmarksx.append(xx)
                    landmarksy.append(yy)
                    landmarksz.append(zz)

        landmarks = [idx3, landmarksx, landmarksy, landmarksz]
        features = [idx3, MAR, EAR, HeadTopZ, HeadBotZ, Head_leftZ, Head_rightZ]
        lables = ['idx', 'MAR', 'EAR', 'HeadTopZ', 'HeadBotZ', 'Head_leftZ', 'Head_rightZ']
        return features, lables, landmarks


def save_data(file_to_save, name):
    with open(name, 'wb') as f:
        pickle.dump(file_to_save, f)


def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[2], eye[3])
    B = dist.euclidean(eye[4], eye[5])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[1])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear


#######################################################################################################################
# This script extracts and saves the images (creates folders for each subject), crops the images (saves in cropped
# path) and collects the handcrafted features and landmarks from the cropped images. Images are only saved if a face is
# detected. The frame numbers are also recorded.
#######################################################################################################################
# These paths need amending

# Location of images
images_path = './Data/images/'

# Location to create subject folders and save cropped images
cropping_path = './Data/Cropping/'

# The path where the videos are stored
videos_path = './Data/videos_i8/'

# The path to save the feature files output
save_path = './Behavioural_Extraction_Files/'

#######################################################################################################################
# This does not need amending

# MediaPipe landmark locations
# Eye
Leye = [33, 133, 160, 144, 158, 153]
Reye = [263, 362, 385, 380, 387, 373]

# Mouth idx
Mouth_Upper = [191, 80, 81, 82, 13, 312, 311, 310, 415]
Mouth_Lower = [95, 88, 178, 87, 14, 317, 402, 318, 324]
Mouth_across = [78, 308]

# Head
HeadTop = [10]
HeadBot = [152]

# Side of face
Side_Left = [137]
Side_Right = [454]

# This code to extract images and crop the face
names = Create_folders(videos_path, cropping_path)

# This is the MediaPipe intialisation
face_detection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
mp_drawing = mpDraw

# KSS Values
Drozy_KSS = np.array([3, 6, 7, 3, 7, 4, 7, 7, 2, 6, 3, 5, 7, 8, 3, 7, 6, 2, 3, 4, 4, 8, 9, 3, 7, 8, 2, 3, 7, 4, 9, 2,
                      6, 8, 6, 8])

# FPS (from the Number_Frames)
fps = [30, 15, 15, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 15, 15, 30, 15, 15, 30, 15, 15, 30, 15, 15, 30, 15,
       15, 15, 15, 30, 15, 15, 30, 30]

# List the videos
names = os.listdir(videos_path)
names = names[:-1]
trial = []
for a in names:
    trial.append(a.split('.')[0])

# Intialise arrays
subs_data = []  # Store subject number and trial
KSS = []
data, landmarks_data = [], []
frame = []

# Run through all trials
for abc in range(0, len(Drozy_KSS)):
    images = []
    cropped_path = cropping_path + str(trial[abc])
    os.chdir(cropped_path)

    p1 = os.listdir(images_path + trial[abc])
    if p1[0] == 'desktop.ini':
        p1 = p1[1:]
    p1 = natsort.natsorted(p1)  # Sort the images
    print(trial[abc])

    # Read every second frame when fps is 30 amnd all frames when fps is 15
    if fps[abc] == 30:
        fps_flag = 2
    else:
        fps_flag = 1

    # Initialise counter
    x = 0

    # Run whilst there is still images
    while x < len(p1):
        images.append(images_path + trial[abc] + '/' + p1[x])

        flag = 1
        if fps_flag == 2:
            x = x + 2
        else:
            x = x + 1

    # Collects feature information
    features, labels, landmarks = get_features(images)
    features = np.array(features)

    # Formats features
    KSS.append(Drozy_KSS[abc])
    data.append(features)
    subs_data.append(trial[abc])
    landmarks = np.array(landmarks)
    landmarks_data.append(landmarks)

# Save files
os.chdir(save_path)
save_data(data, "MP_HC_data.pkl")
save_data(KSS, "MP_HC_KSS.pkl")
save_data(subs_data, "MP_HC_subs_data.pkl")
save_data(landmarks_data, "MP_HC_landmark_data.pkl")
