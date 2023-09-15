# import libraries
import os
#os.add_dll_directory("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.4/bin")
import cv2
import numpy as np


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


#######################################################################################################################
# This code to extract images
# Location to create subject folders and save images
images_path = './Data/images/'

# Videos location
videos_path = './Data/videos_i8/'
#######################################################################################################################
# Create relevant folders
names = Create_folders(videos_path, images_path)

for a in names:
    video_path = videos_path + str(a)
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()

    count = 0

    trial = a.split('.')[0]
    image_path = images_path + str(trial)

    os.chdir(image_path)
    while success:
        cv2.imwrite("frame%d.jpg" % count, image)  # save frame as JPEG file
        success, image = vidcap.read()
        count += 1


