import os
import cv2
import numpy as np
import pickle

def save_data(file_to_save, name):
    with open(name, 'wb') as f:
        pickle.dump(file_to_save, f)


#######################################################################################################################
# This script saves the number of frames per video. This is used in later scripts.
#######################################################################################################################
# This section requires altering

# The path where the videos are stored
videos_path = './Data/videos_i8/'

# The location to save the frame counts to
save_path = './Behavioural_Extraction_Files/'

#######################################################################################################################
# Intialise variables
framespersecond = []
number_frames = []

# Get the list of video names
names = os.listdir(videos_path)

# Run for each video
for a in names:
    # Where this code was found
    # https://stackoverflow.com/questions/33311153/python-extracting-and-saving-video-frames

    # Locate the video
    video_path = videos_path + str(a)
    videocap = cv2.VideoCapture(video_path)

    # Fps is also used elsewhere but already listed in the script
    framespersecond.append(int(videocap.get(cv2.CAP_PROP_FPS)))

    # Keep number of frames
    number_frames.append(int(videocap.get(cv2.CAP_PROP_FRAME_COUNT)))

#print(framespersecond)

# There is a non-relevant number at the end
number_frames = number_frames[:-1]

# Save the data
os.chdir(save_path)
save_data(number_frames, "Number_frames.pkl")
