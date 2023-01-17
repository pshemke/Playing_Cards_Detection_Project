'''
Program designed to indetify cards from camera
'''
import cv2
import time
import os


import numpy as np


import Card_detector_classes
import Card_detector_functions
import Video

# static camera seting
camera_width = 1280
camera_height = 720
frames_per_sec = 10

# calculate frame rate afer first show
frames_per_sec_calc = 1
freq = cv2.getTickFrequency()

#start the camera
video_stream = Video.Camera_stream((camera_width, camera_height), frames_per_sec, 0).start()
time.sleep(10)

# load train rank and suit images
tranin_ranks = Card_detector_functions.load_ranks('Card_Imgs/')
train_suits = Card_detector_functions.load_suits('Card_Imgs/')

'''
    MAIN LOOP
'''

quit_cam = 0

while quit_cam == 0:
    
    #take last frame from camera
    cur_image = video_stream.read() 
    
    #starting timer for calculating frame_rate
    timer1 = cv2.getTickCount()
    
    pre_processed_card = Card_detector_functions.preprocess_frame(cur_image)
    
    count_sort, count_is_card= Card_detector_functions.find_card(pre_processed_card)