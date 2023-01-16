'''
Program designed to indetify cards from camera
'''
import cv2
import time
import os


import numpy as np


import Card_detector_classes
import Card_detector_functions

# static camera seting
camera_width = 1280
camera_height = 720
frames_per_sec = 10

# calculate frame rate afer first show
frames_per_sec_calc = 1
freq = cv2.getTickFrequency()

# load train rank and suit images
tranin_ranks = Card_detector_functions.load_ranks('/Card_image/')
train_suits = Card_detector_functions.load_suits('/Card_image/')