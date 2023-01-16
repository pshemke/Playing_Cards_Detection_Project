'''
Program designed to indetify cards from camera
'''
import cv2
import time
import os


import numpy as np


# static camera seting
camera_width = 1280
camera_height = 720
frames_per_sec = 10

# calculate frame rate afer first show
frames_per_sec_calc = 1
freq = cv2.getTickFrequency()

