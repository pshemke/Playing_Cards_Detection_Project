'''
Program designed to indetify cards from camera
'''
import cv2
import time
import os
import math


import numpy as np


import Card_detector_classes
import Card_detector_functions
import Video

# static camera seting
camera_width = 1280
camera_height = 720
frames_per_sec_for_camera = 60

# calculate frame rate afer first show
frames_per_sec_calc = 1
freq = cv2.getTickFrequency()

#define font
font = cv2.FONT_HERSHEY_SIMPLEX #TODO (maybe if we want to change font)

#start the camera
video_stream = Video.Camera_stream((camera_width, camera_height), frames_per_sec_for_camera, 0).start()
time.sleep(10)

# load train rank and suit images
#tranin_ranks = Card_detector_functions.load_ranks('Card_Imgs/')
#train_suits = Card_detector_functions.load_suits('Card_Imgs/')

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
    
    contour_sort, contour_is_card= Card_detector_functions.find_card(pre_processed_card)
    
    #do if there are card countours
    if(len(contour_sort) != 0):
        cards = []
        c = 0
        
        for i in range(len(contour_sort)):
            if(contour_is_card[i] == 1):
                cards.append() #TODO
                
                '''
                    preprocess card funct #TODO
                    match card funct #TODO
                    draw result funct #TODO
                '''
                
                c += 1
    
    #draw 
    if(len(cards) != 0):
        '''
            draw countours on cards (maybe dont need?) #TODO
        '''
        pass #TODO
    
    #display FPS in the corner
    cv2.putText(cur_image, "FPS: " + str(int(frames_per_sec_calc)), (10, 26), font, 0.7, (255, 0, 255), cv2.LINE_AA)
    
    #display frame od the screen
    cv2.imshow("Card detector", cur_image)
    
    timer2 = cv2.getTickCount()
    
    passed_time = math.abs(timer2-timer1) / freq
    
    frames_per_sec_calc = 1 / passed_time
    
    #check if 'e' key is pressed, if so exit mian loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'):
        quit_cam = 1
        
#close everywith after main loop exit
cv2.destroyAllWindows()
video_stream.stop()