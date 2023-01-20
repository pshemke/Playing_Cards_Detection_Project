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
frames_per_sec_for_camera = 10

# calculate frame rate afer first show
frames_per_sec_calc = 1
freq = cv2.getTickFrequency()

#define font
font = cv2.FONT_HERSHEY_SIMPLEX

#start the camera 0 for build in camera, 1 for USB camera
video_stream = Video.Camera_stream((camera_width, camera_height), frames_per_sec_for_camera, 0).start()
#video_stream = Video.Camera_stream((camera_width, camera_height), frames_per_sec_for_camera, 1).start()
time.sleep(1)

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
    
    pre_processed_img = Card_detector_functions.preprocess_frame(cur_image)
    
    contour_sort, contour_is_card= Card_detector_functions.find_card(pre_processed_img)
    
    #do if there are card countours
    if(len(contour_sort) != 0):
        cards = []
        c = 0
        for i in range(len(contour_sort)):
            if(contour_is_card[i] == 1):
                cards.append(Card_detector_functions.process_card(contour_sort[i], cur_image))
                
                cards[c].best_rank_match,cards[c].best_suit_match,cards[c].rank_diff,cards[c].suit_diff = Card_detector_functions.match_card(cards[c],tranin_ranks,train_suits)

                image = Card_detector_functions.draw_results(cur_image, cards[c])
                c += 1
    
        #draw 
        if(len(cards) != 0):
            temp_contour = []
            for i in range(len(cards)):
                temp_contour.append(cards[i].contour)
            
            cv2.drawContours(cur_image, temp_contour, -1, (255, 0, 0), 2)
    
    #display FPS in the corner
    cv2.putText(cur_image, "FPS: " + str(int(frames_per_sec_calc)), (10, 26), font, 1, (255, 0, 255), 1, cv2.LINE_AA)
    
    #display frame od the screen
    cv2.imshow("Card detector", cur_image)
    
    timer2 = cv2.getTickCount()
    
    passed_time = (timer2-timer1) / freq
    
    frames_per_sec_calc = 1 / passed_time
    
    #check if 'e' key is pressed, if so exit mian loop
    key = cv2.waitKey(1) & 0xFF
    if key == ord('e'):
        quit_cam = 1
        
#close everywith after main loop exit
cv2.destroyAllWindows()
video_stream.stop()