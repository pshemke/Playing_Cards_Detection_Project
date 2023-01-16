import cv2
import time


import numpy as np


import Card_detector_classes

'''
static variables
'''
#adaptive threshold
bkg_threshold = 60
card_threshold = 30

#width and height of corner, where importatnt data are
corner_width = 32
corner_height = 84

def load_ranks(path):
    train_ranks = []
    i = 0
    
    card_to_get = ["Ace", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Jack", "Queen", "King"]
    
    for name in card_to_get:
        #append array by class object
        train_ranks.append(Card_detector_classes.Train_ranks())
        train_ranks[i].name = name
        
        filename = name + '.jpg' #define namefile to be downloaded
        
        train_ranks[i].image = cv2.imread(path + filename, cv2.IMREAD_GRAYSCALE)
        i += 1
    return train_ranks        
    
def load_suits(path):
    train_suits = []
    i = 0
    
    card_to_get = ["Clubs", "Hearts", "Spades", "Diamonds"]
    
    for name in card_to_get:
        #append array by class object
        train_suits.append(Card_detector_classes.Train_suits())
        train_suits[i].name = name
        
        filename = name + '.jpg' #define namefile to be downloaded
        
        train_suits[i].image = cv2.imread(path + filename, cv2.IMREAD_GRAYSCALE)
        i += 1
    return train_suits  

def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    
    #adaptive treshold
    img_weight, img_height = np.shape(frame)[:2]
    bkg_level = gray[int(img_height/100)][int(img_weight/2)]
    threshold_level = bkg_level + bkg_threshold
    
    retval, threshold = cv2.threshold(blur, threshold_level, 255, cv2.THRESH_BINARY)
    
    return threshold

def find_card(pre_processed_frame):
    
    dummy, contours, hier = cv2.findCoutours(pre_processed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    sort = sorted(range(len(contours)), key= lambda i : cv2.contourArea(contours[i]), reverse = True)     #sort finded contours
    
    #if there is no countours finded, do nothing
    
    if len(contours) == 0:
        return [],[]
    
    