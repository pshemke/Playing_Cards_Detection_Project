import cv2
import time


import numpy as np


import Card_detector_classes

'''
static variables
'''
#adaptive threshold
BKG_TRESHOLD = 60
CARD_THRESHOLD = 30

#size of cards
CARD_AREA_MIN = 120000
CARD_AREA_MAX = 25000

#width and height of corner, where importatnt data are
CORNER_WIDTH = 32
CORNER_HEIGHT = 84

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
    threshold_level = bkg_level + BKG_TRESHOLD
    
    retval, threshold = cv2.threshold(blur, threshold_level, 255, cv2.THRESH_BINARY)
    
    return threshold

def find_card(pre_processed_frame):
    
    _, contours, hierarchy = cv2.findCoutours(pre_processed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    sort = sorted(range(len(contours)), key= lambda i : cv2.contourArea(contours[i]), reverse = True)     #sort finded contours
    
    #if there is no countours finded, do nothing
    if len(contours) == 0:
        return [],[]
    
    #otherwise, process finded contours
    contours_sort = []
    hierarchy_sort = []
    
    contours_is_card = np.zeros(len(contours))
    
    #now insert sorted list into free table
    for i in sort:
        contours_sort.append(contours[i])
        hierarchy_sort.append(hierarchy[0, i])
        
    '''now determine which contour is card by criteria
        1) smaller area than max card size
        2) bigger area than min card size
        3) have no parents
        4) have 4 corners
    '''
    print("a")
    for i in range(len(contours_sort)):
        size = cv2.contourArea(contours_sort[i])
        retval = cv2.arcLength(contours_sort[i], True)
        approxima = cv2.approxPolyDP(contours_sort[i], 0.01*retval, True)
        
        if((size > CARD_AREA_MIN) and (size < CARD_AREA_MAX)):
            if hierarchy_sort[i][3] == -1:
                if len(approxima) == 4:
                    contours_is_card[i] = 1
    
    return contours_sort, hierarchy_sort
    