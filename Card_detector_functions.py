import cv2
import time
import sys


import numpy as np


import Card_detector_classes

'''
static variables
'''
#adaptive threshold
BKG_TRESHOLD = 130
CARD_THRESHOLD = 30

#size of cards
CARD_AREA_MAX = 1200000
CARD_AREA_MIN = 10000

#width and height of corner, where importatnt data are
CORNER_WIDTH = 32
CORNER_HEIGHT = 84

# Dimensions of rank train images
RANK_WIDTH = 70
RANK_HEIGHT = 125

# Dimensions of suit train images
SUIT_WIDTH = 70
SUIT_HEIGHT = 100

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
    bkg_level = []
    bkg_level.append(gray[int(img_height/2)][int(img_weight/2)])
    bkg_level.append(gray[0][0])
    bkg_level.append(gray[-1][0])
    bkg_level.append(gray[0][-1])
    bkg_level.append(gray[-1][-1])
    
    threshold_level = sum(bkg_level)/len(bkg_level) + BKG_TRESHOLD
    
    retval, threshold = cv2.threshold(blur, threshold_level, 255, cv2.THRESH_BINARY)
    
    return threshold

def flattener(img, points, width, height):
    temp_rectangle = np.zeros((4, 2), dtype= 'float32')
    
    points_sum = np.sum(points, axis= 2)
    tl = points[np.argmin(points_sum)]
    br = points[np.argmax(points_sum)]
    
    points_diff = np.diff(points, axis= -1)
    tr = points[np.argmin(points_diff)]
    bl = points[np.argmax(points_diff)]
    
    if(width <= 0.8*height):        #if card oriented vertically
        temp_rectangle[0] = tl
        temp_rectangle[1] = tr
        temp_rectangle[2] = br
        temp_rectangle[3] = bl  
    elif(width >= 1.2*height):      #if card oriented horizontally
        temp_rectangle[0] = bl
        temp_rectangle[1] = tl
        temp_rectangle[2] = tr
        temp_rectangle[3] = br
    else:
        if points[1][0][0] >= points[3][0][0]:  #else if card is oriented diamon, we need to identyficate which point is which
            temp_rectangle[0] = points[1][0]    #top left
            temp_rectangle[1] = points[0][0]    #top right
            temp_rectangle[2] = points[3][0]    #bottom left
            temp_rectangle[3] = points[2][0]    #bottom right
        else:
            temp_rectangle[0] = points[0][0]    #top left
            temp_rectangle[1] = points[3][0]    #top right
            temp_rectangle[2] = points[2][0]    #bottom left
            temp_rectangle[3] = points[1][0]    #bottom right                           
            
    max_width = 200
    max_height = 300
    
    #create destination array, calculate perspective to transform matrix
    
    destination = np.array([[0, 0], 
                            [max_width - 1, 0], 
                            [max_width - 1 ,max_height - 1], 
                            [0, max_height - 1]], np.float32)
    
    M = cv2.getPerspectiveTransform(temp_rectangle, destination)

    #create wrapped card image
    wrap = cv2.warpPerspective(img, M, (max_width, max_height))
    wrap = cv2.cvtColor(wrap, cv2.COLOR_BGR2GRAY)
    
    return wrap

def find_card(pre_processed_frame):
    
    contours, hierarchy = cv2.findContours(pre_processed_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    sort = sorted(range(len(contours)), key=lambda i : cv2.contourArea(contours[i]),reverse=True)     #sort finded contours
    
    #if there is no countours finded, do nothing
    if len(contours) == 0:
        return [], []
    
    #otherwise, process finded contours
    contours_sort = []
    hierarchy_sort = []
    
    contours_is_card = np.zeros(len(contours),dtype=int)
    
    #now insert sorted list into free table
    for i in sort:
        contours_sort.append(contours[i])
        hierarchy_sort.append(hierarchy[0][i])
        
    '''now determine which contour is card by criteria
        1) smaller area than max card size
        2) bigger area than min card size
        3) have no parents
        4) have 4 corners
    '''
    for i in range(len(contours_sort)):
        size = cv2.contourArea(contours_sort[i])
        retval = cv2.arcLength(contours_sort[i], True)
        approxima = cv2.approxPolyDP(contours_sort[i], 0.01*retval, True)

        
        if((size > CARD_AREA_MIN) and (size < CARD_AREA_MAX) and (hierarchy_sort[i][3] == -1) and (len(approxima) == 4)):
            contours_is_card[i] = 1
        '''
        if((hierarchy_sort[i][3] == -1) and (len(approxima) == 4)):
            contours_is_card[i] = 1
        '''
    
    return contours_sort, contours_is_card

def process_card(contour, cur_image):
    
    #inicjalize new instance of Query_card
    querry_card = Card_detector_classes.Query_card()
    
    querry_card.contour = contour
    
    #find perimeter if card, use it to approximate corner points
    retval = cv2.arcLength(contour, True)
    approxima = cv2.approxPolyDP(contour, 0.01*retval, True)
    points = np.float32(approxima)
    querry_card.corner_pts = points
    
    #detect width and height
    x, y, width, height = cv2.boundingRect(contour)
    querry_card.width = width
    querry_card.height = height
    
    
    #find center of card by takink x, y and 4 corners
    average = np.sum(points, axis=0)/len(points)
    center_x = int(average[0][0])
    center_y = int(average[0][1])
    querry_card.center = [center_x, center_y]
    
    
    #wrap card into 200x300 flat image using perspective transform
    querry_card.wrap = flattener(cur_image, points, width, height)
    
    #get corner of card image and do a 4x zoom
    Qcorner = querry_card.wrap[0:CORNER_HEIGHT, 0:CORNER_WIDTH]
    Qcorner_zoom = cv2.resize(Qcorner, (0,0), fx=4, fy=4)

    #sample white pixel density to find appropriate threshold level
    white_level = Qcorner_zoom[15,int((CORNER_WIDTH*4)/2)]
    thresh_level = white_level - CARD_THRESHOLD
    if (thresh_level <= 0):
        thresh_level = 1
    retval, query_thresh = cv2.threshold(Qcorner_zoom, thresh_level, 255, cv2. THRESH_BINARY_INV)
    
    #split into top (rank) and bottom (suit)
    Qrank = query_thresh[20:185, 0:128]
    Qsuit = query_thresh[186:336, 0:128]

    #find rank contour and bounding rectangle, isolate and find largest contour
    Qrank_cnts, hier = cv2.findContours(Qrank, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qrank_cnts = sorted(Qrank_cnts, key=cv2.contourArea,reverse=True)
    cv2.drawContours(cur_image, Qrank_cnts, -1, (0, 255, 0), 2) #TODO tobe removed in develop
    
    #find bounding rectangle for largest contour, use it to resize query rank
    #image to match dimensions of the train rank image
    if len(Qrank_cnts) != 0:
        x1,y1,w1,h1 = cv2.boundingRect(Qrank_cnts[0])
        Qrank_roi = Qrank[y1:y1+h1, x1:x1+w1]
        Qrank_sized = cv2.resize(Qrank_roi, (RANK_WIDTH,RANK_HEIGHT), 0, 0)
        querry_card.rank_img = Qrank_sized
    #find suit contour and bounding rectangle, isolate and find largest contour
    Qsuit_cnts, hier = cv2.findContours(Qsuit, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    Qsuit_cnts = sorted(Qsuit_cnts, key=cv2.contourArea,reverse=True)
    
    #find bounding rectangle for largest contour, use it to resize query suit
    #image to match dimensions of the train suit image
    if len(Qsuit_cnts) != 0:
        x2,y2,w2,h2 = cv2.boundingRect(Qsuit_cnts[0])
        Qsuit_roi = Qsuit[y2:y2+h2, x2:x2+w2]
        Qsuit_sized = cv2.resize(Qsuit_roi, (SUIT_WIDTH, SUIT_HEIGHT), 0, 0)
        querry_card.suit_img = Qsuit_sized


    return querry_card