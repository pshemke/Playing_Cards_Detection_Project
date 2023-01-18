import cv2
import time


import numpy as np


'''
    Class for storing querry cards
'''
class Query_card:
    def __init__(self):
        self.widh, self.height = 0, 0   #detected width and height
        self.countour = []              #describes countour of card
        self.corner_pts = []            #describes corner points of card
        self.center = []                #describes center of detected card
        self.wrap = []                  #flatted, grayed and blurred image
        self.rank_img = []              #temp, sized image of card's rank
        self.suit_img = []              #temp, sized image of card's suit
        self.best_rank_match = "NULL"   #best match card's rank to teaching rank
        self.best_suit_match = "NULL"   #best match card's suit to teaching suit
        self.rank_diff = 0              #difference betwen rank image and best teaching rank
        self.suit_diff = 0              #difference betwen suit image and best teaching suit


'''
    Classes for storing card to be compared to
'''


class Train_ranks:      #Class designed to store rank images of training cards
    def __init__(self):
        self.image = []
        self.name = "NULL"
        
class Train_suits:      #Class designed to store suits images of training cards
    def __init__(self):
        self.image = []
        self.name = "NULL"