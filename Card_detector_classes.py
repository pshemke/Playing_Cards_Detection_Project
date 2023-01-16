import numpy as np
import cv2
import time

'''
    Class for storing querry cards
'''
class Detected_card:
    def __init__(self):
        self.widh, self.height = 0, 0 #detected width and height


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