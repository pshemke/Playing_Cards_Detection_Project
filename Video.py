import cv2


from threading import Thread



class Camera_stream:
    def __init__(self, res=(640, 480), framerate = 30, src=0):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(3, res[0])
        self.stream.set(4, res[1])
        
        self.frame = []
        
        (self.grabbed, self.frame) = self.stream.read()    #read first frame from stream
        
        self.stopped = False #create var to possible stop the camera
        
    def start(self):
        Thread(target= self.update, args= ()).start() #start thread for camera
        return self
    
    def update(self):
        while True:
            if self.stopped:    #if camera stoped, stop loop
                self.stream.release() #close camera
                return
            else:
                (self.grabbed, self.frame) = self.stream.read() #grab next frame
    
    def read(self):
        return self.frame   #return last frame
     
    def stop(self):
        self.stopped = True #set stop var to true so main lopp of Treah will stop