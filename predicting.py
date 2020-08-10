import numpy as np
import cv2
import pickle
import time
import serial
 
frameWidth= 640        
frameHeight = 480
brightness = 180
threshold = 0.75         
font = cv2.FONT_HERSHEY_SIMPLEX
 
cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10, brightness)

pickle_in=open("/content/drive/My Drive/dynamic_speed_limiter/model_trained.p","rb")  
model=pickle.load(pickle_in)
 
def sendData(data):
    ser = serial.Serial(           
        port='/dev/ttyAMA0',
        baudrate = 9600,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        bytesize=serial.EIGHTBITS,
        timeout=1
    )
    ser.write(data)

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img

def equalize(img):
    img =cv2.equalizeHist(img)
    return img

def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

def getCalssName(classNo):
    if   classNo == 0: return '20 km/h'
    elif classNo == 1: return '30 km/h'
    elif classNo == 2: return '50 km/h'
    elif classNo == 3: return '60 km/h'
    elif classNo == 4: return '70 km/h'

while True:
    success, imgOrignal = cap.read()
    curImg = cv2.resize(imgOrignal, (32, 32))
    curImg = preprocessing(curImg)
    predictions = model.predict(curImg)
    classIndex = model.predict_classes(curImg)
    probabilityValue =np.amax(predictions)
    if probabilityValue > threshold:
        sendData(getCalssName(classIndex))
    if cv2.waitKey(1) and 0xFF == ord('q'):
        break