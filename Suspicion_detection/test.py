import cv2 ,time
import numpy as np
import face_recognition
import os
from datetime import datetime
import winsound
from pygame import mixer
import requests
import geocoder
g = geocoder.ip('117.216.103.150')
face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
gun_cascade=cv2.CascadeClassifier("cascade.xml")
palm_cascade = cv2.CascadeClassifier("palm.xml")
path = 'member'
images = []
classNames = []
myList = os.listdir(path)
print(myList)
for cl in myList:
   curImg = cv2.imread(f'{path}/{cl}')
   images.append(curImg)
   classNames.append(os.path.splitext(cl)[0])
print(classNames)

def listToString(s): 
    float_string=""
    for num in s:
        float_string=float_string+str(num)+" "
    return float_string

def send_msg(text):
    token = "5500588613:AAH5XwGTh3YCJgJhZDnxTjYDKR-ct9AhFro"
    chat_id = "-775064526"
    url_req = "https://api.telegram.org/bot" + token + "/sendMessage" +"?chat_id=" + chat_id + "&text=" + text
    results = requests.get(url_req)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
def markAttendance(name):
    with open('entry.csv','r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'n{name},{dtString}') 
# #### FOR CAPTURING SCREEN RATHER THAN WEBCAM
# # def captureScreen(bbox=(300,300,690+300,530+300)):
# #     capScr = np.array(ImageGrab.grab(bbox))
# #     capScr = cv2.cvtColor(capScr, cv2.COLOR_RGB2BGR)
# #     return capScr
encodeListKnown = findEncodings(images)
print('Encoding Complete')
video=cv2.VideoCapture(0)
mixer.init()
sound = mixer.Sound('Alarm.wav')
KNOWN_DISTANCE = 30.2
KNOWN_WIDTH = 14.3 
GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
fonts = cv2.FONT_HERSHEY_COMPLEX
def focal_length(measured_distance, real_width, width_in_rf_image):
    focal_length_value = (width_in_rf_image * measured_distance) / real_width
    return focal_length_value
def distance_finder(focal_length, real_face_width, face_width_in_frame):
    distance = (real_face_width * focal_length) / face_width_in_frame
    return distance
def face_data(image):
    face_width = 0
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, h, w) in faces:
        face_width = w
    return face_width
ref_image = cv2.imread("Ref_image.png")
ref_image_face_width = face_data(ref_image)
focal_length_found = focal_length(KNOWN_DISTANCE, KNOWN_WIDTH, ref_image_face_width)
while True:
    check,frame=video.read()  
    frame = cv2.flip(frame, flipCode = 1)
    imgS = cv2.resize(frame,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)

    for encodeFace,faceLoc in zip(encodesCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis)
        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            winsound.Beep(500,200)
            print(name)
            y1,x2,y2,x1 = faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
            markAttendance(name)
        if faceDis[matchIndex]< 0.50:
            name = classNames[matchIndex].upper()
            send_msg(name+" was found at shop 27, near kamraj road, coimbatore\n"+listToString(g.latlng))
            markAttendance(name)
        else: name = 'Unknown'
        #print(name)
        y1,x2,y2,x1 = faceLoc
        y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
        cv2.putText(frame,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        face_width_in_frame = face_data(frame)
        if face_width_in_frame != 0:
            Distance = distance_finder(focal_length_found, KNOWN_WIDTH, face_width_in_frame)
            cv2.putText(frame, f"Distance = {round(Distance,2)} CM", (50, 50), fonts, 1, (WHITE), 2)
            if Distance <= 100 and Distance > 50:
                winsound.Beep(500,200)
                cv2.putText(frame,"You are coming too close", (50, 450), fonts, 1, (WHITE), 2)
            elif (Distance <= 50):
                cv2.putText(frame,"calling cops", (50, 450), fonts, 1, (WHITE), 2)
                send_msg(" Someone breaking camera at shop 27, near kamraj road, coimbatore\n lat/long"+listToString(g.latlng))
        palm = palm_cascade.detectMultiScale(frame)
        cnt = 0     
        for (x, y, w, h) in palm:
            cv2.rectangle(frame, (x,y) , (x+w, y+h) , (12, 56, 56) , 2)
            cv2.putText(frame, "palm Detected", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (12, 56, 56), 3)
            cnt += 1
            if cnt >= 2:
                sound.play()
                cv2.putText(frame,"Suspicion Detected", (50, 400), fonts, 1, (WHITE), 2)    
                send_msg("burglary at shop 27, near kamraj road, coimbatore\n lat/long"+listToString(g.latlng))
        gun = gun_cascade.detectMultiScale(frame)
        for (x, y, w, h) in gun:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 68, 32), 1) 
            cv2.putText(frame, "GUN DETECTED!!!", (x + 5, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 68, 32), 2)
            winsound.Beep(500,200)
            send_msg(" Arms detected at shop 27, near kamraj road, coimbatore\n lat/long"+listToString(g.latlng))
    cv2.imshow('CAM',frame)
    key=cv2.waitKey(1)
    if key==ord('q'):
         break

video.release()
cv2.destroyAllWindows()        
