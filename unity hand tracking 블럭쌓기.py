#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from cvzone.HandTrackingModule import HandDetector
import socket


#파라미터
width, height = 1280,720


#웹캠 설정
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

#Hand Detector
detector = HandDetector(maxHands = 2, detectionCon = 0.8)

#Communication
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
serverAddressPort = ("127.0.0.1", 5052)


while True:
    #웹캠에서 프레임 가져오기
    success,img = cap.read()
    # Hands
    hands, img = detector.findHands(img)
    
    
    data = []
    #랜드마크 value - (x,y,z) * 21
    if hands:
        # Get the first hand detected
        hand = hands[0]
        # Get the landmark list
        lmlist = hand['lmList']
        for lm in lmlist:
            data.extend([lm[0], height - lm[1], lm[2]])
        sock.sendto(str.encode(str(data)), serverAddressPort)    
    
    cv2.imshow("Image",img)
    cv2.waitKey(1)
    


# In[ ]:




