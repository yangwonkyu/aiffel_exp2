#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' conda install protobuf==3.19.0')


# In[ ]:


import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

offset = 20
imgSize = 300

folder = "Data1/C"
counter = 0

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255 #뒤에 255를 추가해줘야 하얀 화면이 나온다 기본적으로는 검은색
        
        
        imgCrop = img[y - offset :y + h + offset, x - offset:x + w + offset]
        
        imgCropShape = imgCrop.shape
                
               
        aspectRatio = h/w
        
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k*w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal)/2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k*h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal)/2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
        
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)
        
    #s 키를 눌러서 이미지를 저장 위에 folder에서 원하는 폴더를 입력하면 그 폴더에 이미지가 저장된다.
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1 
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg',imgWhite)
        print(counter)


# In[ ]:




