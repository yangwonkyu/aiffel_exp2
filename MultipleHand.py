#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon = 0.8, maxHands=2)


while True:
    success, img = cap.read()
    hands, img = detector.findHands(img) #with draw
    #hands = detector.findHands(img, draw = False) #no draw
    
    if hands:
        # Hand 1
        hand1 = hands[0]
        lmList1 = hand1["lmList"] #List of 21 lendmarks points
        bbox1 = hand1["bbox"] #bounding box info x,y,w,h
        centerPoint1 = hand1["center"] #center of the hand cx, cy
        handType1 = hand1["type"] #handtype left or right
        
        #print(len(lmList1), lmList1) #lmlist 출력
        #print(bbox1) #bounding box 출력
        #print(centerPoint1) #centerpoint 출력
        #print(handType1) #왼손 오른손 출력 지금 hand1이라 한쪽 손만 출력
        fingers1 = detector.fingersUp(hand1)
        #length, info, img = detector.findDistance(lmList1[8], lmList1[12], img) #with Draw
        #length, info = detector.findDistance(lmList1[8], lmList1[12]) # no Draw
        
        if len(hands)==2:
            hand2 = hands[1]
            lmList2 = hand2["lmList"] #List of 21 lendmarks points
            bbox2 = hand2["bbox"] #bounding box info x,y,w,h
            centerPoint2 = hand2["center"] #center of the hand cx, cy
            handType2 = hand2["type"] #handtype left or right
            
            fingers2 = detector.fingersUp(hand2)
            
            #print(fingers1, fingers2)
            length, info, img = detector.findDistance(lmList1[8][1:], lmList2[8][1:], img) #with Draw 두 손
            #length, info, img = detector.findDistance(centerPoints1, centerPoints2, img) #with Draw 두 손 중앙

    cv2.imshow("Image", img)
    cv2.waitKey(1)
    


# In[ ]:




