#!/usr/bin/env python
# coding: utf-8

# ## hand tracking(basic)
# 
# 이 모델은 mediapipe를 이용했기 때문에 가장 먼저 mediapipe를 설치한다.

# In[1]:


get_ipython().system(' pip install mediapipe')


# cv2와 mediapipe 그리고 실시간 영상의 fps를 구하기 위해 time을 불러온다.

# In[5]:


import cv2
import mediapipe as mp
import time

#나같은 경우 윈도우 환경에서 작업하기 때문에 0값을 넣어줬다.
cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0


while True:
    # 영상의 색이 제대로 나오기 위해 RGB로 변경을 해줘야 한다.
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                print(id, cx, cy) 
                # 이제 영상 속 손에 표시될 원의 색과 크기를 설정해준다.
                if id == 0:
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
                
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
            
   # 실시간 fps를 표현하기 위한 명령어
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    # 표시된 fps의 글자 스타일과 크기, 위치와 색을 설정해준다.
    cv2.putText(img,str((fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,
                (255,0,255),3)
            
    cv2.imshow("Image",img)
    cv2.waitKey(1)


# ! [image.png](attachment:image.png)

# In[ ]:




