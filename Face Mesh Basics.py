#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture("video2.mp4") 
pTime = 0


mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
faceMesh = mpFaceMesh.FaceMesh(max_num_faces = 1)
drawSpec = mpDraw.DrawingSpec(thickness = 1, circle_radius = 2)


while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    if results.multi_face_landmarks:
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(img, faceLms, mpFaceMesh.FACEMESH_CONTOURS, drawSpec, drawSpec)
            #FACE_CONNECTIONS에서 FACEMESH_CONTOURS로 바뀜
            
            for id, lm in enumerate(faceLms.landmark):
                #print(lm)
                ih, iw, ic = img.shape
                x, y = int(lm.x*iw), int(lm.y*ih)
                print(id,x,y)
    
        
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS : {int(fps)}',(20,70),cv2.FONT_HERSHEY_PLAIN,
               3,(0,255,0), 3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)


# In[ ]:


def __init__(self,
            static_image_mode = False,
            max_num_faces = 1,
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5):


# In[2]:


def draw_landmarks(
    image:np.ndarray,
    landmark_list: landmark_pb2.NormalizedLandmarkList,
    connections: List[Tuple[int, int]] = None,
    landmark_drawing_spec : DrawingSpec = DrawingSpec(color = RED_COLOR),
    connection_drawing_spec : DrawingSpec =DrawingSpec()):


# In[ ]:




