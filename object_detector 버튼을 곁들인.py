#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system(' pip install cx_freeze')


# In[ ]:


import cv2
from gui_buttons import Buttons

#initialize buttons
button = Buttons()
button.add_button("person", 20, 20)
button.add_button("cell phone", 20, 100)
button.add_button("mouse", 20, 180)
button.add_button("keyboard", 20, 260)
button.add_button("book", 20, 340)


# Opencv DNN
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size = (320 ,320), scale = 1/255)

#Load class lists
classes =[]
with open("dnn_model/classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
        
print("Objects list")
print(classes)


        

#Initialize camera
cap = cv2.VideoCapture(0) #0이 기본 웹캠 1이 두번째 웹캠
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# FULL HD 1920 X 1080 




def click_button(event, x, y, flags, params):
    global button_person
    if event == cv2.EVENT_LBUTTONDOWN:
        button.button_click(x, y)


#create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", click_button)


while True:
    #get Frame
    ret, frame = cap.read()
    
    # Get active buttons list
    active_buttons = button.active_buttons_list()
    print("Active buttons", active_buttons)

    #object detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        (x, y, w, h) = bbox
        class_name = classes[class_id]
        
        if class_name in active_buttons:               
            cv2.putText(frame, class_name, (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (200,0,50), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (200,0,50), 3)
        
        
    #Display buttons
    button.display_buttons(frame)
    
    
    cv2.imshow("Frame",frame)
    key = cv2.waitKey(1)  #0일경우 눌러야 한장한장 움직인다. 1이여야 실시간 영상이 된다.
    if key == 27:
        break
        
cap.release()
cv2.destoryAllWindows()


# In[ ]:




