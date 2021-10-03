from cv2 import cv2
import numpy as np
import time
from datetime import date
import logging
from datetime import datetime


#load video
cap = cv2.VideoCapture("./karhutla.mp4")
confThreshold = 0.25
nmsThreshold = 0.3

prev_frame_time = 0
new_frame_time = 0

#ekstrak file coco.names
classFile = './data/coco.names'
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

#load yolo
cfg = './cfg/custom-yolov4-tiny-detector.cfg'
weights = './backup/custom-yolov4-tiny-detector_final.weights'
net = cv2.dnn.readNetFromDarknet(cfg,weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def findObject(outputs, img):
        hT, wT , cT = img.shape
        bbox = []
        classIds = []
        confs = []

        for output in outputs :
                for det in output :
                        scores = det[5:]
                        classId = np.argmax(scores)
                        confidence = scores[classId]
                        if confidence > confThreshold:
                                w,h = int(det[2]*wT) , int(det[3]*hT)
                                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                                bbox.append([x,y,w,h])
                                classIds.append(classId)
                                confs.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(bbox,confs, confThreshold, nmsThreshold)

        for i in indices :
               i = i[0]
               box = bbox[i]
               x,y,w,h = box[0], box[1], box[2], box[3]
               if f'{classNames[classIds[i]].upper()}' == 'API':
                       cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,255), 2)
                       cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%' , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,255),2)
               elif f'{classNames[classIds[i]].upper()}' == 'ASAP':
                        cv2.rectangle(img, (x,y), (x+w, y+h), (240, 159, 10), 2)
                        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%' , (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 159, 10),2)
               if f'{classNames[classIds[i]].upper()}' == 'API' or f'{classNames[classIds[i]].upper()}' == 'ASAP':
                       logging.basicConfig(filename="logfile.log", level=logging.INFO)
                       logging.info(datetime.now().strftime("%d-%m-%Y : %H_%M_%S") + ':' + f'{classNames[classIds[i]].upper()}:{int(confs[i]*100)}%')
                       print("terdeteksi")
        
while True :
    success, img = cap.read()
    h, w , c = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (320, 320),[0,0,0],1,crop=False)
    net.setInput(blob)
    font = cv2.FONT_HERSHEY_SIMPLEX 
    new_frame_time = time.time() 
    fps = 1/(new_frame_time-prev_frame_time) 
    prev_frame_time = new_frame_time 
  
    # converting the fps into integer 
    fps = int(fps) 
  
    # converting the fps to string so that we can display it on frame 
    # by using putText function 
    fps = str(fps) 
    cv2.putText(img,fps+' fps', (w-50, h-10), font, 0.5, (100, 255, 0), 1, cv2.LINE_AA) 
    LayerNames = net.getLayerNames()
    outputNames = [LayerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    #print (outputNames)
    #print(net.getUnconnectedOutLayers())

    outputs = net.forward(outputNames)

    findObject(outputs, img)
    
    cv2.imshow("video", img)
    key = cv2.waitKey(1)
     # 'q' to stop
    if key == ord('q'):
        break
