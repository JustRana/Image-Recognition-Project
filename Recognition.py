import cv2

img = cv2.imread('images/friends-581753_640.jpg')
#Initialize an empty list for class names
classnames = []
#Recall file thing
classfile = 'files/thing.names'

#Open and read the class file
with open(classfile, 'rt') as a:
    classnames = a.read().rstrip('\n').split('\n')

#Set the paths to the pre-trained model and the configuration file
p = 'files/frozen_inference_graph.pb'
v = 'files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

#Load the pre-trained model
net = cv2.dnn_DetectionModel(p, v)

#Set input size, scale, mean, and swapRB for the model
net.setInputSize(320, 230)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

#Run object detection on the image
classIds, confs, bbox = net.detect(img, confThreshold=0.5)

#Draw bounding boxes and add labels for detected objects
for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
    labelSize, baseLine = cv2.getTextSize(classnames[classId-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    left = box[0]
    top = max(box[1], labelSize[1] + 10)
    right = left + labelSize[0]
    bottom = top + labelSize[1] - 10
    cv2.rectangle(img, (left - 1, top - labelSize[1] - 10), (right + 1, bottom + 1), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, classnames[classId-1], (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0 , 0 , 0), 1)
    
   
#Display the image with object detection results
cv2.imshow('Image Recognition', img)
cv2.waitKey(0)

