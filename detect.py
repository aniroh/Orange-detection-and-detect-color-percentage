#dependencies
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Orange detector and percentage of colors present')

# Add the arguments
parser.add_argument("-img",'--image',type=str,required=True ,help='Path to image file.')

# Execute the parse_args() method
args = vars(parser.parse_args())
path_img=args["image"]

# Load Yolo
# path1="drive/MyDrive/crowd counting/"
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
img = cv2.imread(path_img) #this line of code for testing purposes, input image
img = cv2.resize(img, None, fx=1.0, fy=1.0)
height, width, channels = img.shape

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

#color info 
color_dict_HSV = { 
              'white': [[180, 18, 255], [0, 0, 231]],
              'red': [[180, 255, 255], [159, 50, 70]],
              'green': [[75, 255, 255], [36, 50, 70]],
              'blue': [[128, 255, 255], [76, 50, 70]],
              'yellow': [[35, 255, 255], [25, 150, 150]],
              'orange': [[24, 255, 255], [0, 50, 70]]}
color1=['#ffffff','#ff0000','#008000','#0000ff','#ffff00','#ffa500']

# Showing informations on the screen
class_ids = []
confidences = []
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:] #from 85 elements, we select from 5th to 85th elements, i think this refers to 80 classes of coco.names
        class_id = np.argmax(scores) #selecting a class with max score
        confidence = scores[class_id]
        if confidence > 0.3 and class_id==49: #only select orange which has id 49
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            # Rectangle coordinates
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

def show():
    col1=[] 
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)): 
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 3)
            
            
            roi = img[y:y+h, x:x+w]
            notr = cv2.bitwise_not(roi[:,:,0])
            ret,thresh1 = cv2.threshold(notr,127,255,cv2.THRESH_BINARY)
            notres=cv2.merge([thresh1,thresh1,thresh1])
            res = cv2.bitwise_and(notres,roi)
            # cv2.imshow(res)
            hsv = cv2.cvtColor(res, cv2.COLOR_BGR2HSV)
            col=[0]*6
            for j in range(hsv.shape[0]):
              for i in range(hsv.shape[1]):
                h,s,v=hsv[j,i]
                if (180>=h and h>=0) and (255>=s and s>=0) and (30>=v and v>=0):
                  continue
                else:
                  total=0
                  for key,value in color_dict_HSV.items():
              
                    if (value[0][0]>=h and h>=value[1][0]) and (value[0][1]>=s and s>=value[1][1]) and (value[0][2]>=v and v>=value[1][2]):
                      col[total]+=1
                      break
                    total+=1
            
            col1.append(col)
    col1 = np.asarray(col1)
    col1=np.sum(col1, axis = 0)
    temp=0
    plt.figure(figsize = (8, 6))
    plt.pie(col1, labels = color_dict_HSV.keys(), colors = color1)
    plt.show()
    for i in color_dict_HSV.keys():
        print(i+"% :"+str(float(col1[temp]*100/sum(col1))))
        temp+=1         
    cv2.imshow("Image",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.3, 0.5)
show()