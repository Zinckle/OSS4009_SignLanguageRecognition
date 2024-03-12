import cv2
import mediapipe as mp
import time
import math
import numpy as np

from keras.datasets import mnist
from keras.src.utils import img_to_array
from tensorflow.keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
import tensorflow as tf

model = tf.keras.models.load_model('numberDetect.h5')

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=2,
                      min_detection_confidence=0.5,
                      min_tracking_confidence=0.5)
mpDraw = mp.solutions.drawing_utils

pTime = 0
cTime = 0
success, img = cap.read()
overlay = np.zeros([480,640,3],dtype=np.uint8)
start_time = 0
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    thumb = []
    pointer = []
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x *w), int(lm.y*h)
                #if id ==0:
                cv2.circle(img, (cx,cy), 3, (255,0,255), cv2.FILLED)
                if id == 4:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
                    thumb = [cx, cy]
                elif id == 8:
                    cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)
                    pointer = [cx, cy]

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    #print(math.dist(thumb, pointer))

    if(math.dist(thumb, pointer)) > 50:

        cv2.circle(overlay, (pointer[0], pointer[1]), 15, (255, 255, 255), cv2.FILLED)
        #cv2.imshow("overlay", overlay)
        start_time = time.time()

    if (time.time() - start_time > 1.5 and (time.time()-start_time != time.time())):
        roi = cv2.resize(overlay, (28, 28), interpolation=cv2.INTER_AREA)

        img_array = roi[:,:,1]/ 255.0  # Normalize pixel values to be between 0 and 1
        reshaped_array = np.expand_dims(np.expand_dims(img_array, axis=0), axis=-1)
        y = model.predict(reshaped_array)
        print(np.argmax(y[0]))
        overlay = np.zeros([480, 640, 3], dtype=np.uint8)
        start_time = 0
        cv2.putText(overlay, str(np.argmax(y[0])), (300, 3 00), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)


    #cv2.imshow("overlay", overlay)
    added_image = cv2.addWeighted(img, 1, overlay, 1, 0)
    cv2.imshow("Image", added_image)

    cv2.waitKey(1)