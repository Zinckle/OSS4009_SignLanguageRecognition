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


def distance(point1, point2):
    """Calculate distance between two points."""
    return math.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)


def midpoint(point1, point2):
    """Calculate midpoint between two points."""
    return ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2)


def make_square(point1, point2):
    """Create a square with sides equal to the longest side of the original rectangle."""
    d = distance(point1, point2)
    side_length = d  # Length of the side of the square

    # Determine the midpoint of the original rectangle
    mid = midpoint(point1, point2)

    # Calculate the coordinates of the top-left and bottom-right points of the square
    top_left = (mid[0] - side_length / 2, mid[1] - side_length / 2)
    bottom_right = (mid[0] + side_length / 2, mid[1] + side_length / 2)

    return top_left, bottom_right


def extract_region(image, point1, point2):
    # Calculate top-left and bottom-right coordinates
    x1, y1 = point1
    x2, y2 = point2
    top_left = (min(x1, x2), min(y1, y2))
    bottom_right = (max(x1, x2), max(y1, y2))

    # Extract the region of interest (ROI)
    roi = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    return roi


def getLetter(result):
    try:
        if result > 8:
            result += 1
        output = chr(96 + result[0])
        return output
    except:
        return "Error"


model = tf.keras.models.load_model('numberDetect.h5')
model1 = tf.keras.models.load_model('sign_minst_cnn_50_Epochs.h5')
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
overlay = np.zeros([480, 640, 3], dtype=np.uint8)
start_time = 0
while True:
    success, img1 = cap.read()
    img = cv2.flip(img1, 1)
    readImg = cv2.flip(img1, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    thumb = []
    pointer = []

    g_x, g_y, l_x, l_y = 0, 0, 0, 0

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)

                if id == 0:
                    g_x = cx
                    g_y = cy
                    l_x = cx
                    l_y = cy
                else:
                    g_x = g_x if g_x > cx else cx
                    g_y = g_y if g_y > cy else cy
                    l_x = l_x if l_x < cx else cx
                    l_y = l_y if l_y < cy else cy

                cv2.circle(img, (cx, cy), 3, (255, 0, 255), cv2.FILLED)
                if id == 4:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
                    thumb = [cx, cy]
                elif id == 8:
                    cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)
                    pointer = [cx, cy]

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

    top_left, bottom_right = make_square((g_x, g_y), (l_x, l_y))

    roi = extract_region(readImg, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])))

    if roi.any():
        # cv2.rectangle(readImg, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])),(0, 0, 255), 5)
        cv2.imshow("test", roi)
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        # cv2.imshow("test", roi)
        roi = roi.reshape(1, 28, 28, 1)
        predict_x = model1.predict(roi, 1, verbose=0)
        print("predict_x: ", predict_x)
        result = np.argmax(predict_x, axis=1)
        print("result: ", result)
        cv2.putText(img, getLetter(result), (300, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

    if ((math.dist(thumb, pointer)) > 50) and False:
        cv2.circle(overlay, (pointer[0], pointer[1]), 15, (255, 255, 255), cv2.FILLED)
        # cv2.imshow("overlay", overlay)
        start_time = time.time()

    if (time.time() - start_time > 1.5 and (time.time() - start_time != time.time()) and False):
        roi = cv2.resize(overlay, (28, 28), interpolation=cv2.INTER_AREA)

        img_array = roi[:, :, 1] / 255.0  # Normalize pixel values to be between 0 and 1
        reshaped_array = np.expand_dims(np.expand_dims(img_array, axis=0), axis=-1)
        y = model.predict(reshaped_array)
        print(np.argmax(y[0]))
        overlay = np.zeros([480, 640, 3], dtype=np.uint8)
        start_time = 0
        cv2.putText(overlay, str(np.argmax(y[0])), (300, 300), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # cv2.imshow("overlay", overlay)
    added_image = cv2.addWeighted(img, 1, overlay, 1, 0)
    cv2.imshow("Image", added_image)

    cv2.waitKey(1)
