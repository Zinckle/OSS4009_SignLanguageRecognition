import cv2
import mediapipe as mp
import time
import math
import numpy as np
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
    top_left = ((mid[0] - side_length / 2) * 0.9, (mid[1] - side_length / 2) * 0.95)
    bottom_right = ((mid[0] + side_length / 2) * 1.1, (mid[1] + side_length / 2) * 1.05)

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
    signLetters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y']
    return signLetters[result[0]]


def greatest_outlier(points):
    if len(points) < 3:
        return None  # Cannot calculate outlier for less than 3 points
    max_distance = 0
    outlier = None
    for i in range(len(points)):
        distances = []
        for j in range(len(points)):
            if i != j:
                distances.append(distance(points[i], points[j]))
        median_distance = sorted(distances)[len(distances) // 2]
        if median_distance > max_distance:
            max_distance = median_distance
            outlier = points[i]
    return outlier


model = tf.keras.models.load_model('lmnist.h5')
model1 = tf.keras.models.load_model('smnist.h5')
cap = cv2.VideoCapture(0)
letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
           'U', 'V', 'W', 'X', 'Y', 'Z']

#letters = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
#           'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
#           'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
#           'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'd', 'e',
#           'f', 'g', 'h', 'n', 'q', 'r', 't']
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False,
                      max_num_hands=1,
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
    pinky = []
    pointer = []

    g_x, g_y, l_x, l_y = 0, 0, 0, 0
    pointArray = []
    outlier = [0,0]
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
                    #make point array here so we dont include id=0 as it is often the default outlier
                    pointArray.append((cx, cy))

                if id == 20:
                    cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)
                    pinky = (cx, cy)
                elif id == 8:
                    cv2.circle(img, (cx, cy), 7, (0, 255, 0), cv2.FILLED)
                    pointer = (cx, cy)

            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            outlier = greatest_outlier(pointArray)
    top_left, bottom_right = make_square((g_x, g_y), (l_x, l_y))
    roi = extract_region(readImg, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])))

    if roi.any():
        cv2.rectangle(img, (int(top_left[0]), int(top_left[1])), (int(bottom_right[0]), int(bottom_right[1])),(0, 0, 255), 5)
        # roi = remove(roi)

        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
       # roi = cv2.flip(overlayROI, 1)
        #roi = np.rot90(overlayROI)
        cv2.imshow("test", roi)
        roi = roi[:, :] / 255.0
        roi = roi.reshape(1, 28, 28, 1)
        predict_x = model1.predict(roi, 1, verbose=0)
        result = np.argmax(predict_x, axis=1)
        cv2.putText(img, getLetter(result), (10, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)

    if ((outlier == pointer) or (outlier == pinky)):
        cv2.circle(overlay, (outlier[0], outlier[1]), 15, (255, 255, 255), cv2.FILLED)
        # cv2.imshow("overlay", overlay)
        start_time = time.time()

    if (time.time() - start_time > 1.5 and (time.time() - start_time != time.time())):
        cv2.imshow("ove", overlay)
        overlayROI = cv2.resize(overlay, (28, 28), interpolation=cv2.INTER_AREA)
        overlayROI = cv2.flip(overlayROI, 1)
        overlayROI = np.rot90(overlayROI)
        #overlayROI = np.rot90(overlayROI)
        cv2.imshow("overlay", overlayROI)
        img_array = overlayROI[:, :, 1] / 255.0  # Normalize pixel values to be between 0 and 1
        reshaped_array = img_array .reshape(1, 28, 28, 1)

        #cv2.imshow("overlay", reshaped_array)
        y = model.predict(reshaped_array)
        print(np.argmax(y[0]))
        print(letters[np.argmax(y[0])])
        overlay = np.zeros([480, 640, 3], dtype=np.uint8)
        start_time = 0
        cv2.putText(overlay, str(letters[np.argmax(y[0])]), (10, 150), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 3)

    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, str(int(fps)), (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

    # cv2.imshow("overlay", overlay)
    added_image = cv2.addWeighted(img, 1, overlay, 1, 0)
    cv2.imshow("Image", added_image)

    cv2.waitKey(1)
