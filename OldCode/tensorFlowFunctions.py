import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
import csv

def extractFromCSV(path):
    labels = []
    data = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        escape = 0
        for row in reader:
            if escape > 999:
                break
            if escape == 0:
                escape += 1
                continue
            escape += 1
            rowData = row[0].split(",")
            labels.append(int(rowData[0]))
            newImage = []
            for i in range(1, 29):
                newImage.append(np.array(rowData[1 + ((i - 1) * 28):1 + (i * 28)]).astype(int))
            data.append(newImage)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels

def make_numbers(canvasGrid, maskGrid, data, labels, grid_size=16):
    for _ in range(3):
        # pickup random index
        idx = np.random.randint(len(data))

        # class of digit
        kls = labels[idx]

        # random position for digit
        px, py = np.random.randint(0, 100), np.random.randint(0, 100)

        # digit belong which mask position
        mx, my = (px + 14) // grid_size, (py + 14) // grid_size
        channels = maskGrid[my][mx]

        # prevent duplicated problem
        if channels[0] > 0:
            continue

        channels[0] = 1.0
        channels[1] = px - (mx * grid_size)  # x1
        channels[2] = py - (my * grid_size)  # y1
        channels[3] = 28.0  # x2, in this demo image only 28 px as width
        channels[4] = 28.0  # y2, in this demo image only 28 px as height
        channels[5 + kls] = 1.0

        # put digit in X
        canvasGrid[py:py + 28, px:px + 28] += data[idx]


def make_data(data, labels, size=64):
    canvasGrid = np.zeros((size, 128, 128, 3), dtype=np.float32)
    maskGrid = np.zeros((size, 8, 8, 32), dtype=np.float32)
    for i in range(size):
        make_numbers(canvasGrid[i], maskGrid[i], data, labels)

    X = np.clip(canvasGrid, 0.0, 1.0)
    return X, maskGrid


def get_color_by_probability(p):
    if p < 0.3:
        return (1., 0., 0.)
    if p < 0.7:
        return (1., 1., 0.)
    return (0., 1., 0.)


def show_predict(X, y, threshold=0.1, grid_size=32):
    X = X.copy()
    for mx in range(8):
        for my in range(8):
            channels = y[my][mx]
            prob, x1, y1, x2, y2 = channels[:5]

            # if prob < threshold we won't show any thing
            if prob < threshold:
                continue

            color = get_color_by_probability(prob)
            # bounding box
            px, py = (mx * grid_size) + x1, (my * grid_size) + y1
            cv2.rectangle(X, (int(px), int(py)), (int(px + x2), int(py + y2)), color, 1)

            # label
            cv2.rectangle(X, (int(px), int(py - 10)), (int(px + 12), int(py)), color, -1)
            kls = np.argmax(channels[5:])
            cv2.putText(X, f'{kls}', (int(px + 2), int(py - 2)), cv2.FONT_HERSHEY_PLAIN , 0.5, (0.0, 0.0, 0.0))

    cv2.imshow('X', X)
    #plt.imshow(X)
    #plt.show()

def buildNN():
    x = x_input = layers.Input(shape=(128, 128, 3))

    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)  # size: 64x64

    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)  # size: 64x64

    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)  # size: 32x32

    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)  # size: 16x16

    x = layers.Conv2D(32, kernel_size=3, padding='same', activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.BatchNormalization()(x)  # size: 8x8x

    x_prob = layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid', name='x_prob')(x)
    x_boxes = layers.Conv2D(4, kernel_size=3, padding='same', name='x_boxes')(x)
    x_cls = layers.Conv2D(10, kernel_size=3, padding='same', activation='sigmoid', name='x_cls')(x)

    # ---

    gate = tf.where(x_prob > 0.5, tf.ones_like(x_prob), tf.zeros_like(x_prob))
    x_boxes = x_boxes * gate
    x_cls = x_cls * gate

    # ---

    x = layers.Concatenate()([x_prob, x_boxes, x_cls])

    return x, x_input
