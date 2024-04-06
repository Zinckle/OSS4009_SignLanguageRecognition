import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
import csv
import tensorFlowFunctions as tff

idx_p = [0]
idx_bb = [1, 2, 3, 4]
idx_cls = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]

@tf.function
def loss_bb(y_true, y_pred):
    y_true = tf.gather(y_true, idx_bb, axis=-1)
    y_pred = tf.gather(y_pred, idx_bb, axis=-1)

    loss = tf.keras.losses.mean_squared_error(y_true, y_pred)
    return tf.reduce_mean(loss[loss > 0.0])
@tf.function
def loss_p(y_true, y_pred):
    y_true = tf.gather(y_true, idx_p, axis=-1)
    y_pred = tf.gather(y_pred, idx_p, axis=-1)

    loss = tf.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_sum(loss)
@tf.function
def loss_cls(y_true, y_pred):
    y_true = tf.gather(y_true, idx_cls, axis=-1)
    y_pred = tf.gather(y_pred, idx_cls, axis=-1)

    loss = tf.losses.binary_crossentropy(y_true, y_pred)
    return tf.reduce_sum(loss)
@tf.function
def loss_func(y_true, y_pred):
    return loss_bb(y_true, y_pred) + loss_p(y_true, y_pred) + loss_cls(y_true, y_pred)
# define a video capture object
vid = cv2.VideoCapture(0)

model = tf.keras.models.load_model('my_model', custom_objects={'loss_func': loss_func})
model.summary()

while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()

    #crop
    frame = frame[:, 80:580]
    #dowsample
    down_width = 128
    down_height = 128
    down_points = (down_width, down_height)
    frame = cv2.resize(frame, down_points, interpolation=cv2.INTER_LINEAR)

    #cv2.imshow('frame', frame)


    # the 'q' button is set as the
    # quitting button you may use anyq
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    print(frame.shape)

    #X_num = np.expand_dims(frame, axis=-1).astype(np.float32) / 255.0

    frame = frame / 255.0

    canvasGrid = np.zeros((1, 128, 128, 3), dtype=np.float32)

    for i in range(1):
        canvasGrid[i] = frame

    X = np.clip(canvasGrid, 0.0, 1.0)

    y = model.predict(X)
    #print("found: ", y)
    tff.show_predict(X[0], y[0], threshold=0.9)


# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()