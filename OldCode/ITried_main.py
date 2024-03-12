import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
import csv
import tensorFlowFunctions as tff

# preapre handwritten digits


test_data = "Dataset/sign_mnist_test/sign_mnist_test.csv"
train_data = "Dataset/sign_mnist_train/sign_mnist_train.csv"


# extract the test data into a usable form


    X_num = np.array(testData)
    y_num = np.array(testLabels)


trainLabels = []
trainData = []
with open(train_data, newline='') as csvfile:
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
        trainLabels.append(int(rowData[0]))
        newImage = []
        for i in range(1, 29):
            newImage.append(np.array(rowData[1 + ((i - 1) * 28):1 + (i * 28)]).astype(int))
        trainData.append(newImage)
trainData = np.array(trainData)
trainLabels = np.array(trainLabels)

# (X_num, y_num), _ = tf.keras.datasets.mnist.load_data()
X_num = np.expand_dims(X_num, axis=-1).astype(np.float32) / 255.0

grid_size = 16  # image_size / mask_size

# test
X, y = tff.make_data(size=1)
tff.show_predict(X[0], y[0])

x, x_input = tff.buildNN()

# ---


model = tf.keras.models.Model(x_input, x)
model.summary()

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


opt = tf.keras.optimizers.Adam(learning_rate=0.003)
model.compile(loss=loss_func, optimizer=opt)


def preview(numbers=None, threshold=0.1):
    X, y = make_data(size=1)
    print("expected: ", y)
    y = model.predict(X)
    print("found: ", y)
    show_predict(X[0], y[0], threshold=threshold)


preview()

batch_size = 32
X_train, y_train = make_data(size=batch_size * 100)

trainData = np.array(trainData)
trainLabels
model.fit(X_train, y_train, batch_size=batch_size, epochs=5, shuffle=True)
preview()
model.summary()

loss, acc = model.evaluate(testData, testLabels, verbose=2)
print("Restored model, accuracy: {:5.2f}%".format(100 * acc))

preview(threshold=0.7)
