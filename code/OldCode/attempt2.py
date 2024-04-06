import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import cv2
import matplotlib.pyplot as plt
import csv
import tensorFlowFunctions as tff

test_data = "Dataset/sign_mnist_test/sign_mnist_test.csv"
train_data = "Dataset/sign_mnist_train/sign_mnist_train.csv"


# extract the test data into a usable form
X_num, y_num = tff.extractFromCSV(train_data)

# (X_num, y_num), _ = tf.keras.datasets.mnist.load_data()
X_num = np.expand_dims(X_num, axis=-1).astype(np.float32) / 255.0

grid_size = 16  # image_size / mask_size

# test
#X, y = tff.make_data(X_num, y_num,size=1)
#tff.show_predict(X[0], y[0])

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
    X, y = tff.make_data(X_num, y_num, size=1)
    #print("expected: ", y)
    y = model.predict(X)
    #print("found: ", y)
    tff.show_predict(X[0], y[0], threshold=threshold)


batch_size = 32

# Create training data
X_train, y_train = tff.make_data(X_num, y_num, size=batch_size * 100)

# Load and preprocess test data
testData, testLabels = tff.extractFromCSV(test_data)
testData = np.expand_dims(testData, axis=-1).astype(np.float32) / 255.0
X_test, y_test = tff.make_data(testData, testLabels, size=batch_size * 100)

# Ensure that the model has been defined before fitting
# model = ...

# Fit the model to the training data
#history = model.fit(X_train, y_train, batch_size=batch_size, epochs=5, shuffle=True)
model = tf.keras.models.load_model('my_model', custom_objects={'loss_func': loss_func})
#model.save('my_model.keras')
#model.save('my_model', save_format='tf')
#model.save('my_model.h5')
# Evaluate the model on the test data
#test_result, rtheth = model.evaluate(X_test, y_test, verbose=2)

# Print or log the test accuracy

preview()
print(model.summary())

preview(threshold=0.7)
