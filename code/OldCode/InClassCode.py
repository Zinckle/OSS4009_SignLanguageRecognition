

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from matplotlib import pyplot
import csv
import numpy as np

print(tf.version.VERSION)
test_data = "Dataset/sign_mnist_test/sign_mnist_test.csv"
train_data = "Dataset/sign_mnist_train/sign_mnist_train.csv"

# extract the test data into a usable form
testLabels = []
testData = []
with open(test_data, newline='') as csvfile:
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
        testLabels.append(int(rowData[0]))
        newImage = []
        for i in range(1, 29):
            newImage.append(np.array(rowData[1 + ((i - 1) * 28):1 + (i * 28)]).astype(int))
        testData.append(newImage)



# extract the training data into a usable form
trainLabels = []
trainData = []
with open(train_data, newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    escape = 0
    for row in reader:
        if escape > 2999:
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

# y = label
# x = data
x_train = np.array(trainData)
y_train = np.array(trainLabels)
x_test = np.array(testData)
y_test = np.array(testLabels)
#(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize pixel values to between 0 and 1
x_train, x_test = x_train[..., tf.newaxis], x_test[..., tf.newaxis]  # Add a channel dimension
y_train, y_test = to_categorical(y_train, 25), to_categorical(y_test, 25)  # One-hot encode labels

# Plot the 100th to the 109th training samples of the MNIST dataset
for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(x_train[i], cmap=pyplot.get_cmap('gray'))

pyplot.show()


# Function to create a neural network with specified loss and activation functions
def create_neural_network(loss_function, activation_function):
    model = models.Sequential()
    model.add(layers.Flatten(input_shape=(28, 28, 1)))
    model.add(layers.Dense(128, activation=activation_function))
    model.add(layers.Dense(25, activation='softmax'))

    model.compile(optimizer='adam', loss=loss_function, metrics=['accuracy'])
    return model


# Function to train and evaluate a model
def train_evaluate_model(model, epochs=50):
    history = model.fit(x_train, y_train, epochs=epochs, validation_data=(x_test, y_test), verbose=0)
    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
    return history, test_loss, test_acc


# Define lists of loss functions and activation functions to compare
loss_functions = ['mean_squared_error']
activation_functions = ['relu']

# Create a plot for each combination of loss and activation functions
for loss_function in loss_functions:
    for activation_function in activation_functions:
        model = create_neural_network(loss_function, activation_function)
        history, test_loss, test_acc = train_evaluate_model(model)

        # Plot training loss over epochs
        plt.plot(history.history['loss'], label=f'{loss_function}_{activation_function}')

    plt.title(f'Training Loss Over Epochs, {loss_function}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
